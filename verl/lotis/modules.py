"""LOTIS modules for learnable sequence and token weighting."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.lotis.config import LengthWeightConfig, TISWeightConfig


class RBFLengthWeightModule(nn.Module):
    """Computes sequence-level weights using RBF kernels on z-score normalized lengths.
    
    phi(z) = exp(sum_k alpha_k * K(z, mu_k)) where K is Gaussian kernel.
    Output is normalized to preserve gradient energy.
    """

    def __init__(self, config: LengthWeightConfig):
        super().__init__()
        self.config = config
        
        # Learnable alpha parameters
        self.alphas = nn.Parameter(
            torch.full((config.num_rbf_kernels,), config.alpha_init)
        )
        
        # Fixed RBF centers
        if config.rbf_centers is None:
            centers = torch.linspace(-2.0, 2.0, config.num_rbf_kernels)
        else:
            centers = torch.tensor(config.rbf_centers, dtype=torch.float32)
        self.register_buffer("centers", centers)
        self.register_buffer("bandwidth_sq_2", torch.tensor(2.0 * config.rbf_bandwidth ** 2))

    def forward(self, response_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute phi weights from response mask.
        
        Args:
            response_mask: (batch_size, seq_len) binary mask
            
        Returns:
            phi: (batch_size,) weights normalized to mean=1
            metrics: dict of metrics
        """
        # Get sequence lengths
        lengths = response_mask.sum(dim=-1).float()  # (batch_size,)
        
        # Z-score normalization within batch
        mu = lengths.mean()
        sigma = lengths.std().clamp(min=1e-8)
        z = (lengths - mu) / sigma  # (batch_size,)
        
        # RBF kernels: K(z, mu_k) = exp(-(z - mu_k)^2 / (2 * sigma^2))
        z_exp = z.unsqueeze(-1)  # (batch_size, 1)
        kernels = torch.exp(-(z_exp - self.centers) ** 2 / self.bandwidth_sq_2)  # (batch_size, K)
        
        # phi = exp(sum_k alpha_k * K(z, mu_k))
        weighted_sum = (self.alphas * kernels).sum(dim=-1)  # (batch_size,)
        phi_raw = torch.exp(weighted_sum)
        
        # Normalize to mean=1 (preserves gradient energy)
        phi = phi_raw / phi_raw.mean().clamp(min=1e-8)
        
        # Clip for stability
        phi = phi.clamp(self.config.phi_clip_min, self.config.phi_clip_max)
        
        metrics = {
            "lotis/response_length_mean": mu.item(),
            "lotis/response_length_std": sigma.item(),
            # Center kernel weight (assuming centers are sorted and middle one is ~0)
            "lotis/rbf_kernel_weight_center": self.alphas[self.config.num_rbf_kernels // 2].item(),
            "lotis/rbf_alphas_mean": self.alphas.mean().item(),
            "lotis/rbf_phi_mean": phi.mean().item(),
            "lotis/rbf_phi_std": phi.std().item(),
            "lotis/rbf_phi_max": phi.max().item(),
            "lotis/rbf_phi_min": phi.min().item(),
        }
        
        # Log individual alphas (optional, maybe limit if K is large)
        for i, alpha in enumerate(self.alphas):
            metrics[f"lotis/rbf_alpha_{i}"] = alpha.item()
        
        return phi, metrics


class TISWeightModule(nn.Module):
    """Computes token-level importance weights based on policy divergence.
    
    w_t = |log pi_theta - log pi_ref|^beta
    Output is normalized per sequence to preserve gradient energy.
    """

    def __init__(self, config: TISWeightConfig):
        super().__init__()
        self.config = config
        
        # Learnable beta via softplus to ensure > 0
        self._beta_raw = nn.Parameter(torch.tensor(0.0))
        self._init_beta(config.beta_init)

    def _init_beta(self, target: float):
        """Initialize beta_raw so beta_max * tanh(softplus(beta_raw)) = target."""
        max_val = getattr(self.config, "beta_max", 3.0)
        # Clamp target to valid range (0, max_val)
        target = max(1e-6, min(max_val - 1e-6, target))
        # y = arctanh(target / max_val)
        ratio = target / max_val
        # arctanh(x) = 0.5 * log((1+x)/(1-x))
        y = 0.5 * math.log((1 + ratio) / (1 - ratio))
        # x = inv_softplus(y)
        if y > 20:
            self._beta_raw.data.fill_(y)
        else:
            val = math.exp(y) - 1
            val = max(1e-6, val)
            self._beta_raw.data.fill_(math.log(val))

    @property
    def beta(self) -> torch.Tensor:
        # Bounded beta: beta_max * tanh(softplus(beta_raw))
        # Prevents sparsity collapse, ensuring credit assignment to "scaffolding" tokens
        max_val = getattr(self.config, "beta_max", 3.0)
        return max_val * torch.tanh(F.softplus(self._beta_raw))

    def forward(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute TIS weights from log probability divergence.
        
        Args:
            log_probs: (batch_size, seq_len) current policy log probs (detached)
            ref_log_probs: (batch_size, seq_len) reference policy log probs
            response_mask: (batch_size, seq_len) binary mask
            
        Returns:
            W: (batch_size, seq_len) weights normalized per sequence
            metrics: dict of metrics
        """
        # Compute divergence (detach to prevent gradient through diff)
        divergence = torch.abs(log_probs.detach() - ref_log_probs.detach())
        
        # Clip divergence to prevent exploding weights/gradients from outliers
        # This ensures we still learn from large divergences (gradient flows through beta)
        # without numerical instability.
        if hasattr(self.config, "div_clip") and self.config.div_clip > 0:
            divergence = divergence.clamp(min=1e-6, max=self.config.div_clip)
        else:
            divergence = divergence.clamp(min=1e-6)
        
        # Apply learnable power
        w_raw = divergence.pow(self.beta)  # (batch_size, seq_len)
        
        # Mask and normalize per sequence
        w_masked = w_raw * response_mask
        seq_sums = w_masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        seq_lens = response_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        
        # Normalize so each sequence sums to its length (preserves gradient energy)
        W = w_masked / seq_sums * seq_lens
        
        # Clip for stability
        W = W.clamp(self.config.wt_clip_min, self.config.wt_clip_max) * response_mask
        
        # KL divergence metrics
        kl_term = torch.exp(ref_log_probs) * (ref_log_probs - log_probs) * response_mask
        ref_kl = kl_term.sum() / response_mask.sum()
        
        # Weighted KL (weighted by TIS weights)
        # W is normalized per sequence, so we can just multiply
        ref_kl_weighted = (kl_term * W).sum() / response_mask.sum()

        w_valid = W[response_mask.bool()]
        metrics = {
            "lotis/tis_beta": self.beta.item(),
            "lotis/tis_weight_mean": w_valid.mean().item(),
            "lotis/tis_weight_std": w_valid.std().item(),
            "lotis/tis_weight_max": w_valid.max().item(),
            "lotis/tis_weight_min": w_valid.min().item(),
            "lotis/tis_sparsity_ratio": W.max().item() / (w_valid.mean().item() + 1e-6),
            "lotis/ref_kl_total": ref_kl.item(),
            "lotis/ref_kl_weighted": ref_kl_weighted.item(),
        }
        
        return W, metrics
