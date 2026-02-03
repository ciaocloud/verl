"""LOTIS modules for learnable sequence and token weighting."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.lotis.config import LengthWeightConfig, TokenWeightConfig


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


class KLTokenWeightModule(nn.Module):
    """Token weighting based on KL divergence between policy and reference model.
    
    w_t = |log pi_theta - log pi_ref|^gamma
    Output is normalized per sequence to preserve gradient energy.
    """

    def __init__(self, config: TokenWeightConfig):
        super().__init__()
        self.config = config
        
        # Learnable gamma via softplus to ensure > 0
        self._gamma_raw = nn.Parameter(torch.tensor(0.0))
        self._init_gamma(config.gamma_init)

    def _init_gamma(self, target: float):
        """Initialize gamma_raw so gamma_max * tanh(softplus(gamma_raw)) = target."""
        max_val = getattr(self.config, "gamma_max", 3.0)
        # Clamp target to valid range (0, max_val)
        target = max(1e-6, min(max_val - 1e-6, target))
        # y = arctanh(target / max_val)
        ratio = target / max_val
        # arctanh(x) = 0.5 * log((1+x)/(1-x))
        y = 0.5 * math.log((1 + ratio) / (1 - ratio))
        # x = inv_softplus(y)
        if y > 20:
            self._gamma_raw.data.fill_(y)
        else:
            val = math.exp(y) - 1
            val = max(1e-6, val)
            self._gamma_raw.data.fill_(math.log(val))

    @property
    def gamma(self) -> torch.Tensor:
        # Bounded gamma: gamma_max * tanh(softplus(gamma_raw))
        # Prevents sparsity collapse, ensuring credit assignment to "scaffolding" tokens
        max_val = getattr(self.config, "gamma_max", 3.0)
        return max_val * torch.tanh(F.softplus(self._gamma_raw))

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
        w_raw = divergence.pow(self.gamma)  # (batch_size, seq_len)
        
        # Mask and normalize per sequence
        w_masked = w_raw * response_mask
        seq_sums = w_masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        seq_lens = response_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        
        # Normalize so each sequence sums to its length (preserves gradient energy)
        W = w_masked / seq_sums * seq_lens
        
        # Clip for stability
        W = W.clamp(self.config.psi_clip_min, self.config.psi_clip_max) * response_mask
        
        # KL divergence metrics
        kl_term = torch.exp(ref_log_probs) * (ref_log_probs - log_probs) * response_mask
        ref_kl = kl_term.sum() / response_mask.sum()
        
        # Weighted KL (weighted by TIS weights)
        # W is normalized per sequence, so we can just multiply
        ref_kl_weighted = (kl_term * W).sum() / response_mask.sum()

        w_valid = W[response_mask.bool()]
        metrics = {
            "lotis/tis_gamma": self.gamma.item(),
            "lotis/tis_weight_mean": w_valid.mean().item(),
            "lotis/tis_weight_std": w_valid.std().item(),
            "lotis/tis_weight_max": w_valid.max().item(),
            "lotis/tis_weight_min": w_valid.min().item(),
            "lotis/tis_sparsity_ratio": W.max().item() / (w_valid.mean().item() + 1e-6),
            "lotis/ref_kl_total": ref_kl.item(),
            "lotis/ref_kl_weighted": ref_kl_weighted.item(),
        }
        
        return W, metrics

class MLPTokenWeightModule(nn.Module):
    """Computes token-level importance weights based on MLP over hidden states.
    
    psi_t = MLP(h_t)
    Output is normalized per sequence to preserve gradient energy.
    Symbol: psi (ψ)
    """
    
    def __init__(self, config: TokenWeightConfig, hidden_dim: int):
        super().__init__()
        self.config = config
        
        layers = []
        in_dim = hidden_dim
        
        activation_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "silu": nn.SiLU,
            "swish": nn.SiLU,
        }
        act_cls = activation_map.get(config.mlp_activation.lower(), nn.SiLU)
        
        # Pre-norm
        layers.append(nn.LayerNorm(hidden_dim))

        for _ in range(config.mlp_num_layers):
            layers.append(nn.Linear(in_dim, config.mlp_hidden_dim))
            layers.append(act_cls())
            in_dim = config.mlp_hidden_dim
            
        # Final projection to scalar weight (raw)
        layers.append(nn.Linear(in_dim, 1))
        # Ensure positive weights
        layers.append(nn.Softplus())
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize MLP weights for stable starting point."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # The last layer is at index -2 (before Softplus)
        last_linear = self.mlp[-2]
        nn.init.normal_(last_linear.weight, mean=0.0, std=0.001)
        # Softplus(0.5413) ≈ 1.0. This makes raw weights start at 1.0.
        nn.init.constant_(last_linear.bias, 0.5413)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute TIS weights from hidden states.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim) last layer hidden states
            response_mask: (batch_size, seq_len) binary mask
            
        Returns:
            W: (batch_size, seq_len) weights normalized per sequence
            metrics: dict of metrics
        """
        # Detach hidden states to prevent gradient flow to actor backbone
        # Cast to MLP's dtype (e.g., float32) to match LayerNorm parameters
        # (batch_size, seq_len, 1) -> (batch, seq_len)
        mlp_dtype = next(self.mlp.parameters()).dtype
        w_raw = self.mlp(hidden_states.detach().to(mlp_dtype)).squeeze(-1)
        
        # Mask and normalize per sequence
        w_masked = w_raw * response_mask
        seq_sums = w_masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        seq_lens = response_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        
        # Normalize so each sequence sums to its length (preserves gradient energy)
        psi = w_masked / seq_sums * seq_lens
        
        # Clip for stability
        psi = psi.clamp(self.config.psi_clip_min, self.config.psi_clip_max) * response_mask
        
        psi_valid = psi[response_mask.bool()]
        metrics = {
            "lotis/psi_weight_mean": psi_valid.mean().item(),
            "lotis/psi_weight_std": psi_valid.std().item(),
            "lotis/psi_weight_max": psi_valid.max().item(),
            "lotis/psi_weight_min": psi_valid.min().item(),
            "lotis/psi_raw_mlp_mean": w_masked[response_mask.bool()].mean().item(),
        }
        
        # Cast back to input dtype for consistency with other tensors in loss
        return psi.to(response_mask.dtype), metrics
