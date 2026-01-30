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

    def forward(self, response_mask: torch.Tensor) -> torch.Tensor:
        """Compute phi weights from response mask.
        
        Args:
            response_mask: (batch_size, seq_len) binary mask
            
        Returns:
            phi: (batch_size,) weights normalized to mean=1
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
        phi = phi.clamp(self.config.clip_min, self.config.clip_max)
        
        return phi


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
        """Initialize beta_raw so softplus(beta_raw) = target."""
        if target > 20:
            self._beta_raw.data.fill_(target)
        else:
            self._beta_raw.data.fill_(math.log(math.exp(target) - 1))

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self._beta_raw)

    def forward(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute TIS weights from log probability divergence.
        
        Args:
            log_probs: (batch_size, seq_len) current policy log probs (detached)
            ref_log_probs: (batch_size, seq_len) reference policy log probs
            response_mask: (batch_size, seq_len) binary mask
            
        Returns:
            W: (batch_size, seq_len) weights normalized per sequence
        """
        # Compute divergence (detach to prevent gradient through diff)
        divergence = torch.abs(log_probs.detach() - ref_log_probs.detach())
        
        # Apply learnable power
        w_raw = (divergence + 1e-8).pow(self.beta)  # (batch_size, seq_len)
        
        # Mask and normalize per sequence
        w_masked = w_raw * response_mask
        seq_sums = w_masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        seq_lens = response_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        
        # Normalize so each sequence sums to its length (preserves gradient energy)
        W = w_masked / seq_sums * seq_lens
        
        # Clip for stability
        W = W.clamp(self.config.clip_min, self.config.clip_max) * response_mask
        
        return W
