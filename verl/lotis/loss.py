"""LOTIS policy loss function."""

import torch
from typing import Optional, Dict, Any

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss


@register_policy_loss("lotis")
def compute_lotis_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config=None,
    # LOTIS weights (computed by actor's lotis module)
    phi_weights: Optional[torch.Tensor] = None,
    tis_weights: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Compute LOTIS policy loss with sequence and token weighting.
    
    Args:
        old_log_prob: (batch_size, seq_len) log probs from old policy
        log_prob: (batch_size, seq_len) log probs from current policy  
        advantages: (batch_size, seq_len) advantage estimates
        response_mask: (batch_size, seq_len) binary mask
        loss_agg_mode: aggregation mode (used when phi_weights is None)
        config: actor config with clip_ratio settings
        phi_weights: (batch_size,) sequence-level weights from RBF module
        tis_weights: (batch_size, seq_len) token-level weights from TIS module
        
    Returns:
        pg_loss: scalar policy gradient loss
        metrics: dict with pg_clipfrac, ppo_kl, etc.
    """
    clip_ratio = config.clip_ratio
    clip_ratio_low = getattr(config, "clip_ratio_low", None) or clip_ratio
    clip_ratio_high = getattr(config, "clip_ratio_high", None) or clip_ratio
    
    # Standard PPO ratio computation
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    
    # Apply TIS weights to ratio if provided
    if tis_weights is not None:
        weighted_ratio = ratio * tis_weights
        clipped_ratio = torch.clamp(weighted_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    else:
        weighted_ratio = ratio
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    
    # Clipped surrogate objective
    pg_losses1 = -advantages * weighted_ratio
    pg_losses2 = -advantages * clipped_ratio
    pg_losses = torch.maximum(pg_losses1, pg_losses2)
    
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    
    # Aggregate loss
    if phi_weights is not None:
        # Sequence-level weighting: weight each sequence's mean loss by phi
        seq_losses = (pg_losses * response_mask).sum(dim=-1)  # (batch_size,)
        seq_lens = response_mask.sum(dim=-1).clamp(min=1)
        weighted_seq_losses = phi_weights * seq_losses / seq_lens
        pg_loss = weighted_seq_losses.mean()
    else:
        # Standard aggregation
        pg_loss = agg_loss(pg_losses, response_mask, loss_agg_mode)
    
    # Build metrics
    metrics = {
        "pg_clipfrac": pg_clipfrac.detach(),
        "ppo_kl": ppo_kl.detach(),
        "pg_clipfrac_lower": torch.tensor(0.0, device=pg_loss.device),
    }
    
    # Add weight stats
    if phi_weights is not None:
        metrics["lotis/phi_mean"] = phi_weights.mean().detach()
    if tis_weights is not None:
        metrics["lotis/tis_mean"] = tis_weights[response_mask.bool()].mean().detach()
    
    return pg_loss, metrics
