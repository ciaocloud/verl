from typing import Any, Optional

import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
from verl.workers.config.actor import ActorConfig


@register_policy_loss("power_grpo")
def compute_policy_loss_power_grpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-sum",  # GRPO style, NOT token-mean
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    assert config is not None, "config is required"
    
    # Get config values
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)
    
    # Get length_alpha from policy_loss config (default 0.5 for sqrt)
    length_alpha = config.policy_loss.get("length_alpha", 0.5)
    
    # Compute PPO ratio (this is ALWAYS applied - fundamental to PPO)
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    
    # Standard PPO clipping
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    
    # Dual-clip for negative advantages
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    
    # === KEY: Apply L^(-alpha) length weighting ===
    if length_alpha > 0:
        seq_lengths = response_mask.sum(dim=-1, keepdim=True).clamp(min=1).float()  # (bs, 1)
        length_weights = 1.0 / (seq_lengths ** length_alpha)  # (bs, 1)
        # Expand to (bs, seq_len) and apply
        pg_losses = pg_losses * length_weights
    
    # Apply rollout correction if provided (for off-policy)
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights
    
    # Aggregate using seq-mean-token-sum (GRPO style)
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode="seq-mean-token-sum",  # GRPO style, NOT token-mean
        **config.global_batch_info
    )
    
    # Metrics
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "actor/length_alpha": length_alpha,
    }
    
    return pg_loss, pg_metrics


if __name__ == "__main__":
    from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY
    
    # Check registration
    assert "power_grpo" in POLICY_LOSS_REGISTRY
    print("✓ Policy loss 'power_grpo' registered successfully")
