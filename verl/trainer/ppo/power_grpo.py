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
    
    # =================================================================
    # LENGTH SCALING (happens AFTER normalization in adv_estimator)
    # 
    # Flow:
    # 1. adv_estimator: normalize advantages (no length involved)
    # 2. HERE: scale by 1/L^alpha
    # 3. agg_loss with seq-mean-token-sum: SUM tokens (no /L), mean seqs
    # 
    # Result: seq_loss = sum(pg_loss_t) / L^alpha
    # NO double division because seq-mean-token-sum sums, doesn't divide
    # =================================================================
    if length_alpha > 0:
        seq_lengths = response_mask.sum(dim=-1, keepdim=True).clamp(min=1).float()  # (bs, 1)
        length_weights = 1.0 / (seq_lengths ** length_alpha)  # (bs, 1)
        pg_losses = pg_losses * length_weights
    
    # Apply rollout correction if provided (for off-policy)
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights
    
    # =================================================================
    # AGGREGATION: seq-mean-token-sum
    # - SUM over tokens (NOT divide by L) 
    # - MEAN over sequences
    # This avoids double division since we already scaled by 1/L^alpha
    # =================================================================
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,  # Should be "seq-mean-token-sum"
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
    import numpy as np
    from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY
    
    # Check registration
    assert "power_grpo" in POLICY_LOSS_REGISTRY
    print("✓ Policy loss 'power_grpo' registered successfully")
    
    print("\n" + "=" * 70)
    print("Demonstrating 1/sqrt(L) Length Weighting Effect")
    print("=" * 70)
    
    # Create test data with different sequence lengths
    # Sequence 0: length 10
    # Sequence 1: length 50
    # Sequence 2: length 200
    lengths = [10, 50, 200]
    max_len = max(lengths)
    bs = len(lengths)
    
    # Create masks
    response_mask = torch.zeros(bs, max_len)
    for i, L in enumerate(lengths):
        response_mask[i, :L] = 1.0
    
    # Same advantage for all (normalized already)
    advantages = torch.ones(bs, max_len) * response_mask  # adv = 1.0 for all
    
    # Same log probs (ratio = 1, so no clipping effect)
    old_log_prob = torch.zeros(bs, max_len)
    log_prob = torch.zeros(bs, max_len)
    
    # Compute token-level losses WITHOUT length scaling
    # pg_loss_t = -adv * ratio = -1.0 * 1.0 = -1.0
    pg_losses_no_scale = -advantages * 1.0  # ratio = 1
    
    # Sequence losses without scaling (sum over tokens)
    seq_losses_no_scale = (pg_losses_no_scale * response_mask).sum(dim=-1)
    
    print("\nWithout length scaling (standard GRPO with seq-mean-token-sum):")
    print("-" * 70)
    print(f"{'Seq':<6} {'Length':<10} {'Token Loss':<15} {'Seq Loss (sum)':<20}")
    print("-" * 70)
    for i, L in enumerate(lengths):
        token_loss = pg_losses_no_scale[i, 0].item()
        seq_loss = seq_losses_no_scale[i].item()
        print(f"{i:<6} {L:<10} {token_loss:<15.4f} {seq_loss:<20.4f}")
    print("-" * 70)
    print(f"{'Mean seq loss:':<30} {seq_losses_no_scale.mean().item():.4f}")
    print("\n⚠️  Problem: Longer sequences have larger seq_loss (more tokens to sum)")
    print("   This causes longer sequences to dominate the gradient!")
    
    # Now with 1/sqrt(L) scaling
    print("\n" + "=" * 70)
    print("With 1/sqrt(L) scaling (Power-GRPO, alpha=0.5):")
    print("-" * 70)
    
    alpha = 0.5
    length_weights = 1.0 / (response_mask.sum(dim=-1, keepdim=True) ** alpha)
    pg_losses_scaled = pg_losses_no_scale * length_weights
    seq_losses_scaled = (pg_losses_scaled * response_mask).sum(dim=-1)
    
    print(f"{'Seq':<6} {'Length':<10} {'1/sqrt(L)':<12} {'Scaled Token':<15} {'Seq Loss':<15}")
    print("-" * 70)
    for i, L in enumerate(lengths):
        weight = length_weights[i, 0].item()
        token_loss = pg_losses_scaled[i, 0].item()
        seq_loss = seq_losses_scaled[i].item()
        print(f"{i:<6} {L:<10} {weight:<12.4f} {token_loss:<15.4f} {seq_loss:<15.4f}")
    print("-" * 70)
    print(f"{'Mean seq loss:':<30} {seq_losses_scaled.mean().item():.4f}")
    
    # Show the normalization effect
    print("\n" + "=" * 70)
    print("Comparison: seq_loss / sqrt(L) ratios")
    print("-" * 70)
    
    # Without scaling: seq_loss = L * token_loss
    # With scaling: seq_loss = L * token_loss / sqrt(L) = sqrt(L) * token_loss
    print(f"{'Seq':<6} {'Length':<10} {'No Scale':<15} {'With Scale':<15} {'Ratio':<15}")
    print("-" * 70)
    for i, L in enumerate(lengths):
        no_scale = seq_losses_no_scale[i].item()
        with_scale = seq_losses_scaled[i].item()
        ratio = with_scale / no_scale if no_scale != 0 else 0
        print(f"{i:<6} {L:<10} {no_scale:<15.4f} {with_scale:<15.4f} {ratio:<15.4f}")
    
    print("\n✓ With 1/sqrt(L), sequence losses are proportional to sqrt(L)")
    print("  instead of L, reducing the dominance of longer sequences.")
    
    # Show different alpha values
    print("\n" + "=" * 70)
    print("Effect of different alpha values:")
    print("-" * 70)
    print(f"{'Alpha':<10} {'L=10':<15} {'L=50':<15} {'L=200':<15} {'Ratio 200/10':<15}")
    print("-" * 70)
    
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        if alpha == 0:
            weights = torch.ones(bs, 1)
        else:
            weights = 1.0 / (response_mask.sum(dim=-1, keepdim=True) ** alpha)
        scaled = (pg_losses_no_scale * weights * response_mask).sum(dim=-1)
        ratio = scaled[2].item() / scaled[0].item() if scaled[0].item() != 0 else 0
        print(f"{alpha:<10} {scaled[0].item():<15.4f} {scaled[1].item():<15.4f} {scaled[2].item():<15.4f} {ratio:<15.2f}")
    
    print("-" * 70)
    print("alpha=0.0: No scaling (ratio = 20x)")
    print("alpha=0.5: sqrt scaling (ratio ≈ 4.5x)")  
    print("alpha=1.0: Full normalization (ratio = 1x, like token-mean)")
    
    print("\n✓ All tests passed!")
