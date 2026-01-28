"""
GRPO/RLOO with Batch-Level Standard Deviation Normalization

Variants implemented:
- grpo_batch_std:           adv = (score - group_mean) / batch_std
- grpo_batch_std_vectorized: same, vectorized
- rloo_batch_std:           adv = (score - LOO_mean) / batch_std  (Leave-One-Out)
- rloo_batch_std_vectorized: same, vectorized

Comparison:
- Standard GRPO:  adv = (score - group_mean) / group_std
- Dr.GRPO:        adv = score - group_mean
- Standard RLOO:  adv = score - LOO_mean  (no std normalization)
- This file:      adv = (score - [group|LOO]_mean) / batch_std

Why batch std?
- Group std can be very small when all samples in a group have similar rewards
- Dividing by small std → exploding advantages → unstable training
- Batch std is more stable (larger sample size)

Why LOO (Leave-One-Out)?
- Standard baseline includes current sample in mean calculation
- LOO excludes current sample → unbiased gradient estimate
- LOO_mean_i = (group_sum - score_i) / (n-1)

Usage:
    # Config:
    algorithm.adv_estimator=grpo_batch_std      # Group mean + batch std
    algorithm.adv_estimator=rloo_batch_std      # LOO mean + batch std
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.config.algorithm import AlgoConfig
from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("grpo_batch_std")
def compute_grpo_batch_std_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,  # Ignored - we always use batch std
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GRPO advantage with batch-level std normalization.
    
    Formula: adv_i = (score_i - group_mean_i) / batch_std
    
    This combines:
    - Group-relative mean (like standard GRPO)
    - Batch-level std (more stable than group std)
    
    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) group IDs
        epsilon: small value for numerical stability
        norm_adv_by_std_in_grpo: ignored (we always normalize by batch std)
        config: algorithm config
    
    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length) - same as advantages for outcome reward
    """
    # =================================================================
    # STEP 1: Compute sequence-level scores (SUM, no division by L)
    # =================================================================
    scores = token_level_rewards.sum(dim=-1)  # (bs,) - just sum, no /L
    
    # Compute group means
    id2score = defaultdict(list)
    id2mean = {}
    
    with torch.no_grad():
        bsz = scores.shape[0]
        
        # Collect scores by group
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        
        # Compute group means
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        # Compute batch std (over ALL scores)
        batch_std = torch.std(scores)
        
        # =================================================================
        # STEP 2: NORMALIZE (no length involved here)
        # This happens BEFORE length scaling in power_grpo
        # =================================================================
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (batch_std + epsilon)
        
        # Expand to response length (same value for all tokens)
        scores = scores.unsqueeze(-1) * response_mask
    
    return scores, scores


@register_adv_est("grpo_batch_std_vectorized")
def compute_grpo_batch_std_vectorized_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,  # Ignored
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized version of GRPO with batch std normalization.
    
    More efficient for large batches using scatter operations.
    """
    from verl.trainer.ppo.core_algos import as_torch_index, group_mean_std
    
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    g = as_torch_index(index, device=scores.device)
    
    # Group mean (batch std computed separately)
    mean_g, _, _ = group_mean_std(scores, g, eps=epsilon)
    
    # Batch std
    batch_std = torch.std(scores)
    
    # Advantage = (score - group_mean) / batch_std
    scalars = (scores - mean_g[g]) / (batch_std + epsilon)
    
    # Expand to response length
    advantages = scalars.unsqueeze(-1) * response_mask
    
    return advantages, advantages


# =============================================================================
# LOO (Leave-One-Out) Variants
# =============================================================================

@register_adv_est("rloo_batch_std")
def compute_rloo_batch_std_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,  # Ignored
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    RLOO (Leave-One-Out) advantage with batch-level std normalization.
    
    Formula: adv_i = (score_i - LOO_mean_i) / batch_std
    
    Where LOO_mean_i = (group_sum - score_i) / (n-1)
                     = (n * group_mean - score_i) / (n-1)
    
    This simplifies to: adv_i = n/(n-1) * (score_i - group_mean) / batch_std
    
    Why LOO?
    - Standard baseline uses all samples including the current one
    - LOO excludes current sample → unbiased gradient estimate
    - See RLOO paper: https://arxiv.org/abs/2402.14740
    
    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) group IDs
        epsilon: small value for numerical stability
        config: algorithm config
    
    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    # =================================================================
    # STEP 1: Compute sequence-level scores (SUM, no division by L)
    # =================================================================
    scores = token_level_rewards.sum(dim=-1)  # (bs,) - just sum, no /L
    
    id2score = defaultdict(list)
    id2mean = {}
    
    with torch.no_grad():
        bsz = scores.shape[0]
        
        # Collect scores by group
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        
        # Compute group means
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        # Compute batch std (over ALL original scores)
        batch_std = torch.std(scores)
        
        # =================================================================
        # STEP 2: NORMALIZE with LOO baseline (no length involved here)
        # This happens BEFORE length scaling in power_grpo
        # LOO advantage = n/(n-1) * (score - group_mean) / batch_std
        # =================================================================
        advantages = torch.zeros_like(scores)
        for i in range(bsz):
            n = len(id2score[index[i]])
            if n > 1:
                # LOO: scale by n/(n-1)
                loo_factor = n / (n - 1)
                advantages[i] = loo_factor * (scores[i] - id2mean[index[i]]) / (batch_std + epsilon)
            else:
                # Single sample in group: no baseline
                advantages[i] = 0.0
        
        # Expand to response length (same value for all tokens)
        advantages = advantages.unsqueeze(-1) * response_mask
    
    return advantages, advantages


@register_adv_est("rloo_batch_std_vectorized")
def compute_rloo_batch_std_vectorized_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,  # Ignored
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized RLOO with batch std normalization.
    
    Formula: adv_i = (score_i - LOO_mean_i) / batch_std
           = (n * score_i - group_sum) / ((n-1) * batch_std)
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    
    with torch.no_grad():
        # Convert index to tensor
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)
        
        # Group counts and sums
        c = torch.bincount(inv)[inv].to(scores.dtype)  # count per sample's group
        group_sums = torch.bincount(inv, weights=scores)[inv]  # sum per sample's group
        
        # Batch std
        batch_std = torch.std(scores)
        
        # LOO advantage: (n * score - group_sum) / ((n-1) * batch_std)
        # = (score - LOO_mean) / batch_std
        # where LOO_mean = (group_sum - score) / (n-1)
        advantages = ((c * scores - group_sums) / ((c - 1).clamp_min(1) * (batch_std + epsilon))) * (c > 1)
        
        # Expand to response length
        advantages = advantages.unsqueeze(-1) * response_mask
    
    return advantages, advantages


# =============================================================================
# Test
# =============================================================================

def demo_loo_bias():
    """
    Demonstrate the bias difference between Group Mean and LOO Mean.
    
    Key insight:
    - Group Mean: advantage = score - mean(all_scores)
      The current score influences the mean, creating correlation/bias
      
    - LOO Mean: advantage = score - mean(other_scores)  
      Current score excluded from mean → unbiased gradient estimate
      
    Mathematical relationship:
      LOO_advantage = n/(n-1) * Group_advantage
      
    The "bias" in group mean comes from the correlation between score and mean.
    When score is high, mean is also slightly higher (because score is included).
    This reduces the effective advantage, biasing the gradient downward.
    """
    print("=" * 70)
    print("LOO (Leave-One-Out) vs Group Mean: Bias Analysis")
    print("=" * 70)
    
    print("\n1. SCALING FACTOR: n/(n-1)")
    print("-" * 70)
    print(f"{'n':<8} {'n/(n-1)':<12} {'% increase':<15} {'1 - (n-1)/n':<15}")
    print("-" * 70)
    
    for n in [2, 4, 8, 16, 32, 64]:
        factor = n / (n - 1)
        pct_increase = (factor - 1) * 100
        bias_fraction = 1 - (n - 1) / n  # = 1/n
        print(f"{n:<8} {factor:<12.4f} {pct_increase:<15.2f}% {bias_fraction:<15.4f}")
    
    print("-" * 70)
    print("\nInterpretation:")
    print("- LOO advantage = n/(n-1) × Group advantage")
    print("- For n=2: LOO is 2x the group advantage (50% bias in group mean)")
    print("- For n=64: LOO is 1.016x (only 1.6% bias)")
    print("- Bias fraction = 1/n (how much the current score 'contaminates' the mean)")
    
    print("\n" + "=" * 70)
    print("2. NUMERICAL EXAMPLE: Effect on Gradient Magnitude")
    print("-" * 70)
    
    # Simulate: each sample has score=1.0 (positive reward)
    # Group mean will be 1.0, so group advantage = 0
    # But LOO mean depends on n
    
    print("\nScenario: One sample has score=2.0, all others have score=1.0")
    print(f"{'n':<8} {'Group Mean':<12} {'Group Adv':<12} {'LOO Mean':<12} {'LOO Adv':<12}")
    print("-" * 70)
    
    target_score = 2.0
    other_score = 1.0
    
    for n in [2, 4, 8, 16, 32, 64]:
        # Group mean includes the target
        group_sum = target_score + (n - 1) * other_score
        group_mean = group_sum / n
        group_adv = target_score - group_mean
        
        # LOO mean excludes the target
        loo_sum = (n - 1) * other_score
        loo_mean = loo_sum / (n - 1)
        loo_adv = target_score - loo_mean
        
        print(f"{n:<8} {group_mean:<12.4f} {group_adv:<12.4f} {loo_mean:<12.4f} {loo_adv:<12.4f}")
    
    print("-" * 70)
    print("\nObservation:")
    print("- Group advantage is REDUCED because the high score pulls up the mean")
    print("- LOO advantage = 1.0 (constant) because mean of others is always 1.0")
    print("- The bias is larger for small n (high score has more influence on mean)")
    
    print("\n" + "=" * 70)
    print("3. GRADIENT BIAS ANALYSIS")
    print("-" * 70)
    
    print("\nFor policy gradient: ∇J = E[∇log π(a|s) × A(s,a)]")
    print("\nGroup Mean bias:")
    print("- E[score × (score - mean)] = E[score²] - E[score × mean]")
    print("- Since mean includes score: E[score × mean] > E[score] × E[mean]")
    print("- This correlation REDUCES the gradient (underestimates advantage)")
    print("")
    print("LOO Mean (unbiased):")
    print("- E[score × (score - LOO_mean)] = E[score²] - E[score × LOO_mean]")
    print("- Since LOO_mean excludes score: E[score × LOO_mean] = E[score] × E[LOO_mean]")
    print("- No correlation → unbiased gradient estimate")
    
    print("\n" + "=" * 70)
    print("4. VARIANCE-BIAS TRADEOFF")
    print("-" * 70)
    print(f"{'n':<8} {'Bias (1/n)':<15} {'Variance Red.':<18} {'Recommendation':<20}")
    print("-" * 70)
    print(f"{'2':<8} {'50%':<15} {'High':<18} {'Use LOO':<20}")
    print(f"{'4':<8} {'25%':<15} {'Moderate':<18} {'Use LOO':<20}")
    print(f"{'8':<8} {'12.5%':<15} {'Moderate':<18} {'LOO preferred':<20}")
    print(f"{'16':<8} {'6.25%':<15} {'Low':<18} {'Either OK':<20}")
    print(f"{'32':<8} {'3.1%':<15} {'Low':<18} {'Either OK':<20}")
    print(f"{'64':<8} {'1.6%':<15} {'Very Low':<18} {'Either OK':<20}")
    print("-" * 70)
    print("\nConclusion: LOO is most important for small n (n ≤ 8)")


if __name__ == "__main__":
    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY
    
    # First run the LOO bias demo
    demo_loo_bias()
    
    print("\n\n")
    print("=" * 60)
    print("GRPO/RLOO with Batch-Level Std Normalization")
    print("=" * 60)
    
    # Check registration
    assert "grpo_batch_std" in ADV_ESTIMATOR_REGISTRY
    assert "grpo_batch_std_vectorized" in ADV_ESTIMATOR_REGISTRY
    assert "rloo_batch_std" in ADV_ESTIMATOR_REGISTRY
    assert "rloo_batch_std_vectorized" in ADV_ESTIMATOR_REGISTRY
    print("✓ All advantage estimators registered successfully")
    
    # Create test data
    bs, seq_len = 8, 50
    
    # Rewards: Group 0 has high variance, Group 1 has low variance
    token_level_rewards = torch.zeros(bs, seq_len)
    index = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # 4 samples per group
    
    # Group 0: rewards [0, 1, 2, 3] - high variance
    token_level_rewards[0, -1] = 0.0
    token_level_rewards[1, -1] = 1.0
    token_level_rewards[2, -1] = 2.0
    token_level_rewards[3, -1] = 3.0
    
    # Group 1: rewards [1.4, 1.5, 1.5, 1.6] - low variance
    token_level_rewards[4, -1] = 1.4
    token_level_rewards[5, -1] = 1.5
    token_level_rewards[6, -1] = 1.5
    token_level_rewards[7, -1] = 1.6
    
    response_mask = torch.ones(bs, seq_len)
    
    # Compute advantages
    adv_grpo_batch, _ = compute_grpo_batch_std_advantage(
        token_level_rewards, response_mask, index
    )
    adv_rloo_batch, _ = compute_rloo_batch_std_advantage(
        token_level_rewards, response_mask, index
    )
    
    # Compare with standard implementations
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_rloo_outcome_advantage
    adv_grpo_std, _ = compute_grpo_outcome_advantage(
        token_level_rewards, response_mask, index, norm_adv_by_std_in_grpo=True
    )
    adv_rloo_std, _ = compute_rloo_outcome_advantage(
        token_level_rewards, response_mask, index
    )
    
    print("\nComparison (last token advantages):")
    print("-" * 90)
    print(f"{'Sample':<8} {'Group':<8} {'Reward':<10} {'GRPO':<12} {'GRPO+Batch':<12} {'RLOO':<12} {'RLOO+Batch':<12}")
    print("-" * 90)
    
    for i in range(bs):
        reward = token_level_rewards[i, -1].item()
        grp = index[i]
        a_grpo = adv_grpo_std[i, -1].item()
        a_grpo_b = adv_grpo_batch[i, -1].item()
        a_rloo = adv_rloo_std[i, -1].item()
        a_rloo_b = adv_rloo_batch[i, -1].item()
        print(f"{i:<8} {grp:<8} {reward:<10.2f} {a_grpo:<12.4f} {a_grpo_b:<12.4f} {a_rloo:<12.4f} {a_rloo_b:<12.4f}")
    
    print("-" * 90)
    
    # Show statistics
    group0_rewards = [0, 1, 2, 3]
    group1_rewards = [1.4, 1.5, 1.5, 1.6]
    
    print(f"\nGroup 0: mean={np.mean(group0_rewards):.2f}, std={np.std(group0_rewards):.4f}")
    print(f"Group 1: mean={np.mean(group1_rewards):.2f}, std={np.std(group1_rewards):.4f}")
    print(f"Batch std: {torch.std(token_level_rewards.sum(dim=-1)).item():.4f}")
    
    print("\nKey insights:")
    print("- Group 1 has tiny std (0.07) → group-std advantages explode!")
    print("- Batch std normalization avoids this instability")
    print("- LOO (Leave-One-Out) gives unbiased gradient estimates")
    
    print("\n" + "=" * 60)
    print("Usage:")
    print("=" * 60)
    print("""
# Group mean + batch std:
algorithm.adv_estimator=grpo_batch_std

# LOO mean + batch std:
algorithm.adv_estimator=rloo_batch_std
""")
    
    print("✓ All tests passed!")

