# LOGO Metrics Reference

All metrics are logged under the `logo/` prefix every training step. Each metric is tagged:
- **(batch)**: computed on the current training batch
- **(step)**: computed from the prompts updated this step
- **(store)**: aggregated over the entire meta-store
- **(cumulative)**: accumulated across all steps so far

---

## 1. Advantage Internals

Computed in `verl/logo/advantage.py`. The hybrid advantage is:

```
A_i = (R_i - mu_group) + lambda * (R_i - V_stored)
A_final = A_i / std(A_batch)
```

where `lambda = exp(-sigma_group^2 / tau)` (adaptive).

| Metric | Tag | Formula | What it tells you |
|--------|-----|---------|-------------------|
| `logo/lambda_mean` | batch | `mean(exp(-sigma_group^2 / tau))` | Average blending weight. **1.0** = fully trusting stored value. **~0** = ignoring stored value (degenerating to GRPO). |
| `logo/lambda_std` | batch | `std(lambda)` across samples | Spread of lambda across groups. **High** = varies by group (healthy). **0** = uniform. |
| `logo/sigma_group_mean` | batch | `mean(std(R) per group)` | Average within-group reward std. Drives lambda via `exp(-sigma^2/tau)`. |
| `logo/grpo_adv_abs_mean` | batch | `mean(\|R_i - mu_group\|)` | Magnitude of the GRPO (local) advantage component. |
| `logo/ppo_adv_abs_mean` | batch | `mean(\|R_i - V_stored\|)` | Magnitude of the stored-value (global) advantage component. |
| `logo/global_local_ratio` | batch | `ppo_adv_abs_mean / grpo_adv_abs_mean` | Relative strength. **0.2-0.5** = gentle nudge (healthy). **>1.0** = stored value overpowers GRPO. |
| `logo/sign_conflict_rate` | batch | `mean(sign(grpo_adv) != sign(ppo_adv))` | How often GRPO and stored value disagree on direction. **<10%** = aligned. **>50%** = memory and model disagree (red flag). |

---

## 2. Stored Value Health

### Batch-level (V_stored vs actual reward on current batch)

From `ray_trainer.py`:

| Metric | Tag | Formula | What it tells you |
|--------|-----|---------|-------------------|
| `logo/value_reward_corr` | batch | `pearson(V_stored, sum(R_tokens))` | **The key LOGO metric.** Correlation between stored values and realized rewards. **>0.3** = predictive. **~0** = noise. **Negative** = anti-correlated (very bad). |
| `logo/value_prediction_error` | batch | `mean((V_stored - sum(R_tokens))^2)` | MSE between predicted and actual reward. Should decrease as the store learns. Spikes = policy moved into new territory. |
| `logo/batch_reward_mean` | batch | `mean(sum(R_tokens))` | Mean reward of training batch. Should hover **0.4-0.7** if the sampler is finding the frontier. |

### Step-level (computed on prompts updated this step)

From `meta_store.py` `update()`:

| Metric | Tag | Formula | What it tells you |
|--------|-----|---------|-------------------|
| `logo/decay_mean` | step | `mean(gamma_applied)` where `gamma = 1/(1 + sensitivity * \|V_old - R_bar\|)` (adaptive) or `gamma_base^delta_t` (fixed) | Average decay factor. **~0.9** = stable values, memory persists. **~0.1** = large drift, aggressively discounting. |
| `logo/decay_std` | step | `std(gamma_applied)` | Spread of decay. **High** = some prompts stable, others drifting (healthy heterogeneity). |
| `logo/value_drift` | step | `mean(\|V_old - R_bar\|)` | How far old values were from new observations. Measures "surprise". Should decrease over training. |
| `logo/value_uncertainty` | step | `mean(sqrt(V_new * (1 - V_new)))` | Post-update uncertainty. Should stay **0.3-0.5**. Below 0.1 = replaying well-known prompts. |

### Store-level (aggregated over all prompts in Memory)

From `meta_store.py` `get_statistics()`:

| Metric | Tag | Formula | What it tells you |
|--------|-----|---------|-------------------|
| `logo/store_value_mean` | store | `mean(V)` over all Memory prompts | Overall difficulty landscape estimate. |
| `logo/store_value_std` | store | `std(V)` | Spread. **High** = well-differentiated landscape. **~0** = all prompts look the same. |
| `logo/store_value_min` | store | `min(V)` | Lowest stored value. |
| `logo/store_value_max` | store | `max(V)` | Highest stored value. Healthy: min near 0, max near 1 = polarized landscape. |
| `logo/store_size` | store | `len(Memory)` | Number of prompts with rollout data. |
| `logo/memory_fraction` | store | `len(Memory) / len(All)` | Fraction explored. Should grow toward 1.0. |
| `logo/alpha_mean` | store | `mean(alpha)` (Bayesian only) | Average posterior alpha. |
| `logo/beta_mean` | store | `mean(beta)` (Bayesian only) | Average posterior beta. `alpha >> beta` = most prompts solvable. |
| `logo/concentration_mean` | store | `mean(alpha + beta)` (Bayesian only) | Average total evidence per prompt. Grows with observations. |
| `logo/concentration_std` | store | `std(alpha + beta)` (Bayesian only) | Evidence spread. **High** = some prompts visited much more than others. |
| `logo/staleness_mean` | store | `mean(current_step - last_step)` | Average steps since each prompt was last visited. |
| `logo/staleness_max` | store | `max(current_step - last_step)` | Most neglected prompt. |
| `logo/staleness_p90` | store | 90th percentile of `(current_step - last_step)` | Tail behavior: many forgotten prompts, or just one outlier? |

---

## 3. Sampling & Exploration

### Priority scores (from `sampler.py`)

Recomputed each epoch via `compute_sampling_scores`. For Bayesian mode (Thompson sampling):

```
p_tilde ~ Beta(alpha, beta)
S = sqrt(p_tilde * (1 - p_tilde)) + rho * |p_tilde - exp(logprob_last)| + staleness_bonus * delta_t
```

| Metric | Tag | Formula | What it tells you |
|--------|-----|---------|-------------------|
| `logo/priority_mean` | store | `mean(S)` | Overall priority score level. |
| `logo/priority_std` | store | `std(S)` | Score spread. |
| `logo/priority_max` | store | `max(S)` | Highest-priority prompt. |
| `logo/priority_min` | store | `min(S)` | Lowest-priority prompt. |
| `logo/priority_entropy` | store | `-sum(p_i * log(p_i))` where `p = softmax(S)` | How concentrated the sampling distribution is. **High** = broad exploration. **Low** = focused on few prompts. |
| `logo/priority_entropy_ratio` | store | `entropy / log(N)` | Normalized (0-1). **>0.8** = nearly uniform. **<0.3** = highly concentrated. |
| `logo/priority_top10pct` | store | `mean(top 10% of S)` | How much the highest-priority prompts stand out. |
| `logo/priority_bot10pct` | store | `mean(bottom 10% of S)` | How neglected the lowest-priority prompts are. Large gap from top = strong differentiation. |

### Sample counts (from `sampler.py`)

Per-prompt cumulative count of how many times the sampler has yielded each index. Reported over visited prompts only.

| Metric | Tag | Formula | What it tells you |
|--------|-----|---------|-------------------|
| `logo/sample_count_mean` | cumulative | `mean(count_i)` for visited prompts | Average times sampled per visited prompt. |
| `logo/sample_count_std` | cumulative | `std(count_i)` | Sampling inequality. **High** = some prompts heavily favored. |
| `logo/sample_count_max` | cumulative | `max(count_i)` | Most-sampled prompt. If `max >> mean`, sampler is over-focusing. |
| `logo/sample_count_min` | cumulative | `min(count_i)` | Least-sampled visited prompt. |
| `logo/prompts_sampled` | cumulative | number of distinct indices ever yielded | How many prompts the sampler has touched. |

### Coverage (from `ray_trainer.py`)

| Metric | Tag | Formula | What it tells you |
|--------|-----|---------|-------------------|
| `logo/unique_prompts_seen` | cumulative | count of distinct `logo_prompt_id` in training batches | Should grow steadily. If it plateaus, the sampler is stuck in a loop. |

### RAG miner (from `rag_miner.py`, if enabled)

| Metric | Tag | Formula | What it tells you |
|--------|-----|---------|-------------------|
| `logo/memory_size` | store | `len(Memory)` | Same as store_size. |
| `logo/lake_size` | store | `len(Lake)` | Prompts not yet visited. |
| `logo/exploration_ratio` | store | `lake / total` | Inverse of memory_fraction. |
| `logo/value_cache_size` | store | count of kNN-extrapolated values | How many Lake prompts have estimated values. |

---

## 4. Red Flags

| Condition | What it means | Action |
|-----------|---------------|--------|
| `store_value_mean` -> 0 or 1 | Memory collapsed to one extreme. | Increase `gamma_clip_min` or reduce `sensitivity`. |
| `value_reward_corr` < 0 | Values anti-correlated with actual reward. | Reset meta-store or increase `tau`. |
| `sign_conflict_rate` > 50% | Model and memory completely disagree. | Increase `sensitivity` or `tau`. |
| `lambda_mean` stuck at 0 | Degenerated to pure GRPO. | Increase `tau` or reduce rollout `n`. |
| `unique_prompts_seen` plateaus | Sampler stuck in a loop. | Increase `staleness_bonus` or `epsilon`. |
| `batch_reward_mean` -> 0 | Sampling prompts that are too hard. | Reduce `rho` or use top-K. |
| `batch_reward_mean` -> 1 | Only picking easy prompts. | Increase `staleness_bonus`. |
| `sample_count_max` >> `sample_count_mean` | Heavily uneven visitation. | Increase `epsilon` or `staleness_bonus`. |
| `decay_mean` < 0.2 consistently | Values wiped every step, never accumulating. | Reduce `sensitivity` or increase `gamma_clip_min`. |

---

## 5. Where Each Metric Is Computed

| File | Metrics |
|------|---------|
| `verl/logo/advantage.py` | `lambda_*`, `sigma_group_mean`, `grpo_adv_abs_mean`, `ppo_adv_abs_mean`, `global_local_ratio`, `sign_conflict_rate` |
| `verl/logo/meta_store.py` | `store_value_{mean,std,min,max}`, `store_size`, `staleness_*`, `alpha_*`, `beta_*`, `concentration_*`, `memory_fraction`, `decay_*`, `value_drift`, `value_uncertainty` |
| `verl/logo/sampler.py` | `priority_*`, `sample_count_*`, `prompts_sampled` |
| `verl/logo/rag_miner.py` | `memory_size`, `lake_size`, `exploration_ratio`, `value_cache_size` |
| `verl/trainer/ppo/ray_trainer.py` | `value_reward_corr`, `value_prediction_error`, `batch_reward_mean`, `unique_prompts_seen` |

