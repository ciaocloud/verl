# LOGO: Learning with Optimized Gradients and Opportunity Sampling

Variance-based curriculum learning with critic-free value estimation for sample-efficient RL.

## Components

| Module | File | Description |
|--------|------|-------------|
| Config | `config.py` | `LOGOConfig` and nested dataclasses |
| Meta-Store | `meta_store.py` | CPU-resident per-prompt Beta(alpha,beta) tracker |
| Advantage | `advantage.py` | Hybrid advantage: GRPO group baseline + stored value control variate |
| Sampler | `sampler.py` | Thompson-sampling curriculum sampler (extends `AbstractCurriculumSampler`) |
| RAG Miner | `rag_miner.py` | KNN-based value extrapolation from visited to unvisited prompts |

## Quick Start

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=logo \
    +algorithm.logo.advantage.tau=0.1 \
    +algorithm.logo.decay.mode=adaptive \
    +algorithm.logo.sampling.rho=1.0 \
    ...
```

See `examples/logo_trainer/` for complete scripts.

## How It Works

### Hybrid Advantage

```
A = (R - mu_group) + lambda * (R - V_stored)
    ─────────────────────────────────────────
               sigma_batch + epsilon

lambda = exp(-sigma_group^2 / tau)
```

- High group variance → lambda ≈ 0 → pure GRPO
- Low group variance → lambda ≈ 1 → trust stored value

### Variance-Based Sampling

Each epoch, prompts are scored via Thompson sampling:
```
p_tilde ~ Beta(alpha, beta)
S = sqrt(p_tilde * (1 - p_tilde)) + rho * |p_tilde - P_ref| + staleness_bonus * delta_t
```

Prompts with highest uncertainty (V near 0.5) are prioritised.

### Value Updates

After rollout, per-prompt Beta parameters are updated with decay:
```
alpha <- alpha * gamma^delta_t + sum(R)
beta  <- beta  * gamma^delta_t + sum(1 - R)
```

Adaptive decay adjusts gamma based on drift between stored value and observed reward.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau` | 0.1 | Advantage temperature (lower = more adaptive lambda) |
| `rho` | 1.0 | Confidence gap weight in sampling score |
| `decay.mode` | adaptive | Value update strategy |
| `decay.gamma` | 0.99 | Base exponential decay factor |
| `sampling.staleness_bonus` | 0.01 | Per-step bonus for unvisited prompts |

## Metrics

All metrics are prefixed with `logo/`:
- `value_mean`, `value_std`: Distribution of stored values
- `alpha_mean`, `beta_mean`: Beta distribution parameters
- `staleness_mean`, `staleness_max`: How stale the meta-store is
- `meta_store_size`: Number of tracked prompts
