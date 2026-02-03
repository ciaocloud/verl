# LOTIS: Length-Optimized Token Importance Sampling

Actor-based implementation with DDP sync for distributed training.

## Usage

Configure in YAML:

```yaml
actor_rollout_ref:
  actor:
    lotis:
      length_weight:
        enable: true
        num_rbf_kernels: 5
        alpha_lr: 0.01
      tis_weight:
        enable: true
        beta_init: 1.0
        beta_lr: 0.01
```

**Note:** For TIS weighting, `ref_log_prob` must be available in the batch (computed by reference model during rollout). The actor auto-includes it when TIS is enabled.

## Architecture

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Actor 0        │  │  Actor 1        │  │  Actor 2        │
│                 │  │                 │  │                 │
│  LLM (FSDP)     │  │  LLM (FSDP)     │  │  LLM (FSDP)     │
│  LOTIS (DDP)    │  │  LOTIS (DDP)    │  │  LOTIS (DDP)    │
│                 │  │                 │  │                 │
│  Single optimizer handles both:                          │
│  - LLM params (slow LR)                                  │
│  - LOTIS params (fast LR, no weight decay)               │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                     DDP AllReduce
              (LOTIS params stay in sync!)
```

## Parameters

**Length Weight (RBF)**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable` | `False` | Enable |
| `num_rbf_kernels` | `5` | Number of kernels |
| `alpha_init` | `0.0` | Initial values (0 = standard GRPO) |
| `alpha_lr` | `0.01` | Learning rate |
| `phi_clip_min` | `0.2` | Weight lower bound |
| `phi_clip_max` | `5.0` | Weight upper bound |

**TIS Weight**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable` | `False` | Enable |
| `gamma_init` | `1.0` | Initial gamma |
| `gamma_lr` | `0.01` | Learning rate |
| `gamma_max` | `3.0` | Max gamma (soft cap) |
| `psi_clip_min` | `0.2` | Weight lower bound |
| `psi_clip_max` | `5.0` | Weight upper bound |
| `div_clip` | `2.0` | Divergence clamp bound |

## Metrics

- `lotis/rbf_alphas_mean`: RBF alpha params
- `lotis/tis_gamma`: TIS gamma param
- `lotis/rbf_phi_mean`: Sequence weight (phi) mean
- `lotis/tis_psi_mean`: Token weight (psi) mean
