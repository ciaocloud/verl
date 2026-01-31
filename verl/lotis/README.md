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
| `clip_min/max` | `0.1/10.0` | Weight bounds |

**TIS Weight**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable` | `False` | Enable |
| `beta_init` | `1.0` | Initial beta |
| `beta_lr` | `0.01` | Learning rate |
| `clip_min/max` | `0.1/10.0` | Weight bounds |

## Metrics

- `lotis/rbf_alphas_mean`: RBF alpha params
- `lotis/tis_beta`: TIS beta param
- `lotis/rbf_phi_mean`: Sequence weight mean
- `lotis/tis_weight_mean`: Token weight mean
