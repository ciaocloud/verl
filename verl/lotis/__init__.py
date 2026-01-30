"""LOTIS: Length-Optimized Token Importance Sampling for GRPO.

Actor-based implementation with DDP sync. Configure in YAML:

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
Note: For TIS, ref_log_prob must be in batch (from reference model).
      The actor auto-includes it when TIS is enabled.
"""

from verl.lotis.config import LOTISConfig, LengthWeightConfig, TISWeightConfig
from verl.lotis.modules import RBFLengthWeightModule, TISWeightModule
# Note: compute_lotis_policy_loss is NOT imported here to avoid circular imports with verl.workers.config. 
# It must be imported where needed (e.g. in actors) to ensure registration.

__all__ = [
    # Config
    "LOTISConfig",
    "LengthWeightConfig",
    "TISWeightConfig",
    # Modules
    "RBFLengthWeightModule",
    "TISWeightModule",
]
