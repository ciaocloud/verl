"""LOTIS: Length-Optimized Token Importance Sampling for GRPO.

Actor-based implementation with DDP sync. Configure in YAML:

    actor_rollout_ref:
      actor:
        lotis:
          length_weight:
            enable: true
            num_rbf_kernels: 5
            lr: 0.01
          token_weight:
            enable: true
            lr: 0.01
            mode: "divergence" # or "mlp"
Note: For TIS, ref_log_prob must be in batch (from reference model).
      The actor auto-includes it when TIS is enabled.
"""

from verl.lotis.config import LOTISConfig, LengthWeightConfig, SequenceWeightConfig, TokenWeightConfig
from verl.lotis.modules import RBFLengthWeightModule, KLTokenWeightModule, MLPTokenWeightModule
from verl.lotis.sequence_module import MLPSequenceWeightModule
# Note: compute_lotis_policy_loss is NOT imported here to avoid circular imports with verl.workers.config.
# It must be imported where needed (e.g. in actors) to ensure registration.

__all__ = [
    # Config
    "LOTISConfig",
    "LengthWeightConfig",
    "TokenWeightConfig",
    "SequenceWeightConfig",
    # Modules
    "RBFLengthWeightModule",
    "KLTokenWeightModule",
    "MLPTokenWeightModule",
    "MLPSequenceWeightModule",
]
