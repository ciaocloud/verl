"""Configuration for LOTIS (Length-Optimized Token Importance Sampling)."""

from dataclasses import dataclass, field
from typing import List, Optional

from verl.base_config import BaseConfig


@dataclass
class LengthWeightConfig(BaseConfig):
    """Config for RBF-based sequence length weighting.
    
    When disabled, falls back to standard GRPO behavior (uniform weighting).
    """
    enable: bool = False
    num_rbf_kernels: int = 5
    rbf_centers: Optional[List[float]] = None  # None = auto linspace(-2, 2, K)
    rbf_bandwidth: float = 1.0
    alpha_init: float = 0.0  # 0 means phi=1 at init (standard GRPO)
    lr: float = 0.001
    weight_decay: float = 1e-4
    phi_clip_min: float = 0.2
    phi_clip_max: float = 5.0


@dataclass
class TokenWeightConfig(BaseConfig):
    """Config for Token Importance Sampling weighting.
    
    When disabled, falls back to uniform token weighting.
    """
    enable: bool = False
    
    # MLP Config (used when mode="mlp")
    mode: str = "mlp"  # "kl" (divergence) or "mlp"
    mlp_hidden_dim: int = 256
    mlp_num_layers: int = 2
    mlp_activation: str = "silu"
    psi_clip_min: float = 0.2
    psi_clip_max: float = 5.0
    
    # MLP Input Features (only applies when mode="mlp")
    # Each scalar feature is projected to scalar_embed_dim before concat with hidden state
    scalar_embed_dim: int = 16
    use_hidden_state: bool = True       # h_t vector: "What am I saying?"
    use_log_prob: bool = True           # Confidence: "Am I sure?"
    use_entropy: bool = True            # Uncertainty: "Did I struggle?"
    use_kl_divergence: bool = True      # Novelty: "Is this new?"
    use_relative_position: bool = True  # Timing: "Intro or conclusion?"
    use_semantic_drift: bool = True     # Focus: "Are we still on topic?"
    use_local_ppl: bool = True          # Smoothed PPL: "Hard reasoning block?"
    
    gamma_init: float = 0.1 
    lr: float = 0.001
    weight_decay: float = 1e-4
    gamma_max: float = 3.0
    div_clip: float = 2.0


@dataclass
class LOTISConfig(BaseConfig):
    """Main LOTIS configuration combining length and TIS weighting."""
    length_weight: LengthWeightConfig = field(default_factory=LengthWeightConfig)
    token_weight: TokenWeightConfig = field(default_factory=TokenWeightConfig)

    @property
    def is_enabled(self) -> bool:
        """Returns True if any LOTIS component is enabled."""
        return self.length_weight.enable or self.token_weight.enable
