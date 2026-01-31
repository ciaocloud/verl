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
    alpha_lr: float = 0.001
    clip_min: float = 0.2
    clip_max: float = 5.0


@dataclass
class TISWeightConfig(BaseConfig):
    """Config for Token Importance Sampling weighting.
    
    When disabled, falls back to uniform token weighting.
    """
    enable: bool = False
    beta_init: float = 0.1 
    beta_lr: float = 0.001
    clip_min: float = 0.2
    clip_max: float = 5.0


@dataclass
class LOTISConfig(BaseConfig):
    """Main LOTIS configuration combining length and TIS weighting."""
    length_weight: LengthWeightConfig = field(default_factory=LengthWeightConfig)
    tis_weight: TISWeightConfig = field(default_factory=TISWeightConfig)

    @property
    def is_enabled(self) -> bool:
        """Returns True if any LOTIS component is enabled."""
        return self.length_weight.enable or self.tis_weight.enable
