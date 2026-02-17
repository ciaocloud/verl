"""Configuration for LOGO algorithm."""

from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig


@dataclass
class SamplingConfig(BaseConfig):
    """Config for variance-based prompt sampling."""
    rho: float = 1.0  # confidence calibration weight in priority score
    top_k: Optional[int] = None  # None = weighted multinomial over all prompts
    staleness_bonus: float = 0.01  # bonus per step since last visit
    epsilon: float = 0.1  # sampler-level epsilon-greedy: probability of replacing a selected index with uniform random (enables Lake exploration)


@dataclass
class AdvantageConfig(BaseConfig):
    """Config for hybrid advantage computation."""
    tau: float = 0.1  # temperature for adaptive lambda
    adaptive_lambda: bool = True
    lambda_fixed: float = 0.5  # fallback when adaptive_lambda=False


@dataclass
class DecayConfig(BaseConfig):
    """Config for value decay."""
    mode: str = "adaptive"  # "fixed" (time-based) or "adaptive" (drift-based)
    gamma: float = 0.99  # base decay factor (for fixed mode)
    sensitivity: float = 2.0  # for adaptive: gamma = 1/(1 + s*|V - R_bar|)
    gamma_clip_min: float = 0.1  # clip bounds for gamma
    gamma_clip_max: float = 0.95


@dataclass
class MinerConfig(BaseConfig):
    """Config for candidate mining."""
    enable: bool = False
    candidate_batch_size: int = 4096  # Lake candidates to scout per mine()
    similarity_threshold: float = 0.7
    k_neighbors: int = 5
    encoder_model: str = "all-MiniLM-L6-v2"  # sentence-transformers model


@dataclass
class PreFlightConfig(BaseConfig):
    """Config for pre-flight initialization epoch."""
    enable: bool = True
    sample_fraction: float = 1.0  # fraction of prompts to scout
    blend: float = 0.5  # V_0 = blend + (R - blend) * exp(mean_logprob)


@dataclass
class LOGOConfig(BaseConfig):
    """Main LOGO configuration."""
    _mutable_fields = {"_meta_store_ref"}

    value_mode: str = "bayesian"  # "bayesian" (Beta distribution) or "ema" (simple EMA)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    advantage: AdvantageConfig = field(default_factory=AdvantageConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)
    miner: MinerConfig = field(default_factory=MinerConfig)
    preflight: PreFlightConfig = field(default_factory=PreFlightConfig)
