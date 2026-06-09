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
    
    mode: str = "mlp"  # "kl" (divergence) or "mlp"
    psi_clip_min: float = 0.2
    psi_clip_max: float = 5.0
    lr: float = 0.001
    weight_decay: float = 1e-4
    # MLP Input Features (only applies when mode="mlp")
    use_hidden_state: bool = True       # h_t vector: "What am I saying?"
    use_log_prob: bool = True           # Confidence: "Am I sure?"
    use_entropy: bool = True            # Uncertainty: "Did I struggle?"
    use_kl_divergence: bool = True      # Novelty: "Is this new?"
    use_relative_position: bool = True  # Timing: "Intro or conclusion?"
    use_semantic_drift: bool = True     # Focus: "Are we still on topic?"
    use_local_ppl: bool = True          # Smoothed PPL: "Hard reasoning block?"
        
    gamma_init: float = 0.1 
    gamma_max: float = 3.0
    div_clip: float = 2.0


@dataclass
class SequenceWeightConfig(BaseConfig):
    """Config for the MLP "meta-critic" sequence-level weighting (VRPO-33).

    A learnable alternative to RBF length weighting: maps a set of per-sequence
    features (length encoding, entropy aggregates, optional group-distributional
    features) to a scalar weight phi that reshapes the GRPO loss. When disabled,
    falls back to standard GRPO behavior (uniform weighting).

    Identity init: the final projection is zero-initialized so phi == 1 at step 0
    (i.e. plain GRPO), mirroring RBF's alpha_init=0 behavior.
    """
    enable: bool = False

    # --- v1 features, ON by default (evidence-backed) ---
    use_length: bool = True             # log-sine length encoding (dominant hack axis)
    use_entropy: bool = True            # mean entropy + high-entropy-token fraction (80/20 forking tokens)
    use_pass_rate: bool = True          # group difficulty = fraction correct in GRPO group
    use_repetition: bool = True         # n-gram redundancy (DAPO-named "repetitive" failure); needs response_ids
    use_truncation: bool = True         # binary: response fills the whole window (overlong/truncated, cf. DAPO overlong filtering)

    # --- speculative features, OFF by default (cheap + removable, but unvalidated/gameable) ---
    use_semantic_drift: bool = False    # 1 - cos(prompt_hidden, mean response hidden); needs hidden_states
    use_accumulated_kl: bool = False    # mean |logpi - logpi_ref| per seq (learned per-sample KL beta); needs ref_log_prob

    # feature hyperparameters
    length_fourier_dims: int = 8        # Fourier basis dims for length (even)
    high_entropy_quantile: float = 0.8  # token counted "high-entropy" above this per-seq quantile
    repetition_ngram: int = 3           # n for n-gram redundancy

    # MLP architecture (mirrors MLPTokenWeightModule / VRPO-29)
    scalar_embed_dim: int = 16
    mlp_hidden_dim: int = 512

    # Output normalization / stability
    phi_clip_min: float = 0.2
    phi_clip_max: float = 5.0

    # Optimizer (own param group on the actor optimizer)
    lr: float = 0.001
    weight_decay: float = 1e-4

    # Anti-collapse aux loss (VRPO-34); coef=0 disables. Penalizes phi collapsing
    # to uniform OR to a few samples via a variance band around target_phi_std.
    collapse_reg_coef: float = 0.0
    target_phi_std: float = 0.3


@dataclass
class LOTISConfig(BaseConfig):
    """Main LOTIS configuration combining length and TIS weighting."""
    length_weight: LengthWeightConfig = field(default_factory=LengthWeightConfig)
    token_weight: TokenWeightConfig = field(default_factory=TokenWeightConfig)
    sequence_weight: SequenceWeightConfig = field(default_factory=SequenceWeightConfig)

    @property
    def is_enabled(self) -> bool:
        """Returns True if any LOTIS component is enabled."""
        return self.length_weight.enable or self.token_weight.enable or self.sequence_weight.enable
