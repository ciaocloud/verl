"""Prompt Meta-Store: per-prompt value tracking for LOGO."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class PromptMetaStore:
    """CPU-resident store mapping prompt IDs to value estimates.

    Implements Memory/Lake topology:
    - Memory: Prompts in ``_store`` with rollout data (used for everything)
    - Lake: Prompts registered but not yet in ``_store`` (scored via RAG miner)

    For Bayesian mode, preflight adds one observation on top of the prior
    (prior + 1 sample).  Natural decay downweights this during training.
    For EMA mode, preflight is skipped (single-sample EMA is just noise).

    Supports two value tracking modes:
    - "bayesian": Tracks Beta(alpha, beta) distribution, V(x) = alpha / (alpha + beta)
    - "ema": Tracks simple exponential moving average V(x)
    """

    # Hardcoded priors
    ALPHA_INIT = 1.0  # Beta distribution prior alpha (bayesian mode)
    BETA_INIT = 1.0   # Beta distribution prior beta (bayesian mode)
    VALUE_INIT = 0.5   # Initial value estimate (ema mode)

    def __init__(self, mode: str = "bayesian"):
        if mode not in ("bayesian", "ema"):
            raise ValueError(f"mode must be 'bayesian' or 'ema', got {mode}")
        self.mode = mode
        self.alpha_init = self.ALPHA_INIT
        self.beta_init = self.BETA_INIT
        self.value_init = self.VALUE_INIT
        self._store: Dict[str, dict] = {}
        self._all_prompt_ids: List[str] = []

        # Step-level accumulators (reset each update() call, read by get_statistics)
        self._last_update_metrics: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Dataset Registration & Memory/Lake Topology
    # ------------------------------------------------------------------

    def register_dataset(self, prompt_ids: List[str]):
        """Register all prompt IDs from the dataset.

        This defines the full universe of prompts (Memory + Lake).
        Prompts are only added to Memory upon first update().
        """
        self._all_prompt_ids = list(prompt_ids)

    def memory(self) -> List[str]:
        """Return Memory: prompt IDs with rollout data."""
        return list(self._store.keys())

    def lake(self) -> List[str]:
        """Return Lake: registered prompt IDs not yet in _store."""
        store_set = set(self._store.keys())
        return [pid for pid in self._all_prompt_ids if pid not in store_set]

    def _fresh_entry(self) -> dict:
        """Create a fresh entry with configured priors."""
        if self.mode == "bayesian":
            a, b = self.alpha_init, self.beta_init
            return {
                "alpha": a,
                "beta": b,
                "value": a / (a + b),
                "n_obs": 0,
                "last_step": 0,
                "last_logprob": 0.0,
                "embedding": None,
            }
        else:  # ema
            return {
                "alpha": 0.0,
                "beta": 0.0,
                "value": self.value_init,
                "n_obs": 0,
                "last_step": 0,
                "last_logprob": 0.0,
                "embedding": None,
            }

    def _lazy_initialize(self, prompt_id: str):
        """Lazily add a prompt to the store upon first visit."""
        if prompt_id not in self._store:
            self._store[prompt_id] = self._fresh_entry()

    def __len__(self):
        return len(self._store)

    def __contains__(self, prompt_id: str) -> bool:
        return prompt_id in self._store

    # ------------------------------------------------------------------
    # Value queries
    # ------------------------------------------------------------------

    def get_value(self, prompt_ids: List[str]) -> torch.Tensor:
        """Return value estimate for each prompt.

        Memory prompts: return stored value.
        Lake prompts: return prior.
        """
        prior = self.alpha_init / (self.alpha_init + self.beta_init) if self.mode == "bayesian" else self.value_init
        values = []
        for pid in prompt_ids:
            entry = self._store.get(pid)
            if entry is None:
                values.append(prior)
            else:
                values.append(entry["value"])
        return torch.tensor(values, dtype=torch.float32)

    def get_alpha_beta(self, prompt_ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return alpha/beta parameters (only meaningful in bayesian mode)."""
        alphas, betas = [], []
        for pid in prompt_ids:
            entry = self._store.get(pid)
            if entry is None:
                alphas.append(self.alpha_init)
                betas.append(self.beta_init)
            else:
                if self.mode == "bayesian":
                    alphas.append(entry["alpha"])
                    betas.append(entry["beta"])
                else:
                    v = entry["value"]
                    alphas.append(v)
                    betas.append(1.0 - v)
        return torch.tensor(alphas, dtype=torch.float32), torch.tensor(betas, dtype=torch.float32)

    def get_variance(self, prompt_ids: List[str]) -> torch.Tensor:
        """Return variance/uncertainty estimate for each prompt.

        Bayesian: Beta distribution variance = ab / ((a+b)^2 (a+b+1))
        EMA: V(1-V) as simple uncertainty proxy
        """
        values = self.get_value(prompt_ids).numpy()
        if self.mode == "bayesian":
            alphas, betas = self.get_alpha_beta(prompt_ids)
            alphas_np = alphas.numpy()
            betas_np = betas.numpy()
            ab_sum = alphas_np + betas_np
            variance = (alphas_np * betas_np) / (ab_sum ** 2 * (ab_sum + 1))
        else:
            variance = values * (1.0 - values)
        return torch.tensor(variance, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update(
        self,
        prompt_ids: List[str],
        rewards: torch.Tensor,
        step: int,
        decay_mode: str = "adaptive",
        gamma: float = 0.99,
        sensitivity: float = 2.0,
        gamma_clip_min: float = 0.1,
        gamma_clip_max: float = 0.95,
    ):
        """Update value estimates for each prompt using per-prompt grouped rewards.

        Args:
            prompt_ids: prompt identifiers (may contain duplicates within a group).
            rewards: 1-D tensor of per-response rewards, same length as prompt_ids.
            step: current training step.
            decay_mode: "fixed" (time-based) or "adaptive" (drift-based).
            gamma: base exponential decay factor (for fixed mode).
            sensitivity: drift sensitivity (for adaptive mode).
            gamma_clip_min/max: clip gamma to prevent extreme decay or memory.
        """
        rewards_np = rewards.detach().cpu().float().numpy()

        grouped: Dict[str, List[float]] = defaultdict(list)
        for pid, r in zip(prompt_ids, rewards_np):
            grouped[pid].append(float(r))

        # Step-level accumulators for metrics
        gammas = []
        drifts = []
        uncertainties = []

        for pid, rs in grouped.items():
            if pid not in self._store:
                self._lazy_initialize(pid)

            entry = self._store[pid]
            v_old = entry["value"]
            r_bar = float(np.mean(rs))
            delta_t = max(step - entry["last_step"], 1)

            # Compute decay factor
            if decay_mode == "fixed":
                decay = gamma ** delta_t
                decay = np.clip(decay, gamma_clip_min, gamma_clip_max)
            elif decay_mode == "adaptive":
                drift = abs(v_old - r_bar)
                decay = 1.0 / (1.0 + sensitivity * drift)
                decay = np.clip(decay, gamma_clip_min, gamma_clip_max)
            else:
                raise ValueError(f"Unknown decay mode: {decay_mode}")

            gammas.append(float(decay))
            drifts.append(abs(v_old - r_bar))

            # Apply update
            if self.mode == "bayesian":
                new_alpha = entry["alpha"] * decay + sum(r for r in rs)
                new_beta = entry["beta"] * decay + sum(1.0 - r for r in rs)
                entry["alpha"] = max(new_alpha, 0.01)
                entry["beta"] = max(new_beta, 0.01)
                entry["value"] = entry["alpha"] / (entry["alpha"] + entry["beta"])
            else:  # ema
                v_new = decay * v_old + (1.0 - decay) * r_bar
                entry["value"] = np.clip(v_new, 0.0, 1.0)

            v_new = entry["value"]
            uncertainties.append(math.sqrt(v_new * (1.0 - v_new)))

            entry["n_obs"] += len(rs)
            entry["last_step"] = step
            entry["last_logprob"] = 0.0  # updated externally if needed

        # Store step-level metrics for get_statistics() to report
        if gammas:
            self._last_update_metrics = {
                "logo/decay_mean": float(np.mean(gammas)),
                "logo/decay_std": float(np.std(gammas)),
                "logo/value_drift": float(np.mean(drifts)),
                "logo/value_uncertainty": float(np.mean(uncertainties)),
            }
        else:
            self._last_update_metrics = {}

    # ------------------------------------------------------------------
    # Sampling scores
    # ------------------------------------------------------------------

    def compute_sampling_scores(
        self,
        prompt_ids: List[str],
        current_step: int,
        rho: float = 1.0,
        staleness_bonus: float = 0.01,
        epsilon: float = 0.1,
        lake_value_estimates: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Priority score for prompt sampling.

        Bayesian mode (Thompson sampling):
            Sample p_tilde ~ Beta(alpha, beta).
            S = sqrt(p_tilde*(1-p_tilde)) + rho * |p_tilde - e^{-NLL_last}| + staleness
            Exploration comes from posterior width: uncertain prompts produce
            more variable p_tilde samples -> occasionally high scores.

        EMA mode (epsilon-greedy):
            With probability epsilon: S = uniform random in [0, 1] (explore).
            Otherwise: S = sqrt(V(1-V)) + rho * |V - e^{-NLL_last}| + staleness.

        Lake prompts: variance from RAG-extrapolated V (or prior) + staleness.
        Empty store: uniform scores (stochastic strategy).
        """
        if not self._store:
            return torch.ones(len(prompt_ids), dtype=torch.float32)

        prior = self.alpha_init / (self.alpha_init + self.beta_init) if self.mode == "bayesian" else self.value_init

        scores = []
        for pid in prompt_ids:
            entry = self._store.get(pid)
            if entry is not None:
                # Memory prompt
                if self.mode == "bayesian":
                    # Thompson sampling: wider posterior -> more exploration
                    p = float(np.random.beta(entry["alpha"], entry["beta"]))
                else:
                    # Epsilon-greedy: occasionally explore randomly
                    if np.random.rand() < epsilon:
                        scores.append(float(np.random.rand()))
                        continue
                    p = entry["value"]

                var_score = math.sqrt(p * (1.0 - p))
                p_ref = math.exp(entry["last_logprob"]) if entry["last_logprob"] < 0 else 0.5
                conf_gap = abs(p - p_ref)
                staleness = staleness_bonus * (current_step - entry["last_step"])
                scores.append(var_score + rho * conf_gap + staleness)
            else:
                # Lake prompt: variance + staleness (no logprob available)
                v = lake_value_estimates.get(pid, prior) if lake_value_estimates else prior
                var_score = math.sqrt(v * (1.0 - v))
                staleness = staleness_bonus * current_step
                scores.append(var_score + staleness)

        return torch.tensor(scores, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Embeddings (for RAG miner)
    # ------------------------------------------------------------------

    def set_embeddings(self, prompt_ids: List[str], embeddings: torch.Tensor):
        """Store prompt embeddings (detached, on CPU).

        Only stores for prompts currently in _store.
        """
        embeddings_cpu = embeddings.detach().cpu()
        for i, pid in enumerate(prompt_ids):
            if pid in self._store:
                self._store[pid]["embedding"] = embeddings_cpu[i]

    def get_embeddings(self, prompt_ids: List[str]) -> Optional[torch.Tensor]:
        """Retrieve embeddings; returns None if any are missing."""
        embs = []
        for pid in prompt_ids:
            entry = self._store.get(pid)
            if entry is None or entry["embedding"] is None:
                return None
            embs.append(entry["embedding"])
        return torch.stack(embs)

    # ------------------------------------------------------------------
    # Legacy aliases
    # ------------------------------------------------------------------

    def visited_prompt_ids(self) -> List[str]:
        """Alias for memory(). Deprecated: use memory() instead."""
        return self.memory()

    def all_prompt_ids(self) -> List[str]:
        """Return all registered prompt IDs (Memory + Lake)."""
        return self._all_prompt_ids

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "all_prompt_ids": self._all_prompt_ids,
            "store": dict(self._store),
        }

    def load_state_dict(self, sd: dict):
        self.mode = sd.get("mode", "bayesian")
        self._all_prompt_ids = sd.get("all_prompt_ids", [])
        self._store = sd["store"]

    # ------------------------------------------------------------------
    # Statistics (for logging)
    # ------------------------------------------------------------------

    def get_statistics(self, current_step: int = 0) -> Dict[str, float]:
        if not self._store:
            return {}
        values = [e["value"] for e in self._store.values()]
        staleness = [current_step - e["last_step"] for e in self._store.values()]
        staleness_sorted = sorted(staleness)

        stats = {
            "logo/store_size": float(len(self._store)),
            "logo/store_value_mean": float(np.mean(values)),
            "logo/store_value_std": float(np.std(values)),
            "logo/store_value_min": float(np.min(values)),
            "logo/store_value_max": float(np.max(values)),
            "logo/staleness_mean": float(np.mean(staleness)),
            "logo/staleness_max": float(np.max(staleness)),
            "logo/staleness_p90": float(staleness_sorted[int(len(staleness_sorted) * 0.9)]),
        }

        # Memory / Lake fraction
        total = len(self._all_prompt_ids) if self._all_prompt_ids else len(self._store)
        stats["logo/memory_fraction"] = float(len(self._store)) / max(total, 1)

        if self.mode == "bayesian":
            alphas = [e["alpha"] for e in self._store.values()]
            betas = [e["beta"] for e in self._store.values()]
            concentrations = [a + b for a, b in zip(alphas, betas)]
            stats.update({
                "logo/alpha_mean": float(np.mean(alphas)),
                "logo/beta_mean": float(np.mean(betas)),
                "logo/concentration_mean": float(np.mean(concentrations)),
                "logo/concentration_std": float(np.std(concentrations)),
            })

        # Include step-level metrics from last update() call
        stats.update(self._last_update_metrics)

        return stats
