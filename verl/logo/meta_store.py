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

    Entries start with no prior (alpha=0, beta=0).  The first observation
    (preflight or training) defines the value directly: V = reward.
    Subsequent observations blend via decay.

    Supports two value tracking modes:
    - "bayesian": Tracks Beta(alpha, beta) distribution, V(x) = alpha / (alpha + beta)
    - "ema": Tracks simple exponential moving average V(x)
    """

    # Entry initialization: no prior for Bayesian — first observation defines the distribution.
    # Lake prompts (never visited) use VALUE_INIT as a neutral default.
    ALPHA_INIT = 0.0   # Bayesian entries start empty (no pseudo-observations)
    BETA_INIT = 0.0    # Bayesian entries start empty
    VALUE_INIT = 0.5   # Default value for Lake queries and initial v_old

    def __init__(self, mode: str = "bayesian"):
        if mode not in ("bayesian", "ema"):
            raise ValueError(f"mode must be 'bayesian' or 'ema', got {mode}")
        self.mode = mode
        self._store: dict = {}
        self._all_prompt_ids: list = []
        self._embeddings: dict = {}  # prompt_id -> embedding (works for Memory AND Lake)

        # Step-level accumulators (reset each update() call, read by get_statistics)
        self._last_update_metrics: dict = {}

    # ------------------------------------------------------------------
    # Dataset Registration & Memory/Lake Topology
    # ------------------------------------------------------------------

    def register_dataset(self, prompt_ids: list):
        """Register all prompt IDs from the dataset.

        This defines the full universe of prompts (Memory + Lake).
        Prompts are only added to Memory upon first update().
        """
        self._all_prompt_ids = list(prompt_ids)

    def memory(self) -> list:
        """Return Memory: prompt IDs with rollout data."""
        return list(self._store.keys())

    def lake(self) -> list:
        """Return Lake: registered prompt IDs not yet in _store."""
        store_set = set(self._store.keys())
        return [pid for pid in self._all_prompt_ids if pid not in store_set]

    def _fresh_entry(self) -> dict:
        """Create a fresh entry with no prior.

        For Bayesian mode: alpha=0, beta=0 so the first observation(s)
        define the distribution entirely (no Beta(1,1) pseudo-counts).
        For EMA mode: starts at VALUE_INIT.
        """
        return {
            "alpha": 0.0,
            "beta": 0.0,
            "value": 0.0,  # placeholder v_old for first update
            "n_obs": 0,
            "last_step": 0,
            "last_logprob": 0.0,
        }

    def _lazy_initialize(self, prompt_id):
        """Lazily add a prompt to the store upon first visit."""
        if prompt_id not in self._store:
            self._store[prompt_id] = self._fresh_entry()

    def __len__(self):
        return len(self._store)

    def __contains__(self, prompt_id) -> bool:
        return prompt_id in self._store

    # ------------------------------------------------------------------
    # Value queries
    # ------------------------------------------------------------------

    def get_value(self, prompt_ids: list) -> torch.Tensor:
        """Return value estimate for each prompt.

        Memory prompts: return stored value.
        Lake prompts: return VALUE_INIT (neutral 0.5).
        """
        values = []
        for pid in prompt_ids:
            entry = self._store.get(pid)
            if entry is None:
                values.append(self.value_init)
            else:
                values.append(entry["value"])
        return torch.tensor(values, dtype=torch.float32)

    def get_alpha_beta(self, prompt_ids: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return alpha/beta parameters (only meaningful in bayesian mode).

        Lake prompts return (0, 0) — no prior information.
        """
        alphas, betas = [], []
        for pid in prompt_ids:
            entry = self._store.get(pid)
            if entry is None:
                alphas.append(0.0)
                betas.append(0.0)
            else:
                if self.mode == "bayesian":
                    alphas.append(entry["alpha"])
                    betas.append(entry["beta"])
                else:
                    v = entry["value"]
                    alphas.append(v)
                    betas.append(1.0 - v)
        return torch.tensor(alphas, dtype=torch.float32), torch.tensor(betas, dtype=torch.float32)

    def get_variance(self, prompt_ids: list) -> torch.Tensor:
        """Return variance/uncertainty estimate for each prompt.

        Bayesian: Beta distribution variance = ab / ((a+b)^2 (a+b+1))
        Uninitialised (alpha=beta=0) or EMA: V(1-V) as simple proxy
        """
        values = self.get_value(prompt_ids).numpy()
        if self.mode == "bayesian":
            alphas, betas = self.get_alpha_beta(prompt_ids)
            alphas_np = alphas.numpy()
            betas_np = betas.numpy()
            ab_sum = alphas_np + betas_np
            # For uninitialised entries (alpha=beta=0), fall back to V*(1-V)
            valid = ab_sum > 0
            variance = np.where(
                valid,
                (alphas_np * betas_np) / (ab_sum ** 2 * (ab_sum + 1)),
                values * (1.0 - values),
            )
        else:
            variance = values * (1.0 - values)
        return torch.tensor(variance, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update(
        self,
        prompt_ids: list,
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

        grouped = defaultdict(list)
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
                if entry["n_obs"] == 0:
                    # First observation: set value directly from reward
                    entry["value"] = np.clip(r_bar, 0.0, 1.0)
                else:
                    v_new = decay * v_old + (1.0 - decay) * r_bar
                    entry["value"] = np.clip(v_new, 0.0, 1.0)

            v_new = entry["value"]
            uncertainties.append(math.sqrt(v_new * (1.0 - v_new)))

            entry["n_obs"] += len(rs)
            entry["last_step"] = step

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

    def update_logprobs(self, prompt_ids: list, logprobs: List[float]):
        """Update last_logprob for prompts in the store.

        Called after meta_store.update() with per-prompt mean logprobs
        from the rollout, so the confidence gap in scoring is meaningful.
        """
        for pid, lp in zip(prompt_ids, logprobs):
            entry = self._store.get(pid)
            if entry is not None:
                entry["last_logprob"] = float(lp)

    # ------------------------------------------------------------------
    # Sampling scores
    # ------------------------------------------------------------------

    def compute_sampling_scores(
        self,
        prompt_ids: list,
        current_step: int,
        rho: float = 1.0,
        staleness_bonus: float = 0.01,
        lake_value_estimates: Optional[Dict[int, float]] = None,
    ) -> torch.Tensor:
        """Priority score for prompt sampling.

        Bayesian mode (Thompson sampling):
            Sample p_tilde ~ Beta(alpha, beta).
            S = sqrt(p_tilde*(1-p_tilde)) + rho * |p_tilde - e^{-NLL_last}| + staleness
            Exploration comes from posterior width: uncertain prompts produce
            more variable p_tilde samples -> occasionally high scores.

        EMA mode (deterministic):
            S = sqrt(V(1-V)) + rho * |V - e^{-NLL_last}| + staleness.
            (Exploration handled by sampler-level epsilon-greedy, not here.)

        Lake prompts with kNN estimates: variance from extrapolated V + staleness.
        Lake prompts without estimates: staleness only (low priority).
        Empty store: uniform scores (stochastic strategy).
        """
        if not self._store:
            return torch.ones(len(prompt_ids), dtype=torch.float32)

        scores = []
        for pid in prompt_ids:
            entry = self._store.get(pid)
            if entry is not None:
                # Memory prompt
                if self.mode == "bayesian":
                    # Thompson sampling: wider posterior -> more exploration
                    p = float(np.random.beta(entry["alpha"], entry["beta"]))
                else:
                    p = entry["value"]

                var_score = math.sqrt(p * (1.0 - p))
                p_ref = math.exp(entry["last_logprob"]) if entry["last_logprob"] < 0 else 0.5
                conf_gap = abs(p - p_ref)
                staleness = staleness_bonus * (current_step - entry["last_step"])
                scores.append(var_score + rho * conf_gap + staleness)
            else:
                # Lake prompt: priority only from kNN estimates or staleness.
                # No blind variance bonus — uninformed prompts get low priority.
                staleness = staleness_bonus * current_step
                if lake_value_estimates and pid in lake_value_estimates:
                    v = lake_value_estimates[pid]
                    var_score = math.sqrt(v * (1.0 - v))
                    scores.append(var_score + staleness)
                else:
                    # No information at all — only staleness drives priority
                    scores.append(staleness)

        return torch.tensor(scores, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Embeddings (for RAG miner)
    # ------------------------------------------------------------------

    def set_embeddings(self, prompt_ids: list, embeddings: torch.Tensor):
        """Store prompt embeddings (detached, on CPU).

        Works for any registered prompt (Memory or Lake).
        """
        embeddings_cpu = embeddings.detach().cpu()
        for i, pid in enumerate(prompt_ids):
            self._embeddings[pid] = embeddings_cpu[i]

    def get_embeddings(self, prompt_ids: list) -> Optional[torch.Tensor]:
        """Retrieve embeddings; returns None if any are missing."""
        embs = []
        for pid in prompt_ids:
            emb = self._embeddings.get(pid)
            if emb is None:
                return None
            embs.append(emb)
        return torch.stack(embs)

    def prompts_without_embeddings(self) -> list:
        """Return registered prompt IDs that don't have embeddings yet."""
        return [pid for pid in self._all_prompt_ids if pid not in self._embeddings]

    # ------------------------------------------------------------------
    # Legacy aliases
    # ------------------------------------------------------------------

    def visited_prompt_ids(self) -> list:
        """Alias for memory(). Deprecated: use memory() instead."""
        return self.memory()

    def all_prompt_ids(self) -> list:
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
            "embeddings": dict(self._embeddings),
        }

    def load_state_dict(self, sd: dict):
        self.mode = sd.get("mode", "bayesian")
        self._all_prompt_ids = sd.get("all_prompt_ids", [])
        self._store = sd["store"]
        self._embeddings = sd.get("embeddings", {})

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
