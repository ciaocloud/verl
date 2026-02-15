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
    - Memory: Visited prompts with empirical rollout data (_store)
    - Lake: Unvisited prompts (tracked via all_prompt_ids, not in _store)

    Supports two value tracking modes:
    - "bayesian": Tracks Beta(alpha, beta) distribution, V(x) = alpha / (alpha + beta)
    - "ema": Tracks simple exponential moving average V(x)

    Lazy initialization: prompts are only added to Memory upon first rollout.
    """

    def __init__(
        self,
        mode: str = "bayesian",
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        value_init: float = 0.5,
    ):
        if mode not in ("bayesian", "ema"):
            raise ValueError(f"mode must be 'bayesian' or 'ema', got {mode}")
        self.mode = mode
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.value_init = value_init
        # Memory: prompt_id -> {alpha, beta, value, last_step, last_logprob, embedding}
        # Only contains prompts that have been rolled out at least once
        self._store: Dict[str, dict] = {}
        # Full dataset of all possible prompt IDs (for Lake computation)
        self._all_prompt_ids: List[str] = []

    # ------------------------------------------------------------------
    # Dataset Registration & Memory/Lake Topology
    # ------------------------------------------------------------------

    def register_dataset(self, prompt_ids: List[str]):
        """Register all prompt IDs from the dataset.

        This defines the full universe of prompts (Memory + Lake).
        Prompts are only added to Memory upon first rollout via update().
        """
        self._all_prompt_ids = list(prompt_ids)

    def memory(self) -> List[str]:
        """Return Memory: prompt IDs with actual rollout data."""
        return list(self._store.keys())

    def lake(self) -> List[str]:
        """Return Lake: prompt IDs not yet visited."""
        memory_set = set(self._store.keys())
        return [pid for pid in self._all_prompt_ids if pid not in memory_set]

    def _lazy_initialize(self, prompt_id: str):
        """Lazily add a prompt to Memory upon first rollout.

        Called internally by update() when a prompt is visited for the first time.
        """
        if prompt_id not in self._store:
            if self.mode == "bayesian":
                a = self.alpha_init
                b = self.beta_init
                self._store[prompt_id] = {
                    "alpha": a,
                    "beta": b,
                    "value": a / (a + b),
                    "last_step": 0,
                    "last_logprob": 0.0,
                    "embedding": None,
                }
            else:  # ema mode
                self._store[prompt_id] = {
                    "alpha": 0.0,  # unused in ema mode
                    "beta": 0.0,
                    "value": self.value_init,
                    "last_step": 0,
                    "last_logprob": 0.0,
                    "embedding": None,
                }

    def __len__(self):
        return len(self._store)

    def __contains__(self, prompt_id: str) -> bool:
        return prompt_id in self._store

    # ------------------------------------------------------------------
    # Value queries
    # ------------------------------------------------------------------

    def get_value(self, prompt_ids: List[str]) -> torch.Tensor:
        """Return value estimate for each prompt.

        For prompts in Memory: return stored value.
        For prompts in Lake: return default (prior).
        """
        values = []
        for pid in prompt_ids:
            entry = self._store.get(pid)
            if entry is None:
                # Prompt in Lake: return prior
                if self.mode == "bayesian":
                    values.append(self.alpha_init / (self.alpha_init + self.beta_init))
                else:
                    values.append(self.value_init)
            else:
                # Prompt in Memory: return stored value
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
                    # In ema mode, synthesize alpha/beta from value for compatibility
                    v = entry["value"]
                    alphas.append(v)
                    betas.append(1.0 - v)
        return torch.tensor(alphas, dtype=torch.float32), torch.tensor(betas, dtype=torch.float32)

    def get_variance(self, prompt_ids: List[str]) -> torch.Tensor:
        """Return variance/uncertainty estimate for each prompt.

        In bayesian mode: Beta distribution variance
        In ema mode: V(1-V) as simple uncertainty proxy
        """
        values = self.get_value(prompt_ids).numpy()
        if self.mode == "bayesian":
            alphas, betas = self.get_alpha_beta(prompt_ids)
            alphas_np = alphas.numpy()
            betas_np = betas.numpy()
            ab_sum = alphas_np + betas_np
            # Var[Beta(a,b)] = ab/((a+b)^2(a+b+1))
            variance = (alphas_np * betas_np) / (ab_sum ** 2 * (ab_sum + 1))
        else:
            # Simple proxy: peaks at 0.5, goes to 0 at extremes
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

        # group rewards by prompt_id
        grouped: Dict[str, List[float]] = defaultdict(list)
        for pid, r in zip(prompt_ids, rewards_np):
            grouped[pid].append(float(r))

        for pid, rs in grouped.items():
            # Lazy initialization: add to Memory if first visit
            if pid not in self._store:
                self._lazy_initialize(pid)

            entry = self._store[pid]
            v_old = entry["value"]
            r_bar = float(np.mean(rs))
            delta_t = max(step - entry["last_step"], 1)

            # Compute decay factor based on mode
            if decay_mode == "fixed":
                # Time-based decay
                decay = gamma ** delta_t
                decay = np.clip(decay, gamma_clip_min, gamma_clip_max)
            elif decay_mode == "adaptive":
                # Drift-based decay (no time component)
                drift = abs(v_old - r_bar)
                decay = 1.0 / (1.0 + sensitivity * drift)
                decay = np.clip(decay, gamma_clip_min, gamma_clip_max)
            else:
                raise ValueError(f"Unknown decay mode: {decay_mode}")

            # Apply update based on value tracking mode
            if self.mode == "bayesian":
                old_alpha = entry["alpha"]
                old_beta = entry["beta"]
                new_alpha = old_alpha * decay + sum(r for r in rs)
                new_beta = old_beta * decay + sum(1.0 - r for r in rs)
                # floor to avoid degenerate distributions
                entry["alpha"] = max(new_alpha, 0.01)
                entry["beta"] = max(new_beta, 0.01)
                entry["value"] = entry["alpha"] / (entry["alpha"] + entry["beta"])
            else:  # ema mode
                v_new = decay * v_old + (1.0 - decay) * r_bar
                entry["value"] = np.clip(v_new, 0.0, 1.0)

            entry["last_step"] = step
            entry["last_logprob"] = 0.0  # updated externally if needed

    # ------------------------------------------------------------------
    # Sampling scores
    # ------------------------------------------------------------------

    def compute_sampling_scores(
        self,
        prompt_ids: List[str],
        current_step: int,
        rho: float = 1.0,
        staleness_bonus: float = 0.01,
    ) -> torch.Tensor:
        """Compute priority score via uncertainty + confidence gap + staleness.

        Memory prompts (in _store):
            Bayesian mode: Draw p_tilde ~ Beta(alpha, beta) (Thompson sampling)
            EMA mode: Use stored value directly
            Score: sqrt(p_tilde * (1-p_tilde)) + rho * |p_tilde - exp(last_logprob)| + staleness

        Lake prompts (not in _store):
            Return max uncertainty score (1.0) to encourage exploration.
        """
        scores = []
        for pid in prompt_ids:
            entry = self._store.get(pid)
            if entry is None:
                # Lake prompt: max uncertainty for exploration
                scores.append(1.0)
                continue

            # Memory prompt: compute score based on stored data
            if self.mode == "bayesian":
                # Thompson sampling from Beta distribution
                a, b = entry["alpha"], entry["beta"]
                p_tilde = float(np.random.beta(a, b))
            else:
                # EMA mode: use stored value directly
                p_tilde = entry["value"]

            # variance proxy: peaks at p=0.5
            var_score = math.sqrt(p_tilde * (1.0 - p_tilde))
            # confidence gap
            p_ref = math.exp(entry["last_logprob"]) if entry["last_logprob"] < 0 else 0.5
            conf_gap = abs(p_tilde - p_ref)
            # staleness
            staleness = staleness_bonus * (current_step - entry["last_step"])
            scores.append(var_score + rho * conf_gap + staleness)

        return torch.tensor(scores, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Embeddings (for RAG miner)
    # ------------------------------------------------------------------

    def set_embeddings(self, prompt_ids: List[str], embeddings: torch.Tensor):
        """Store prompt embeddings (detached, on CPU).

        Only stores embeddings for prompts in Memory.
        Lake prompts are ignored (no rollout data yet).
        """
        embeddings_cpu = embeddings.detach().cpu()
        for i, pid in enumerate(prompt_ids):
            if pid in self._store:  # Only Memory prompts
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
    # Legacy aliases (for backward compatibility)
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
            "alpha_init": self.alpha_init,
            "beta_init": self.beta_init,
            "value_init": self.value_init,
            "all_prompt_ids": self._all_prompt_ids,
            "store": dict(self._store),
        }

    def load_state_dict(self, sd: dict):
        self.mode = sd.get("mode", "bayesian")  # backward compat
        self.alpha_init = sd["alpha_init"]
        self.beta_init = sd["beta_init"]
        self.value_init = sd.get("value_init", 0.5)  # backward compat
        self._all_prompt_ids = sd.get("all_prompt_ids", [])  # backward compat
        self._store = sd["store"]

    # ------------------------------------------------------------------
    # Statistics (for logging)
    # ------------------------------------------------------------------

    def get_statistics(self, current_step: int = 0) -> Dict[str, float]:
        if not self._store:
            return {}
        values = [e["value"] for e in self._store.values()]
        staleness = [current_step - e["last_step"] for e in self._store.values()]
        stats = {
            "logo/meta_store_size": float(len(self._store)),
            "logo/value_mean": float(np.mean(values)),
            "logo/value_std": float(np.std(values)),
            "logo/value_min": float(np.min(values)),
            "logo/value_max": float(np.max(values)),
            "logo/staleness_mean": float(np.mean(staleness)),
            "logo/staleness_max": float(np.max(staleness)),
        }
        if self.mode == "bayesian":
            alphas = [e["alpha"] for e in self._store.values()]
            betas = [e["beta"] for e in self._store.values()]
            stats.update({
                "logo/alpha_mean": float(np.mean(alphas)),
                "logo/beta_mean": float(np.mean(betas)),
                "logo/concentration_mean": float(np.mean([a + b for a, b in zip(alphas, betas)])),
            })
        return stats
