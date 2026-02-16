"""Stochastic Miner: RAG-based zero-shot value extrapolation for unvisited prompts."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F

from verl.logo.meta_store import PromptMetaStore


class StochasticMiner:
    """Extrapolates value estimates from visited (Memory) to unvisited (Lake) prompts.

    Uses cosine similarity on reference-policy embeddings to find nearest
    neighbours in Memory, then takes a weighted average of their values.

    Guessed values are stored in a transient cache and used ONLY for sampling
    priority -- never for advantage computation or gradient baselines.
    """

    def __init__(
        self,
        meta_store: PromptMetaStore,
        similarity_threshold: float = 0.7,
        k_neighbors: int = 5,
    ):
        self.meta_store = meta_store
        self.similarity_threshold = similarity_threshold
        self.k_neighbors = k_neighbors
        self.value_cache: Dict[str, float] = {}  # transient guessed values for Lake prompts

    # ------------------------------------------------------------------
    # Topology management
    # ------------------------------------------------------------------

    def compute_lake_sample_size(self, base_size: int = 128) -> int:
        """M ∝ |L| / (|M| + |L|): decaying exploration schedule."""
        memory = self.meta_store.memory()
        lake = self.meta_store.lake()
        total = len(memory) + len(lake)
        if total == 0:
            return 0
        ratio = len(lake) / total
        return max(1, int(ratio * base_size))

    def clear_moved_to_memory(self):
        """Clear cache entries for prompts that moved out of Lake (into store)."""
        for pid in list(self.value_cache.keys()):
            if pid in self.meta_store._store:
                self.value_cache.pop(pid)

    # ------------------------------------------------------------------
    # Value extrapolation
    # ------------------------------------------------------------------

    def sample_and_extrapolate(self, n_samples: Optional[int] = None) -> Dict[str, float]:
        """Sample from Lake and extrapolate values via KNN on embeddings.

        Uses Memory prompts as RAG source neighbors.
        Similarity computation runs on GPU when available.

        Returns:
            Dict mapping lake prompt_id -> guessed value (also stored in cache).
        """
        lake = self.meta_store.lake()
        memory_ids = self.meta_store.memory()

        if not lake or not memory_ids:
            return {}

        if n_samples is None:
            n_samples = self.compute_lake_sample_size()
        n_samples = min(n_samples, len(lake))
        if n_samples == 0:
            return {}

        lake_ids = list(np.random.choice(lake, size=n_samples, replace=False))

        # get embeddings
        lake_embs = self.meta_store.get_embeddings(lake_ids)
        memory_embs = self.meta_store.get_embeddings(memory_ids)

        if lake_embs is None or memory_embs is None:
            # embeddings not available -- return uniform guess
            guesses = {pid: 0.5 for pid in lake_ids}
            self.value_cache.update(guesses)
            return guesses

        memory_values = self.meta_store.get_value(memory_ids)  # (n_memory,)

        # Move to GPU for similarity computation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lake_embs_d = lake_embs.float().to(device)
        memory_embs_d = memory_embs.float().to(device)
        memory_values_d = memory_values.to(device)

        # cosine similarity: (n_lake, n_memory) — computed on GPU
        with torch.no_grad():
            lake_norm = F.normalize(lake_embs_d, p=2, dim=-1)
            mem_norm = F.normalize(memory_embs_d, p=2, dim=-1)
            sims = lake_norm @ mem_norm.t()

            guesses = {}
            k = min(self.k_neighbors, len(memory_ids))

            for i, pid in enumerate(lake_ids):
                topk_sims, topk_idx = sims[i].topk(k)
                threshold_mask = topk_sims >= self.similarity_threshold

                if threshold_mask.sum() == 0:
                    guesses[pid] = memory_values_d.mean().item()
                else:
                    valid_sims = topk_sims[threshold_mask]
                    valid_vals = memory_values_d[topk_idx[threshold_mask]]
                    guesses[pid] = (valid_sims * valid_vals).sum().item() / valid_sims.sum().item()

        self.value_cache.update(guesses)
        return guesses

    def get_cached_value(self, prompt_id: str) -> Optional[float]:
        return self.value_cache.get(prompt_id)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, float]:
        memory = self.meta_store.memory()
        lake = self.meta_store.lake()
        total = len(memory) + len(lake)
        return {
            "logo/memory_size": float(len(memory)),
            "logo/lake_size": float(len(lake)),
            "logo/exploration_ratio": len(lake) / total if total > 0 else 0.0,
            "logo/value_cache_size": float(len(self.value_cache)),
        }
