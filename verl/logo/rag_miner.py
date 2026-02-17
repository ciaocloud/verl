"""Candidate Miner: two-stage kNN value extrapolation for unvisited prompts."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from verl.logo.meta_store import PromptMetaStore


class RAGMiner:
    """Two-stage candidate mining pipeline for Lake exploration.

    Stage 1 — ``mine()``: Draw ``candidate_batch_size`` prompts from Lake,
    extrapolate values via kNN cosine similarity against Memory embeddings.
    The value cache is **replaced** (not merged) each call so stale guesses
    from prior epochs don't linger.

    Stage 2 — ``frontier()``: Return the union of Memory + Candidates
    (deduplicated). This is the pool the sampler draws from.

    Guessed values are stored in a transient cache and used ONLY for sampling
    priority — never for advantage computation or gradient baselines.
    """

    def __init__(
        self,
        meta_store: PromptMetaStore,
        candidate_batch_size: int = 4096,
        similarity_threshold: float = 0.7,
        k_neighbors: int = 5,
    ):
        self.meta_store = meta_store
        self.candidate_batch_size = candidate_batch_size
        self.similarity_threshold = similarity_threshold
        self.k_neighbors = k_neighbors
        self.value_cache: Dict[str, float] = {}  # transient guessed values for candidates
        self._candidate_ids: List[str] = []  # current candidate set from last mine()

    # ------------------------------------------------------------------
    # Stage 1: Mine candidates from Lake
    # ------------------------------------------------------------------

    def mine(self) -> Dict[str, float]:
        """Draw candidates from Lake and extrapolate values via kNN.

        Replaces the value cache entirely each call (no stale accumulation).

        Returns:
            Dict mapping candidate prompt_id -> guessed value.
        """
        lake = self.meta_store.lake()
        memory_ids = self.meta_store.memory()

        if not lake:
            self.value_cache = {}
            self._candidate_ids = []
            return {}

        n_samples = min(self.candidate_batch_size, len(lake))
        candidate_ids = list(np.random.choice(lake, size=n_samples, replace=False))
        self._candidate_ids = candidate_ids

        if not memory_ids:
            # No Memory to extrapolate from — uniform guess
            guesses = {pid: 0.5 for pid in candidate_ids}
            self.value_cache = guesses
            return guesses

        # Get embeddings
        candidate_embs = self.meta_store.get_embeddings(candidate_ids)
        memory_embs = self.meta_store.get_embeddings(memory_ids)

        if candidate_embs is None or memory_embs is None:
            guesses = {pid: 0.5 for pid in candidate_ids}
            self.value_cache = guesses
            return guesses

        memory_values = self.meta_store.get_value(memory_ids)  # (n_memory,)

        # Move to GPU for similarity computation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cand_embs_d = candidate_embs.float().to(device)
        memory_embs_d = memory_embs.float().to(device)
        memory_values_d = memory_values.to(device)

        # cosine similarity: (n_candidates, n_memory)
        with torch.no_grad():
            cand_norm = F.normalize(cand_embs_d, p=2, dim=-1)
            mem_norm = F.normalize(memory_embs_d, p=2, dim=-1)
            sims = cand_norm @ mem_norm.t()

            guesses = {}
            k = min(self.k_neighbors, len(memory_ids))

            for i, pid in enumerate(candidate_ids):
                topk_sims, topk_idx = sims[i].topk(k)
                threshold_mask = topk_sims >= self.similarity_threshold

                if threshold_mask.sum() == 0:
                    guesses[pid] = memory_values_d.mean().item()
                else:
                    valid_sims = topk_sims[threshold_mask]
                    valid_vals = memory_values_d[topk_idx[threshold_mask]]
                    guesses[pid] = (valid_sims * valid_vals).sum().item() / valid_sims.sum().item()

        # Replace cache entirely (not merge)
        self.value_cache = guesses
        return guesses

    # ------------------------------------------------------------------
    # Stage 2: Build Frontier
    # ------------------------------------------------------------------

    def frontier(self) -> List[str]:
        """Return Memory + Candidates (deduplicated).

        This is the pool the sampler draws from.
        """
        memory_ids = self.meta_store.memory()
        # Deduplicate: candidates that moved to Memory are already there
        candidate_set = set(self._candidate_ids) - set(memory_ids)
        return memory_ids + list(candidate_set)

    # ------------------------------------------------------------------
    # Topology management
    # ------------------------------------------------------------------

    def clear_moved_to_memory(self):
        """Clear cache entries for prompts that moved out of Lake (into store)."""
        for pid in list(self.value_cache.keys()):
            if pid in self.meta_store._store:
                self.value_cache.pop(pid)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, float]:
        memory = self.meta_store.memory()
        lake = self.meta_store.lake()
        total = len(memory) + len(lake)
        frontier = self.frontier()
        return {
            "logo/memory_size": float(len(memory)),
            "logo/lake_size": float(len(lake)),
            "logo/exploration_ratio": len(lake) / total if total > 0 else 0.0,
            "logo/value_cache_size": float(len(self.value_cache)),
            "logo/frontier_size": float(len(frontier)),
            "logo/candidate_count": float(len(self._candidate_ids)),
        }
