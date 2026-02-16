"""LOGO curriculum sampler: variance-based active prompt selection."""

from __future__ import annotations

from collections.abc import Sized
from typing import Iterator, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.logo.config import DecayConfig, LOGOConfig, RAGMinerConfig, SamplingConfig
from verl.logo.meta_store import PromptMetaStore


class LOGOCurriculumSampler(AbstractCurriculumSampler):
    """Prompt sampler driven by Thompson-sampling priority scores.

    Each epoch, scores are recomputed for every prompt via
    ``meta_store.compute_sampling_scores``.  Prompts are then drawn either
    by top-K deterministic selection or weighted multinomial.

    The ``update`` callback (called by the trainer after each step) feeds
    rollout rewards back into the meta-store.
    """

    def __init__(self, data_source: Sized, data_config: DictConfig):
        """
        Args:
            data_source: the training dataset.
            data_config: hydra data config (must contain ``logo`` key with LOGOConfig fields,
                or those fields are set later via ``configure``).
        """
        self.data_source = data_source
        self.data_config = data_config
        self.n_prompts = len(data_source)

        # populated via configure() before training starts
        self.meta_store: Optional[PromptMetaStore] = None
        self.sampling_cfg: Optional[SamplingConfig] = None
        self.decay_cfg: Optional[DecayConfig] = None
        self.rag_miner_cfg: Optional[RAGMinerConfig] = None
        self.stochastic_miner = None  # Optional[StochasticMiner]
        self.current_step: int = 0
        self._epoch: int = 0
        self._scores: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Setup (called by trainer before fit)
    # ------------------------------------------------------------------

    def configure(
        self,
        meta_store: PromptMetaStore,
        logo_config: LOGOConfig,
        stochastic_miner=None,
    ):
        """Attach the meta-store, config, and optional miner after construction.

        Registers all prompt IDs with the meta_store to define Memory/Lake topology.
        """
        self.meta_store = meta_store
        self.sampling_cfg = logo_config.sampling
        self.decay_cfg = logo_config.decay
        self.rag_miner_cfg = logo_config.rag_miner
        self.stochastic_miner = stochastic_miner

        # Register all prompts with meta_store (defines Lake initially)
        all_ids = [str(i) for i in range(self.n_prompts)]
        self.meta_store.register_dataset(all_ids)

    # ------------------------------------------------------------------
    # Sampler interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_prompts

    def __iter__(self) -> Iterator[int]:
        """Yield dataset indices for one epoch, ordered by sampling score."""
        if self.meta_store is None:
            # not configured yet -- fall back to sequential
            yield from range(self.n_prompts)
            return

        self._epoch += 1
        self._refresh_lake_cache()
        self._recompute_scores()

        if self.sampling_cfg is not None and self.sampling_cfg.top_k is not None:
            k = min(self.sampling_cfg.top_k, self.n_prompts)
            indices = torch.topk(self._scores, k=k).indices
        else:
            # weighted multinomial (without replacement)
            weights = torch.softmax(self._scores, dim=0)
            indices = torch.multinomial(weights, num_samples=self.n_prompts, replacement=False)

        yield from indices.tolist()

    # ------------------------------------------------------------------
    # Curriculum callback (called by trainer at end of each step)
    # ------------------------------------------------------------------

    def update(self, batch: DataProto) -> None:
        """Feed rollout results back into the meta-store.

        Expects ``batch`` to carry:
            - ``batch["token_level_rewards"]`` or ``batch["token_level_scores"]``
            - ``batch["response_mask"]``
            - ``non_tensor_batch["uid"]`` (prompt ids **before** repeat expansion)
              or ``non_tensor_batch["logo_prompt_id"]`` set explicitly.
        """
        if self.meta_store is None:
            return

        self.current_step += 1

        # --- extract sequence-level reward ---
        if "token_level_rewards" in batch.batch.keys():
            token_rewards = batch.batch["token_level_rewards"]
        elif "token_level_scores" in batch.batch.keys():
            token_rewards = batch.batch["token_level_scores"]
        else:
            return  # nothing to update with

        mask = batch.batch["response_mask"]
        seq_rewards = (token_rewards * mask).sum(dim=-1)  # (bs,)

        # --- prompt ids ---
        if "logo_prompt_id" in batch.non_tensor_batch:
            prompt_ids = batch.non_tensor_batch["logo_prompt_id"].tolist()
        elif "uid" in batch.non_tensor_batch:
            prompt_ids = batch.non_tensor_batch["uid"].tolist()
        else:
            return

        # --- update meta-store ---
        dcfg = self.decay_cfg
        self.meta_store.update(
            prompt_ids=prompt_ids,
            rewards=seq_rewards,
            step=self.current_step,
            decay_mode=dcfg.mode if dcfg else "adaptive",
            gamma=dcfg.gamma if dcfg else 0.99,
            sensitivity=dcfg.sensitivity if dcfg else 2.0,
            gamma_clip_min=dcfg.gamma_clip_min if dcfg else 0.1,
            gamma_clip_max=dcfg.gamma_clip_max if dcfg else 0.95,
        )

        # Clean up miner cache: prompts that moved from Lake to Memory
        if self.stochastic_miner is not None:
            self.stochastic_miner.clear_moved_to_memory()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh_lake_cache(self):
        """Run the stochastic miner to extrapolate values for Lake prompts.

        Called once per epoch before score computation.  Respects
        ``rag_miner.sample_freq`` — only runs every N epochs.
        """
        if self.stochastic_miner is None:
            return
        rcfg = self.rag_miner_cfg
        freq = rcfg.sample_freq if rcfg else 5
        if self._epoch % freq != 0:
            return
        self.stochastic_miner.sample_and_extrapolate()

    def _recompute_scores(self):
        """Recompute Thompson-sampling priority scores for all prompts."""
        all_ids = self.meta_store.all_prompt_ids()
        if not all_ids:
            self._scores = torch.ones(self.n_prompts)
            return
        scfg = self.sampling_cfg
        lake_values = self.stochastic_miner.value_cache if self.stochastic_miner else None
        self._scores = self.meta_store.compute_sampling_scores(
            prompt_ids=all_ids,
            current_step=self.current_step,
            rho=scfg.rho if scfg else 1.0,
            staleness_bonus=scfg.staleness_bonus if scfg else 0.01,
            epsilon=scfg.epsilon if scfg else 0.1,
            lake_value_estimates=lake_values,
        )
