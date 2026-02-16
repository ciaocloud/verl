"""LOGO curriculum sampler: variance-based active prompt selection."""

from __future__ import annotations

from collections import Counter
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

        # Per-prompt cumulative sample counts (how many times each index was yielded)
        self._sample_counts: Counter = Counter()

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

        idx_list = indices.tolist()
        self._sample_counts.update(idx_list)
        yield from idx_list

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

    # ------------------------------------------------------------------
    # Statistics (for logging)
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """Return priority score and sample count metrics."""
        stats = {}

        # Priority score metrics
        if self._scores is not None and len(self._scores) > 0:
            scores = self._scores
            stats.update({
                "logo/priority_mean": scores.mean().item(),
                "logo/priority_std": scores.std().item(),
                "logo/priority_max": scores.max().item(),
                "logo/priority_min": scores.min().item(),
            })

            # Entropy: how concentrated the sampling distribution is
            weights = torch.softmax(scores, dim=0)
            log_weights = torch.log(weights + 1e-10)
            entropy = -(weights * log_weights).sum().item()
            max_entropy = float(np.log(len(scores)))
            stats["logo/priority_entropy"] = entropy
            stats["logo/priority_entropy_ratio"] = entropy / max(max_entropy, 1e-10)

            # Top-bottom spread
            k = max(1, len(scores) // 10)
            top_k = torch.topk(scores, k=k).values.mean().item()
            bot_k = torch.topk(scores, k=k, largest=False).values.mean().item()
            stats["logo/priority_top10pct"] = top_k
            stats["logo/priority_bot10pct"] = bot_k

        # Sample count metrics (over visited prompts only)
        if self._sample_counts:
            counts = list(self._sample_counts.values())
            stats["logo/sample_count_mean"] = float(np.mean(counts))
            stats["logo/sample_count_std"] = float(np.std(counts))
            stats["logo/sample_count_max"] = float(np.max(counts))
            stats["logo/sample_count_min"] = float(np.min(counts))
            stats["logo/prompts_sampled"] = float(len(counts))

        return stats
