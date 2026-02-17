"""LOGO curriculum sampler: variance-based active prompt selection."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sized
from typing import Iterator, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.logo.config import DecayConfig, LOGOConfig, MinerConfig, SamplingConfig
from verl.logo.meta_store import PromptMetaStore


class LOGOCurriculumSampler(AbstractCurriculumSampler):
    """Prompt sampler driven by Thompson-sampling priority scores.

    Each epoch, the candidate miner scouts Lake prompts, builds a Frontier
    (Memory + Candidates), and scores are computed over the Frontier.
    Prompts are drawn by weighted multinomial from the Frontier.

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
        self.miner_cfg: Optional[MinerConfig] = None
        self.candidate_miner = None  # Optional[CandidateMiner]
        self.current_step: int = 0
        self._epoch: int = 0
        self._scores: Optional[torch.Tensor] = None

        # Per-prompt cumulative sample counts (how many times each index was yielded)
        self._sample_counts: Counter = Counter()
        # Prompt IDs from the last batch (for batch-level metrics)
        self._last_batch_prompt_ids: Optional[list] = None

    # ------------------------------------------------------------------
    # Setup (called by trainer before fit)
    # ------------------------------------------------------------------

    def configure(
        self,
        meta_store: PromptMetaStore,
        logo_config: LOGOConfig,
        candidate_miner=None,
    ):
        """Attach the meta-store, config, and optional miner after construction.

        Registers all prompt IDs with the meta_store to define Memory/Lake topology.
        """
        self.meta_store = meta_store
        self.sampling_cfg = logo_config.sampling
        self.decay_cfg = logo_config.decay
        self.miner_cfg = logo_config.miner
        self.candidate_miner = candidate_miner

        # Register all prompts with meta_store (defines Lake initially)
        all_ids = [str(i) for i in range(self.n_prompts)]
        self.meta_store.register_dataset(all_ids)

    # ------------------------------------------------------------------
    # Sampler interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_prompts

    def __iter__(self) -> Iterator[int]:
        """Yield n_prompts dataset indices drawn from the Frontier.

        Pipeline:
        1. candidate_miner.mine() scouts Lake candidates via kNN
        2. candidate_miner.frontier() returns Memory + Candidates
        3. Score the Frontier via meta_store.compute_sampling_scores
        4. Sample n_prompts indices from Frontier (with replacement if
           |Frontier| < n_prompts)
        5. Epsilon-greedy: each selected index may be replaced with a
           uniform random index from the full dataset

        When no miner is configured, falls back to scoring all prompts
        (original behavior).
        """
        if self.meta_store is None:
            # not configured yet -- fall back to sequential
            yield from range(self.n_prompts)
            return

        self._epoch += 1

        if self.candidate_miner is not None:
            # Stage 1: Mine candidates from Lake
            self.candidate_miner.mine()
            # Stage 2: Build Frontier = Memory + Candidates
            frontier_ids = self.candidate_miner.frontier()

            if not frontier_ids:
                # No frontier (empty dataset) — fall back to sequential
                yield from range(self.n_prompts)
                return

            # Score the Frontier
            self._recompute_scores(prompt_ids=frontier_ids)

            # Build index mapping: frontier position -> dataset index
            frontier_dataset_indices = [int(pid) for pid in frontier_ids]

            # Sample n_prompts from the Frontier
            weights = torch.softmax(self._scores, dim=0)
            use_replacement = len(frontier_ids) < self.n_prompts
            sampled_positions = torch.multinomial(
                weights, num_samples=self.n_prompts, replacement=use_replacement
            )
            idx_list = [frontier_dataset_indices[pos] for pos in sampled_positions.tolist()]
        else:
            # No miner — score all prompts (original behavior)
            self._recompute_scores()

            if self.sampling_cfg is not None and self.sampling_cfg.top_k is not None:
                k = min(self.sampling_cfg.top_k, self.n_prompts)
                indices = torch.topk(self._scores, k=k).indices
            else:
                weights = torch.softmax(self._scores, dim=0)
                indices = torch.multinomial(weights, num_samples=self.n_prompts, replacement=False)

            idx_list = indices.tolist()

        # Sampler-level epsilon-greedy: replace some indices with uniform
        # random from the full dataset
        eps = self.sampling_cfg.epsilon if self.sampling_cfg else 0.1
        if eps > 0:
            for i in range(len(idx_list)):
                if np.random.rand() < eps:
                    idx_list[i] = np.random.randint(0, self.n_prompts)

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

        # Track unique prompts actually trained on this step
        # (deduplicate because rollout.n > 1 repeats prompt_ids)
        if "logo_prompt_id" in batch.non_tensor_batch:
            self._sample_counts.update(set(batch.non_tensor_batch["logo_prompt_id"].tolist()))

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

        # Save unique prompt IDs for batch-level metrics in get_statistics()
        self._last_batch_prompt_ids = list(dict.fromkeys(prompt_ids))

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

        # --- update logprobs for confidence gap scoring ---
        log_probs = None
        if "old_log_probs" in batch.batch:
            log_probs = batch.batch["old_log_probs"]
        elif "rollout_log_probs" in batch.batch:
            log_probs = batch.batch["rollout_log_probs"]

        if log_probs is not None:
            mask_sum = mask.sum(dim=-1).clamp(min=1)
            mean_lp = (log_probs * mask).sum(dim=-1) / mask_sum
            mean_lp_np = mean_lp.detach().cpu().float().numpy()
            # Group by prompt_id and average (rollout.n > 1 repeats)
            grouped_lp = defaultdict(list)
            for pid, lp in zip(prompt_ids, mean_lp_np):
                grouped_lp[pid].append(float(lp))
            unique_ids = list(grouped_lp.keys())
            avg_lps = [float(np.mean(grouped_lp[pid])) for pid in unique_ids]
            self.meta_store.update_logprobs(unique_ids, avg_lps)

        # Clean up miner cache: prompts that moved from Lake to Memory
        if self.candidate_miner is not None:
            self.candidate_miner.clear_moved_to_memory()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recompute_scores(self, prompt_ids=None):
        """Recompute Thompson-sampling priority scores.

        Args:
            prompt_ids: If provided, score only these prompts (Frontier).
                        Otherwise score all prompts.
        """
        if prompt_ids is None:
            prompt_ids = self.meta_store.all_prompt_ids()
        if not prompt_ids:
            self._scores = torch.ones(self.n_prompts)
            return
        scfg = self.sampling_cfg
        lake_values = self.candidate_miner.value_cache if self.candidate_miner else None
        self._scores = self.meta_store.compute_sampling_scores(
            prompt_ids=prompt_ids,
            current_step=self.current_step,
            rho=scfg.rho if scfg else 1.0,
            staleness_bonus=scfg.staleness_bonus if scfg else 0.01,
            lake_value_estimates=lake_values,
        )

    # ------------------------------------------------------------------
    # Statistics (for logging)
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """Return priority score and sample count metrics.

        Recomputes scores from current meta-store state so metrics reflect
        step-by-step updates, not just the stale epoch-start snapshot.
        """
        stats = {}

        # Recompute scores from current meta-store state
        if self.meta_store is not None:
            self._recompute_scores()

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

        # Batch-level priority metrics (scores for prompts in the current step's batch)
        if self._last_batch_prompt_ids and self.meta_store is not None:
            scfg = self.sampling_cfg
            lake_values = self.candidate_miner.value_cache if self.candidate_miner else None
            batch_scores = self.meta_store.compute_sampling_scores(
                prompt_ids=self._last_batch_prompt_ids,
                current_step=self.current_step,
                rho=scfg.rho if scfg else 1.0,
                staleness_bonus=scfg.staleness_bonus if scfg else 0.01,
                lake_value_estimates=lake_values,
            )
            stats.update({
                "logo/batch_priority_mean": batch_scores.mean().item(),
                "logo/batch_priority_std": batch_scores.std().item(),
                "logo/batch_priority_max": batch_scores.max().item(),
                "logo/batch_priority_min": batch_scores.min().item(),
            })

        # Sample count metrics (over visited prompts only)
        if self._sample_counts:
            counts = list(self._sample_counts.values())
            stats["logo/sample_count_mean"] = float(np.mean(counts))
            stats["logo/sample_count_std"] = float(np.std(counts))
            stats["logo/sample_count_max"] = float(np.max(counts))
            stats["logo/sample_count_min"] = float(np.min(counts))
            stats["logo/prompts_sampled"] = float(len(counts))

        return stats
