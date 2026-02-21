"""LOGO curriculum sampler: variance-based active prompt selection."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sized
from typing import Dict, Iterator, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.logo.config import LOGOConfig
from verl.logo.meta_store import PromptMetaStore
from verl.logo.rag_miner import ValuePropagator


class IndexedDataset(torch.utils.data.Dataset):
    """Thin wrapper that injects the dataset index as prompt_index for LOGO sampling."""

    INDEX_KEY = "prompt_index"

    def __init__(self, dataset: torch.utils.data.Dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        item = self._dataset[idx]
        item[self.INDEX_KEY] = idx
        return item

    def __getattr__(self, name):
        return getattr(self._dataset, name)

class LOGOCurriculumSampler(AbstractCurriculumSampler):
    """Prompt sampler driven by Thompson-sampling priority scores.

    Active selection: each epoch yields exactly ``batch_size`` indices
    sampled WITH replacement from priority-weighted scores.  High-priority
    prompts can be revisited multiple times while low-priority ones are
    skipped.  This means one epoch = one training step; control total
    training steps via ``trainer.total_training_steps`` or a large
    ``trainer.total_epochs``.

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
        self.batch_size = int(data_config.get("gen_batch_size", data_config.train_batch_size))

        self.store = PromptMetaStore(mode=data_config.logo.meta_store_mode)
        self.propagator: Optional[ValuePropagator] = None
        if data_config.logo.propagator.enable:
            self.propagator = ValuePropagator(
                meta_store=self.store,
                candidate_batch_size=data_config.logo.propagator.candidate_batch_size,
                similarity_threshold=data_config.logo.propagator.similarity_threshold,
                k_neighbors=data_config.logo.propagator.k_neighbors,
            )
        
        self.current_step: int = 0
        # self._epoch: int = 0
        self._scores: Optional[torch.Tensor] = None

        # # Per-prompt cumulative sample counts (how many times each index was yielded)
        self._sample_counts: Counter = Counter()
        self._unique_prompts_seen: set = set()
        # # Prompt IDs from the last batch (for batch-level metrics)
        self._last_batch_prompt_ids: Optional[list] = None

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
        return self.batch_size

    def __iter__(self) -> Iterator[int]:
        """Yield ``batch_size`` dataset indices via active selection.

        Pipeline:
        1. (Optional) candidate_miner.mine() scouts Lake candidates via kNN
        2. Score prompts via meta_store.compute_sampling_scores
        3. Sample ``batch_size`` indices WITH replacement from softmax(scores/T)
        4. Epsilon-greedy: each selected index may be replaced with a
           uniform random index from the full dataset
        """
        n_samples = self.batch_size
        temperature = self.sampling_cfg.temperature if self.sampling_cfg else 1.0

        if self.propagator is not None:
            self.propagator.propagate()
        lake_vals = self.propagator.value_cache if self.propagator else None
        self._scores = self.meta_store.compute_sampling_scores(
            prompt_ids=self.meta_store.all_prompt_ids(),
            current_step=self.current_step,
            rho=self.sampling_cfg.rho if self.sampling_cfg else 1.0,
            staleness_bonus=self.sampling_cfg.staleness_bonus if self.sampling_cfg else 0.01,
            lake_value_estimates=lake_vals,
        )
        if self.propagator is not None:
            frontier_ids = self.propagator.frontier()
            if len(frontier_ids) == 0:
                yield from np.random.randint(0, self.n_prompts, size=n_samples).tolist()
                return
            frontier_scores = self._scores[frontier_ids]
            weights = torch.softmax(frontier_scores / temperature, dim=0)
            sampled_positions = torch.multinomial(weights, num_samples=n_samples, replacement=True)
            idx_list = [frontier_ids[pos] for pos in sampled_positions.tolist()]
        else:
            weights = torch.softmax(self._scores / temperature, dim=0)
            indices = torch.multinomial(weights, num_samples=n_samples, replacement=True)
            idx_list = indices.tolist()

        # Sampler-level epsilon-greedy
        eps = self.sampling_cfg.epsilon if self.sampling_cfg else 0.0
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
        prompt_ids = self._extract_prompt_ids(batch)
        if prompt_ids is None:
            return
        
        # if "logo_prompt_id" in batch.non_tensor_batch:
        self._sample_counts.update(set(prompt_ids))
        self._unique_prompts_seen.update(set(prompt_ids))

        # --- extract sequence-level reward ---
        if "token_level_rewards" in batch.batch.keys():
            token_rewards = batch.batch["token_level_rewards"]
        elif "token_level_scores" in batch.batch.keys():
            token_rewards = batch.batch["token_level_scores"]
        else:
            return  # nothing to update with

        mask = batch.batch["response_mask"]
        seq_rewards = (token_rewards * mask).sum(dim=-1)  # (bs,)

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
        if self.propagator is not None:
            self.propagator.clear_moved_to_memory()

    # ------------------------------------------------------------------
    # Batch preparation (called by trainer before advantage computation)
    # ------------------------------------------------------------------

    # def prepare_batch(self, batch: DataProto):
    #     """Inject ``v_stored`` into the batch for the LOGO advantage estimator."""
    #     prompt_ids = self._extract_prompt_ids(batch)
    #     if prompt_ids is not None:
    #         batch.batch["v_stored"] = self.meta_store.get_values(prompt_ids)

    @staticmethod
    def _extract_prompt_ids(batch: DataProto) -> Optional[list]:
        if IndexedDataset.INDEX_KEY in batch.non_tensor_batch:
            return batch.non_tensor_batch[IndexedDataset.INDEX_KEY].tolist()
        elif "uid" in batch.non_tensor_batch:
            return batch.non_tensor_batch["uid"].tolist()
        else:
            return None

    # ------------------------------------------------------------------
    # Metrics (for logging)
    # ------------------------------------------------------------------
    def collect_metrics(self, batch: DataProto, step: int) -> Dict[str, float]:
        """Gather all LOGO metrics for logging."""
        metrics: Dict[str, float] = {}

        # Advantage metrics (populated by compute_logo_advantage via meta_info)
        for k, v in batch.meta_info.items():
            if k.startswith("logo/"):
                metrics[k] = v
        # Value vs reward correlation
        if "v_stored" in batch.batch and "token_level_scores" in batch.batch:
            v_s = batch.batch["v_stored"].float()
            r_s = batch.batch["token_level_scores"].sum(dim=-1).float()
            if v_s.std() > 1e-8 and r_s.std() > 1e-8:
                metrics["logo/value_reward_corr"] = torch.corrcoef(torch.stack([v_s, r_s]))[0, 1].item()
            metrics["logo/value_prediction_error"] = (v_s - r_s).pow(2).mean().item()
            metrics["logo/batch_reward_mean"] = r_s.mean().item()

        metrics["logo/unique_prompts_seen"] = float(len(self._unique_prompts_seen))

        # Meta-store statistics
        metrics.update(self.meta_store.get_statistics(current_step=step))

        # Priority score statistics
        metrics.update(self.get_statistics())

        # Propagator statistics
        if self.propagator is not None:
            metrics.update(self.propagator.get_statistics())
        return metrics

    def get_statistics(self) -> dict:
        """Return priority score and sample count metrics.

        Uses cached ``_scores`` from the last ``__iter__`` call to avoid
        expensive recomputation and Thompson sampling noise every step.
        """
        stats = {}

        if self._scores is not None and len(self._scores) > 0:
            scores = self._scores
            stats.update({
                "logo/priority_mean": scores.mean().item(),
                "logo/priority_std": scores.std().item(),
                "logo/priority_max": scores.max().item(),
                "logo/priority_min": scores.min().item(),
            })

            # temperature = self.sampling_cfg.temperature if self.sampling_cfg else 1.0
            # weights = torch.softmax(scores / temperature, dim=0)
            # log_weights = torch.log(weights + 1e-10)
            # entropy = -(weights * log_weights).sum().item()
            # max_entropy = float(np.log(len(scores)))
            # stats["logo/priority_entropy"] = entropy
            # stats["logo/priority_entropy_ratio"] = entropy / max(max_entropy, 1e-10)

            # k = max(1, len(scores) // 10)
            # top_k = torch.topk(scores, k=k).values.mean().item()
            # bot_k = torch.topk(scores, k=k, largest=False).values.mean().item()
            # stats["logo/priority_top10pct"] = top_k
            # stats["logo/priority_bot10pct"] = bot_k

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
