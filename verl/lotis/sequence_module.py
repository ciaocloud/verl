"""LOTIS sequence-level "meta-critic" weighting module (VRPO-33).

A learnable MLP alternative to ``RBFLengthWeightModule``. It maps per-sequence
features to a scalar weight phi that reshapes the GRPO policy-gradient loss
(same role and output contract as the RBF module: returns ``phi: (B,)`` plus a
metrics dict, and slots into the ``lotis_length_module`` call site in
``dp_actor.py``).

Design rationale and the evidence behind each feature are in
``lab/lotis/VRPO-33-meta-critic-investigation.md`` and ``VRPO-33-references.md``.
v1 feature set (all flag-gated; flip a config bool to remove any):

  ON  (evidence-backed):
    - length        : log-sine encoding (length is the dominant hack axis;
                      length-reward relation is nonlinear -> Fourier basis)
    - entropy       : mean entropy + high-entropy-token fraction (the ~20%
                      "forking token" signal, 2506.01939). NOT variance/Hmax-Hmin.
    - pass_rate     : fraction correct in the GRPO group (difficulty / effective
                      sample size). Replaces group-advantage-variance, which for
                      binary reward is just p(1-p) and cannot tell all-wrong from
                      all-solved. Derived from advantages sign, no extra plumbing.
    - repetition    : n-gram redundancy of the response ids (DAPO names
                      "repetitive words" as the long-sample failure). Needs
                      response token ids passed in (one actor-side wiring change).

  OFF (speculative, cheap + removable; documented caveats):
    - semantic_drift: 1 - cos(prompt_hidden, mean response hidden). Likely
                      low-signal on math (on-topic correct AND incorrect); gameable.
    - accumulated_kl: mean |logpi - logpi_ref| per sequence -> lets the MLP learn a
                      per-sample KL penalty beta. CAVEAT: interacts with the global
                      kl_loss_coef; risk of double-penalizing KL.

  DROPPED: rank-in-group, group-advantage-variance (redundant with the advantage
  GRPO already uses / with pass_rate).

Identity init: the final projection is zero-initialized so phi == 1 at step 0
(plain GRPO), mirroring RBF's ``alpha_init=0``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.lotis.config import SequenceWeightConfig
from verl.lotis.modules import SwiGLU


class MLPSequenceWeightModule(nn.Module):
    """Per-sequence weight phi from a small MLP over sequence-level features.

    Output contract (matches RBFLengthWeightModule):
        forward(...) -> (phi: (B,), metrics: dict[str, float])
        phi is normalized to mean 1 and clipped to [phi_clip_min, phi_clip_max].
    """

    def __init__(self, config: SequenceWeightConfig):
        super().__init__()
        self.config = config
        E = config.scalar_embed_dim

        # (name, raw_dim before projection to E). Each feature builder returns
        # (B, raw_dim); each block is projected to E and concatenated.
        self.feature_specs = []
        if config.use_length:
            # log-sine: [z(logL), L/max_len] + Fourier basis
            self.feature_specs.append(("length", 2 + config.length_fourier_dims))
        if config.use_entropy:
            self.feature_specs.append(("entropy", 2))          # [mean ent, high-ent fraction]
        if config.use_pass_rate:
            self.feature_specs.append(("pass_rate", 1))
        if config.use_repetition:
            self.feature_specs.append(("repetition", 1))
        if config.use_truncation:
            self.feature_specs.append(("truncation", 1))
        if config.use_semantic_drift:
            self.feature_specs.append(("semantic_drift", 1))
        if config.use_accumulated_kl:
            self.feature_specs.append(("accumulated_kl", 1))

        if not self.feature_specs:
            raise ValueError("SequenceWeightConfig: at least one feature must be enabled.")

        self.projections = nn.ModuleDict(
            {name: nn.Linear(raw_dim, E) for name, raw_dim in self.feature_specs}
        )

        in_dim = E * len(self.feature_specs)
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            SwiGLU(in_dim, config.mlp_hidden_dim),
            nn.Linear(config.mlp_hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        """Xavier for hidden layers; ZERO final layer => phi == 1 at init (plain GRPO)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        final = self.mlp[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    # ----- feature builders: each returns (B, raw_dim), detached -----

    @staticmethod
    def _length_features(lengths: torch.Tensor, fourier_dims: int, max_len: int) -> torch.Tensor:
        """log-sine length encoding: [z(log L), L/max_len, Fourier(z log L)...]. (B, 2+fourier_dims)"""
        log_l = torch.log(lengths.clamp(min=1.0))
        z_log = (log_l - log_l.mean()) / (log_l.std().clamp(min=1e-6))
        linear = (lengths / max(max_len, 1)).clamp(0, 1)

        half = fourier_dims // 2
        freqs = torch.exp(
            torch.arange(half, device=lengths.device, dtype=torch.float)
            * (math.log(10000.0) / max(half - 1, 1))
        )
        args = z_log.unsqueeze(-1) * freqs.unsqueeze(0) * math.pi  # (B, half)
        fourier = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, 2*half)
        return torch.cat([z_log.unsqueeze(-1), linear.unsqueeze(-1), fourier], dim=-1)

    def _entropy_features(self, entropy: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        """[mean entropy, high-entropy-token fraction]. (B, 2)

        High-entropy fraction is aligned with the ~20% "forking token" finding
        (2506.01939): fraction of response tokens whose entropy exceeds a
        per-sequence quantile threshold.
        """
        mask = response_mask.float()
        seq_len = mask.sum(dim=-1).clamp(min=1)
        mean_ent = (entropy * mask).sum(dim=-1) / seq_len

        ent_masked = entropy.masked_fill(mask == 0, float("nan"))
        thresh = torch.nanquantile(ent_masked, self.config.high_entropy_quantile, dim=-1, keepdim=True)
        # nanquantile -> nan if a row is fully masked; treat those as no high-ent tokens.
        thresh = torch.nan_to_num(thresh, nan=float("inf"))
        high_frac = ((entropy >= thresh) * mask).sum(dim=-1) / seq_len
        return torch.stack([mean_ent, high_frac], dim=-1)

    @staticmethod
    def _pass_rate(advantages: torch.Tensor, response_mask: torch.Tensor, gidx: torch.Tensor) -> torch.Tensor:
        """Group difficulty: fraction of group samples with positive sequence reward proxy. (B, 1)

        Uses the sign of each sample's (mean) advantage as a correctness proxy:
        in GRPO the advantage is the group-centered reward, so adv > 0 marks the
        above-average (typically correct) samples. Pass-rate per group = mean over
        the group of 1[adv > 0]; broadcast back to each member.
        """
        from verl.utils import group_mean_std

        mask = response_mask.float()
        seq_adv = (advantages * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)  # (B,)
        correct = (seq_adv > 0).float()
        mean_g, _, _ = group_mean_std(correct, gidx, device=correct.device)
        return mean_g[gidx].unsqueeze(-1)

    def _repetition(self, response_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        """n-gram redundancy: 1 - (#distinct n-grams / #n-grams) per sequence. (B, 1)

        0 = all n-grams distinct (no repetition); ->1 = highly repetitive. Operates
        on raw token ids, so it is tokenizer-agnostic. O(B*L) with a python loop over
        the batch; fine for typical micro-batch sizes, vectorize if it becomes hot.
        """
        n = self.config.repetition_ngram
        B, L = response_ids.shape
        out = response_ids.new_zeros(B, dtype=torch.float)
        ids = response_ids.long()
        lengths = response_mask.sum(dim=-1).long()
        for b in range(B):
            Lb = int(lengths[b].item())
            if Lb < n + 1:
                continue  # too short to repeat; leave at 0
            seq = ids[b, :Lb]
            grams = seq.unfold(0, n, 1)  # (Lb-n+1, n)
            total = grams.shape[0]
            distinct = torch.unique(grams, dim=0).shape[0]
            out[b] = 1.0 - distinct / max(total, 1)
        return out.unsqueeze(-1)

    @staticmethod
    def _truncation(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Binary overlong flag: 1 if the response fills the whole window. (B, 1)

        A sharp signal at the top of the length distribution that the smooth
        log-sine encoding represents poorly. Grounded in DAPO's overlong
        filtering (truncated samples are a distinct failure regime).
        """
        return (lengths >= max_len).float().unsqueeze(-1)

    @staticmethod
    def _semantic_drift(hidden_states: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        """1 - cos(prompt_hidden, mean response hidden). (B, 1)

        Prompt rep = hidden at position 0 (last non-response / first passed token),
        matching the convention used in MLPTokenWeightModule.compute_semantic_drift.
        SPECULATIVE: likely low-signal on math; gameable.
        """
        mask = response_mask.float().unsqueeze(-1)  # (B, L, 1)
        prompt = hidden_states[:, 0, :]  # (B, D)
        resp_mean = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, D)
        cos = F.cosine_similarity(resp_mean, prompt, dim=-1)  # (B,)
        return (1.0 - cos).unsqueeze(-1)

    @staticmethod
    def _accumulated_kl(log_prob: torch.Tensor, ref_log_prob: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        """Mean per-sequence |logpi - logpi_ref|. (B, 1)

        Lets the MLP learn a per-sample KL penalty. SPECULATIVE + CAVEAT: interacts
        with the global kl_loss_coef; watch for double-penalizing KL.
        """
        mask = response_mask.float()
        kl = (log_prob - ref_log_prob).abs() * mask
        return (kl.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)).unsqueeze(-1)

    def forward(
        self,
        response_mask: torch.Tensor,
        group_indices: torch.Tensor | None = None,
        entropy: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        response_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        log_prob: torch.Tensor | None = None,
        ref_log_prob: torch.Tensor | None = None,
        max_len: int = 4096,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute phi from per-sequence features.

        Args (all tensors detached internally before feeding the MLP):
            response_mask: (B, L) binary mask. Always available at the call site.
            group_indices: (B,) GRPO group ids (uid); required for pass_rate.
            entropy:       (B, L) token entropy; required when use_entropy.
            advantages:    (B, L); required for pass_rate.
            response_ids:  (B, L) response token ids; required when use_repetition.
            hidden_states: (B, L, D); required when use_semantic_drift.
            log_prob:      (B, L); required when use_accumulated_kl.
            ref_log_prob:  (B, L); required when use_accumulated_kl.
            max_len:       normalization constant for the linear length feature.

        Returns:
            phi: (B,) mean-1 normalized, clipped weights.
            metrics: dict of scalar metrics.
        """
        from verl.utils import as_torch_index

        mlp_dtype = next(self.mlp.parameters()).dtype
        lengths = response_mask.sum(dim=-1).float()  # (B,)
        gidx = as_torch_index(group_indices, device=response_mask.device) if group_indices is not None else None

        embedded = []
        for name, _ in self.feature_specs:
            if name == "length":
                feat = self._length_features(lengths, self.config.length_fourier_dims, max_len)
            elif name == "entropy":
                assert entropy is not None, "use_entropy=True requires entropy in the batch"
                feat = self._entropy_features(entropy.detach(), response_mask)
            elif name == "pass_rate":
                assert advantages is not None and gidx is not None, "use_pass_rate requires advantages + uid"
                feat = self._pass_rate(advantages.detach(), response_mask, gidx)
            elif name == "repetition":
                assert response_ids is not None, "use_repetition=True requires response_ids in the batch"
                feat = self._repetition(response_ids, response_mask)
            elif name == "truncation":
                feat = self._truncation(lengths, max_len)
            elif name == "semantic_drift":
                assert hidden_states is not None, "use_semantic_drift=True requires hidden_states"
                feat = self._semantic_drift(hidden_states.detach(), response_mask)
            elif name == "accumulated_kl":
                assert log_prob is not None and ref_log_prob is not None, "use_accumulated_kl requires log_prob + ref_log_prob"
                feat = self._accumulated_kl(log_prob.detach(), ref_log_prob.detach(), response_mask)
            else:  # pragma: no cover - guarded in __init__
                raise ValueError(name)
            embedded.append(self.projections[name](feat.detach().to(mlp_dtype)))

        combined = torch.cat(embedded, dim=-1)  # (B, in_dim)
        phi_raw = F.softplus(self.mlp(combined).squeeze(-1))  # (B,)

        phi = phi_raw / phi_raw.mean().clamp(min=1e-8)
        phi = phi.clamp(self.config.phi_clip_min, self.config.phi_clip_max)

        metrics = {
            "lotis/seq_phi_mean": phi.mean().item(),
            "lotis/seq_phi_std": phi.std().item(),
            "lotis/seq_phi_max": phi.max().item(),
            "lotis/seq_phi_min": phi.min().item(),
        }
        return phi, metrics

    def collapse_reg_loss(self, phi: torch.Tensor) -> torch.Tensor:
        """Anti-collapse aux loss (VRPO-34): keep phi's std near target_phi_std.

        Penalizes BOTH collapse-to-uniform (std -> 0) and collapse-to-few-samples
        (std blows up). Returns a scalar; multiply by collapse_reg_coef in the loss.
        Returns 0 when coef == 0.
        """
        if self.config.collapse_reg_coef <= 0:
            return phi.new_zeros(())
        return (phi.std() - self.config.target_phi_std) ** 2
