"""LOGO hybrid advantage estimator: GRPO group baseline + stored value control variate."""

from typing import Optional

import torch

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import register_adv_est
from verl.utils import as_torch_index, group_mean_std


@register_adv_est("logo")
def compute_logo_hybrid_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index=None,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    v_stored: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Compute LOGO hybrid advantage.

    A_i = (R_i - mu_group) + lambda * (R_i - V_stored)
    A_final = A_i / (sigma_batch + epsilon)

    where lambda = exp(-sigma_group^2 / tau)  (adaptive)

    Args:
        token_level_rewards: (bs, response_length) token-level reward signal.
        response_mask: (bs, response_length) binary mask.
        index: group labels (prompt uid), one per sample.
        config: AlgoConfig with nested ``logo`` config.
        v_stored: (bs,) stored per-prompt value from meta-store.
            Must be pre-populated in data.batch before calling compute_advantage.

    Returns:
        (advantages, returns, metrics) where metrics is a dict of LOGO-specific
        scalar metrics for logging.
    """
    metrics = {}

    with torch.no_grad():
        # sequence-level reward
        scores = token_level_rewards.sum(dim=-1)  # (bs,)

        # --- group statistics ---
        if index is not None:
            g = as_torch_index(index, device=scores.device)
            mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon, device=scores.device)
            mu_group = mean_g[g]  # (bs,)
            sigma_group = std_g[g]  # (bs,)
        else:
            mu_group = scores.mean().expand_as(scores)
            sigma_group = scores.std().expand_as(scores)

        # --- GRPO component ---
        grpo_adv = scores - mu_group

        # --- stored value component ---
        if v_stored is not None:
            v = v_stored.to(scores.device)
            ppo_adv = scores - v
        else:
            # fallback: pure GRPO
            ppo_adv = torch.zeros_like(scores)

        # --- adaptive lambda ---
        logo_cfg = config.get("logo", None) if config is not None else None
        if logo_cfg is not None:
            adv_cfg = logo_cfg.get("advantage", None) if hasattr(logo_cfg, "get") else getattr(logo_cfg, "advantage", None)
        else:
            adv_cfg = None

        tau = 0.1
        use_adaptive = True
        lambda_fixed = 0.5
        if adv_cfg is not None:
            tau = getattr(adv_cfg, "tau", 0.1)
            use_adaptive = getattr(adv_cfg, "adaptive_lambda", True)
            lambda_fixed = getattr(adv_cfg, "lambda_fixed", 0.5)

        if use_adaptive:
            lam = torch.exp(-sigma_group.pow(2) / tau)
        else:
            lam = torch.full_like(scores, lambda_fixed)

        # --- hybrid advantage ---
        hybrid = grpo_adv + lam * ppo_adv

        # --- batch normalisation ---
        batch_std = hybrid.std().clamp(min=epsilon)
        normalised = hybrid / batch_std

        # broadcast to token level
        advantages = normalised.unsqueeze(-1) * response_mask

        # --- metrics ---
        metrics["logo/lambda_mean"] = lam.mean().item()
        metrics["logo/lambda_std"] = lam.std().item()
        metrics["logo/sigma_group_mean"] = sigma_group.mean().item()
        metrics["logo/grpo_adv_abs_mean"] = grpo_adv.abs().mean().item()
        metrics["logo/ppo_adv_abs_mean"] = ppo_adv.abs().mean().item()
        grpo_mag = grpo_adv.abs().mean().item()
        ppo_mag = ppo_adv.abs().mean().item()
        metrics["logo/global_local_ratio"] = ppo_mag / max(grpo_mag, epsilon)

        # sign conflict: GRPO says positive, stored value says negative
        if v_stored is not None:
            conflict = ((grpo_adv > 0) & (ppo_adv < 0)) | ((grpo_adv < 0) & (ppo_adv > 0))
            metrics["logo/sign_conflict_rate"] = conflict.float().mean().item()
        else:
            metrics["logo/sign_conflict_rate"] = 0.0

    return advantages, advantages, metrics
