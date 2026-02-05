"""LOTIS modules for learnable sequence and token weighting."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.lotis.config import LengthWeightConfig, TokenWeightConfig


class RBFLengthWeightModule(nn.Module):
    """Computes sequence-level weights using RBF kernels on z-score normalized lengths.
    
    phi(z) = exp(sum_k alpha_k * K(z, mu_k)) where K is Gaussian kernel.
    Output is normalized to preserve gradient energy.
    """

    def __init__(self, config: LengthWeightConfig):
        super().__init__()
        self.config = config
        
        # Learnable alpha parameters
        self.alphas = nn.Parameter(
            torch.full((config.num_rbf_kernels,), config.alpha_init)
        )
        
        # Fixed RBF centers
        if config.rbf_centers is None:
            centers = torch.linspace(-2.0, 2.0, config.num_rbf_kernels)
        else:
            centers = torch.tensor(config.rbf_centers, dtype=torch.float32)
        self.register_buffer("centers", centers)
        self.register_buffer("bandwidth_sq_2", torch.tensor(2.0 * config.rbf_bandwidth ** 2))

    def forward(self, response_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute phi weights from response mask.
        
        Args:
            response_mask: (batch_size, seq_len) binary mask
            
        Returns:
            phi: (batch_size,) weights normalized to mean=1
            metrics: dict of metrics
        """
        # Get sequence lengths
        lengths = response_mask.sum(dim=-1).float()  # (batch_size,)
        
        # Z-score normalization within batch
        mu = lengths.mean()
        sigma = lengths.std().clamp(min=1e-8)
        z = (lengths - mu) / sigma  # (batch_size,)
        
        # RBF kernels: K(z, mu_k) = exp(-(z - mu_k)^2 / (2 * sigma^2))
        z_exp = z.unsqueeze(-1)  # (batch_size, 1)
        kernels = torch.exp(-(z_exp - self.centers) ** 2 / self.bandwidth_sq_2)  # (batch_size, K)
        
        # phi = exp(sum_k alpha_k * K(z, mu_k))
        weighted_sum = (self.alphas * kernels).sum(dim=-1)  # (batch_size,)
        phi_raw = torch.exp(weighted_sum)
        
        # Normalize to mean=1 (preserves gradient energy)
        phi = phi_raw / phi_raw.mean().clamp(min=1e-8)
        
        # Clip for stability
        phi = phi.clamp(self.config.phi_clip_min, self.config.phi_clip_max)
        
        metrics = {
            "lotis/response_length_mean": mu.item(),
            "lotis/response_length_std": sigma.item(),
            # Center kernel weight (assuming centers are sorted and middle one is ~0)
            "lotis/rbf_kernel_weight_center": self.alphas[self.config.num_rbf_kernels // 2].item(),
            "lotis/rbf_alphas_mean": self.alphas.mean().item(),
            "lotis/rbf_phi_mean": phi.mean().item(),
            "lotis/rbf_phi_std": phi.std().item(),
            "lotis/rbf_phi_max": phi.max().item(),
            "lotis/rbf_phi_min": phi.min().item(),
        }
        
        # Log individual alphas (optional, maybe limit if K is large)
        for i, alpha in enumerate(self.alphas):
            metrics[f"lotis/rbf_alpha_{i}"] = alpha.item()
        
        return phi, metrics


class KLTokenWeightModule(nn.Module):
    """Token weighting based on KL divergence between policy and reference model.
    
    w_t = |log pi_theta - log pi_ref|^gamma
    Output is normalized per sequence to preserve gradient energy.
    """

    def __init__(self, config: TokenWeightConfig):
        super().__init__()
        self.config = config
        
        # Learnable gamma via softplus to ensure > 0
        self._gamma_raw = nn.Parameter(torch.tensor(0.0))
        self._init_gamma(config.gamma_init)

    def _init_gamma(self, target: float):
        """Initialize gamma_raw so gamma_max * tanh(softplus(gamma_raw)) = target."""
        max_val = getattr(self.config, "gamma_max", 3.0)
        # Clamp target to valid range (0, max_val)
        target = max(1e-6, min(max_val - 1e-6, target))
        # y = arctanh(target / max_val)
        ratio = target / max_val
        # arctanh(x) = 0.5 * log((1+x)/(1-x))
        y = 0.5 * math.log((1 + ratio) / (1 - ratio))
        # x = inv_softplus(y)
        if y > 20:
            self._gamma_raw.data.fill_(y)
        else:
            val = math.exp(y) - 1
            val = max(1e-6, val)
            self._gamma_raw.data.fill_(math.log(val))

    @property
    def gamma(self) -> torch.Tensor:
        # Bounded gamma: gamma_max * tanh(softplus(gamma_raw))
        # Prevents sparsity collapse, ensuring credit assignment to "scaffolding" tokens
        max_val = getattr(self.config, "gamma_max", 3.0)
        return max_val * torch.tanh(F.softplus(self._gamma_raw))

    def forward(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute TIS weights from log probability divergence.
        
        Args:
            log_probs: (batch_size, seq_len) current policy log probs (detached)
            ref_log_probs: (batch_size, seq_len) reference policy log probs
            response_mask: (batch_size, seq_len) binary mask
            
        Returns:
            W: (batch_size, seq_len) weights normalized per sequence
            metrics: dict of metrics
        """
        # Compute divergence (detach to prevent gradient through diff)
        divergence = torch.abs(log_probs.detach() - ref_log_probs.detach())
        
        # Clip divergence to prevent exploding weights/gradients from outliers
        # This ensures we still learn from large divergences (gradient flows through beta)
        # without numerical instability.
        if hasattr(self.config, "div_clip") and self.config.div_clip > 0:
            divergence = divergence.clamp(min=1e-6, max=self.config.div_clip)
        else:
            divergence = divergence.clamp(min=1e-6)
        
        # Apply learnable power
        w_raw = divergence.pow(self.gamma)  # (batch_size, seq_len)
        
        # Mask and normalize per sequence
        w_masked = w_raw * response_mask
        seq_sums = w_masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        seq_lens = response_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        
        # Normalize so each sequence sums to its length (preserves gradient energy)
        W = w_masked / seq_sums * seq_lens
        
        # Clip for stability
        W = W.clamp(self.config.psi_clip_min, self.config.psi_clip_max) * response_mask
        
        # KL divergence metrics
        kl_term = torch.exp(ref_log_probs) * (ref_log_probs - log_probs) * response_mask
        ref_kl = kl_term.sum() / response_mask.sum()
        
        # Weighted KL (weighted by TIS weights)
        # W is normalized per sequence, so we can just multiply
        ref_kl_weighted = (kl_term * W).sum() / response_mask.sum()

        w_valid = W[response_mask.bool()]
        metrics = {
            "lotis/tis_gamma": self.gamma.item(),
            "lotis/tis_weight_mean": w_valid.mean().item(),
            "lotis/tis_weight_std": w_valid.std().item(),
            "lotis/tis_weight_max": w_valid.max().item(),
            "lotis/tis_weight_min": w_valid.min().item(),
            "lotis/tis_sparsity_ratio": W.max().item() / (w_valid.mean().item() + 1e-6),
            "lotis/ref_kl_total": ref_kl.item(),
            "lotis/ref_kl_weighted": ref_kl_weighted.item(),
        }
        
        return W, metrics

class SwiGLU(nn.Module):
    """SwiGLU Activation Layer: SwiGLU(x) = SiLU(xW_g) * (xW_v)"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gate = nn.Linear(in_features, out_features, bias=bias)
        self.val = nn.Linear(in_features, out_features, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.gate(x)) * self.val(x)


class MLPTokenWeightModule(nn.Module):
    """Computes token-level importance weights based on MLP over multi-modal features.
    
    Features:
    - hidden_state (h_t): Vector, "What am I saying?"
    - log_prob: Scalar, Confidence - "Am I sure?"
    - entropy: Scalar, Uncertainty - "Did I struggle?"
    - kl_divergence: Scalar, Novelty - "Is this new?"
    - relative_position: Scalar, Timing - "Intro or conclusion?"
    - semantic_drift: Scalar, Focus - "Are we still on topic?"
    - local_ppl: Scalar, Smoothed PPL - "Hard reasoning block?"
    
    Output is normalized per sequence to preserve gradient energy.
    Symbol: psi (ψ)
    """
    
    # Architecture defaults (hardcoded)
    MLP_HIDDEN_DIM = 512
    MLP_NUM_LAYERS = 1
    SCALAR_EMBED_DIM = 16
    HIDDEN_EMBED_DIM = 256
    LOCAL_PPL_WINDOW = 8  # Window size for local PPL
    
    def __init__(self, config: TokenWeightConfig, hidden_dim: int):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim  # Input hidden state dimension
        
        # 3. Build MLP
        # We project scalar features and hidden state, then concat
        # Then pass through SwiGLU-based MLP
        
        # Define scalar feature list
        self.scalar_features = []
        if config.use_log_prob:
            self.scalar_features.append("log_prob")
        if config.use_entropy:
            self.scalar_features.append("entropy")
        if config.use_kl_divergence:
            self.scalar_features.append("kl_divergence")
        if config.use_relative_position:
            self.scalar_features.append("position")
        if config.use_semantic_drift:
            self.scalar_features.append("semantic_drift")
        if config.use_local_ppl:
            self.scalar_features.append("local_ppl")
        
        # Input projections for scalar features
        self.scalar_projections = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(1, self.SCALAR_EMBED_DIM),
                nn.SiLU() # Non-linearity for scalar embedding
            ) for k in self.scalar_features
        })
        
        # Compute total input dim
        input_dim = 0
        if config.use_hidden_state:
            # Project hidden state to HIDDEN_EMBED_DIM
            # This balances the influence of the massive hidden state vs scalar features
            self.hidden_projection = nn.Sequential(
                nn.Linear(hidden_dim, self.HIDDEN_EMBED_DIM),
                nn.SiLU(),
            )
            input_dim += self.HIDDEN_EMBED_DIM
        input_dim += len(self.scalar_features) * self.SCALAR_EMBED_DIM
        
        # Build MLP
        layers = []
        in_dim = input_dim
        
        # Pre-norm on concatenated feature vector
        self.combined_norm = nn.LayerNorm(input_dim)
        
        # Store input_dim for feature importance tracking
        self.input_dim = input_dim

        for _ in range(self.MLP_NUM_LAYERS):
            layers.append(SwiGLU(in_dim, self.MLP_HIDDEN_DIM))
            in_dim = self.MLP_HIDDEN_DIM
            
        # Final projection to scalar
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize MLP weights for stable starting point."""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                # Standard Xavier for linear layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # We want final layer to start small to ensure near-uniform weights initially
        last_linear = self.mlp[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.normal_(last_linear.weight, mean=0.0, std=0.01)
            nn.init.zeros_(last_linear.bias)
    
    @staticmethod
    def compute_relative_positions(response_mask: torch.Tensor) -> torch.Tensor:
        """Compute relative position in [0, 1] for each token.
        
        Args:
            response_mask: (B, L) binary mask
            
        Returns:
            rel_pos: (B, L) where rel_pos[b, t] = t / seq_len[b]
        """
        B, L = response_mask.shape
        device = response_mask.device
        
        # Create position indices: 0, 1, 2, ...
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1).float()
        
        # Get sequence lengths
        seq_lens = response_mask.sum(dim=-1, keepdim=True).clamp(min=1).float()
        
        # Normalize to [0, 1]
        rel_pos = positions / seq_lens
        
        return rel_pos * response_mask
    
    @staticmethod
    def compute_local_ppl(log_prob: torch.Tensor, response_mask: torch.Tensor, window: int = 8) -> torch.Tensor:
        """Compute local perplexity as rolling mean of -log_prob.
        
        Uses causal windowing, properly handling the first tokens by dividing
        by actual number of available tokens rather than full window size.
        
        Args:
            log_prob: (B, L) log probabilities (already detached)
            response_mask: (B, L) binary mask
            window: window size for rolling average
            
        Returns:
            local_ppl: (B, L) smoothed negative log prob
        """
        # Negative log prob (higher = harder token)
        neg_log_prob = (-log_prob * response_mask).detach()
        
        B, L = neg_log_prob.shape
        device = neg_log_prob.device
        
        # Causal padding (pad on left)
        padded = F.pad(neg_log_prob, (window - 1, 0), value=0.0)
        
        # Rolling sum using 1D convolution with uniform kernel
        kernel = torch.ones(1, 1, window, device=device, dtype=neg_log_prob.dtype)
        rolling_sum = F.conv1d(
            padded.unsqueeze(1),  # (B, 1, L + window - 1)
            kernel,
        ).squeeze(1)  # (B, L)
        
        # Compute actual number of valid tokens in each window position
        # For position t, we have min(t+1, window) valid tokens
        # But we also need to account for response_mask
        padded_mask = F.pad(response_mask.float(), (window - 1, 0), value=0.0)
        valid_counts = F.conv1d(
            padded_mask.unsqueeze(1),
            kernel,
        ).squeeze(1).clamp(min=1.0)  # (B, L), clamp to avoid div by zero
        
        # Proper rolling mean
        local_ppl = rolling_sum / valid_counts
        
        return local_ppl * response_mask
    
    @staticmethod
    def compute_semantic_drift(
        hidden_states: torch.Tensor, 
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute semantic drift as 1 - cosine_similarity(h_t, h_prompt_mean).
        
        The prompt hidden state is computed as the mean of all non-response tokens
        (i.e., where response_mask is 0 but the token exists).
        
        For efficiency, we use the hidden state at position 0 (after response starts)
        as the prompt representation since it contains all prior context.
        
        Actually, we use the last NON-response hidden state as the prompt embedding.
        
        Args:
            hidden_states: (B, L, D) hidden states (already detached)
            response_mask: (B, L) binary mask marking response tokens
            
        Returns:
            semantic_drift: (B, L) where higher = more drift from prompt
        """
        B, L, D = hidden_states.shape
        device = hidden_states.device
        
        # The response_mask marks response tokens. The prompt is everything before.
        # Since hidden_states is already sliced to response length, we use position 0
        # (which corresponds to the first response token, whose hidden state 
        # incorporates all prompt context).
        # 
        # Alternative: we could pass prompt_hidden separately, but this is simpler.
        # Using h[0] as prompt representation (transformer has seen full prompt by then)
        prompt_hidden = hidden_states[:, 0:1, :]  # (B, 1, D)
        
        # Compute cosine similarity between each token and prompt
        # Normalize hidden states
        h_norm = F.normalize(hidden_states, p=2, dim=-1)  # (B, L, D)
        prompt_norm = F.normalize(prompt_hidden, p=2, dim=-1)  # (B, 1, D)
        
        # Cosine similarity: dot product of normalized vectors
        cos_sim = (h_norm * prompt_norm).sum(dim=-1)  # (B, L)
        
        # Drift = 1 - similarity (higher drift = less similar to prompt)
        semantic_drift = (1.0 - cos_sim) * response_mask
        
        return semantic_drift
        
    def forward(
        self,
        features: dict[str, torch.Tensor],
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute TIS weights from multi-modal features.
        
        Args:
            features: dict containing:
                - hidden_states: (B, L, D) last layer hidden states
                - log_prob: (B, L) current policy log probs
                - entropy: (B, L) token entropy (optional)
                - ref_log_prob: (B, L) reference policy log probs (optional)
            response_mask: (B, L) binary mask
            
        Returns:
            psi: (B, L) weights normalized per sequence
            metrics: dict of metrics
        """
        mlp_dtype = next(self.mlp.parameters()).dtype
        B, L = response_mask.shape
        
        # Track feature dimensions for importance computation
        feature_dims = {}  # name -> (start_idx, end_idx)
        current_idx = 0
        
        # Build feature list
        feature_tensors = []
        
        # 1. Hidden state (vector feature)
        if self.config.use_hidden_state:
            hidden_states = features["hidden_states"].detach().to(mlp_dtype)
            # Project to lower dimension
            hidden_embedded = self.hidden_projection(hidden_states)
            feature_tensors.append(hidden_embedded)
            feature_dims["hidden"] = (current_idx, current_idx + self.HIDDEN_EMBED_DIM)
            current_idx += self.HIDDEN_EMBED_DIM
        
        # 2. Scalar features - each projected to scalar_embed_dim
        # Log prob (confidence)
        if self.config.use_log_prob and "log_prob" in features:
            log_prob = features["log_prob"].detach().to(mlp_dtype)
            log_prob_embed = self.scalar_projections["log_prob"](
                log_prob.unsqueeze(-1)
            )  # (B, L, embed_dim)
            feature_tensors.append(log_prob_embed)
            feature_dims["log_prob"] = (current_idx, current_idx + self.SCALAR_EMBED_DIM)
            current_idx += self.SCALAR_EMBED_DIM
        
        # Entropy (uncertainty)
        if self.config.use_entropy and "entropy" in features and features["entropy"] is not None:
            entropy = features["entropy"].detach().to(mlp_dtype)
            entropy_embed = self.scalar_projections["entropy"](
                entropy.unsqueeze(-1)
            )
            feature_tensors.append(entropy_embed)
            feature_dims["entropy"] = (current_idx, current_idx + self.SCALAR_EMBED_DIM)
            current_idx += self.SCALAR_EMBED_DIM
        
        # KL divergence (novelty)
        if self.config.use_kl_divergence and "ref_log_prob" in features and features["ref_log_prob"] is not None:
            log_prob = features["log_prob"].detach().to(mlp_dtype)
            ref_log_prob = features["ref_log_prob"].detach().to(mlp_dtype)
            kl_div = (log_prob - ref_log_prob).abs()  # Use absolute KL for stability
            kl_embed = self.scalar_projections["kl_divergence"](
                kl_div.unsqueeze(-1)
            )
            feature_tensors.append(kl_embed)
            feature_dims["kl"] = (current_idx, current_idx + self.SCALAR_EMBED_DIM)
            current_idx += self.SCALAR_EMBED_DIM
        
        # Relative position (timing)
        if self.config.use_relative_position:
            rel_pos = self.compute_relative_positions(response_mask).to(mlp_dtype)
            rel_pos_embed = self.scalar_projections["position"](
                rel_pos.unsqueeze(-1)
            )
            feature_tensors.append(rel_pos_embed)
            feature_dims["position"] = (current_idx, current_idx + self.SCALAR_EMBED_DIM)
            current_idx += self.SCALAR_EMBED_DIM
        
        # Semantic drift (focus)
        if self.config.use_semantic_drift and "hidden_states" in features:
            hidden_states = features["hidden_states"].detach().to(mlp_dtype)
            sem_drift = self.compute_semantic_drift(hidden_states, response_mask)
            sem_drift_embed = self.scalar_projections["semantic_drift"](
                sem_drift.unsqueeze(-1)
            )
            feature_tensors.append(sem_drift_embed)
            feature_dims["drift"] = (current_idx, current_idx + self.SCALAR_EMBED_DIM)
            current_idx += self.SCALAR_EMBED_DIM
        
        # Local PPL (smoothed perplexity)
        if self.config.use_local_ppl and "log_prob" in features:
            log_prob = features["log_prob"].detach().to(mlp_dtype)
            local_ppl = self.compute_local_ppl(log_prob, response_mask, window=self.LOCAL_PPL_WINDOW)
            local_ppl_embed = self.scalar_projections["local_ppl"](
                local_ppl.unsqueeze(-1)
            )
            feature_tensors.append(local_ppl_embed)
            feature_dims["local_ppl"] = (current_idx, current_idx + self.SCALAR_EMBED_DIM)
            current_idx += self.SCALAR_EMBED_DIM
        
        # Concatenate all features
        combined = torch.cat(feature_tensors, dim=-1)  # (B, L, total_dim)
        
        # Apply LayerNorm on concatenated features (before MLP)
        combined = self.combined_norm(combined)
        
        # Register hook to capture gradients for feature importance (during training)
        # NOTE: We use detached copies to avoid retaining the computation graph
        if self.training and combined.requires_grad:
            # Capture only what we need, detached
            combined_detached = combined.detach()
            mask_detached = response_mask.detach()
            dims_copy = dict(feature_dims)  # Copy to avoid closure issues
            
            def save_grad_hook(grad):
                # Compute feature importance immediately when gradient is available
                importance = (grad * combined_detached).abs()
                mask_3d = mask_detached.unsqueeze(-1).expand_as(importance)
                
                for name, (start_idx, end_idx) in dims_copy.items():
                    feat_importance = importance[:, :, start_idx:end_idx]
                    feat_mask = mask_3d[:, :, start_idx:end_idx]
                    valid_importance = feat_importance[feat_mask.bool()]
                    if valid_importance.numel() > 0:
                        key = f"lotis/feat_importance_{name}"
                        if not hasattr(self, '_accumulated_importance'):
                            self._accumulated_importance = {}
                        if key not in self._accumulated_importance:
                            self._accumulated_importance[key] = []
                        self._accumulated_importance[key].append(valid_importance.mean().item())
                return grad  # Return grad unchanged
            
            combined.register_hook(save_grad_hook)
        
        # Forward through MLP to get raw scores
        w_raw = self.mlp(combined).squeeze(-1)  # (B, L)
        
        # Softplus for positivity
        # With init std=0.01, w_raw ~ N(0, 0.01)
        # softplus(w_raw) ~ softplus(0) ~ 0.69
        # Normalized psi will be ~ 1.0 (uniform) at start
        w_raw = F.softplus(w_raw)
        
        # Mask and normalize per sequence
        w_masked = w_raw * response_mask
        seq_sums = w_masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        seq_lens = response_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        
        # Normalize so each sequence sums to its length (preserves gradient energy)
        psi = w_masked / seq_sums * seq_lens
        
        # Clip for stability
        psi = psi.clamp(self.config.psi_clip_min, self.config.psi_clip_max) * response_mask
        
        psi_valid = psi[response_mask.bool()]
        metrics = {
            "lotis/psi_weight_mean": psi_valid.mean().item(),
            "lotis/psi_weight_std": psi_valid.std().item(),
            "lotis/psi_weight_max": psi_valid.max().item(),
            "lotis/psi_weight_min": psi_valid.min().item(),
            "lotis/psi_raw_mean": w_raw[response_mask.bool()].mean().item(),
            "lotis/psi_raw_std": w_raw[response_mask.bool()].std().item(),
            "lotis/psi_num_features": len(feature_tensors),
        }
        
        # Cast back to input dtype for consistency with other tensors in loss
        return psi, metrics
    
    def compute_feature_importance(self) -> dict[str, float]:
        """Return accumulated feature importance metrics from backward hooks.
        
        Call this after loss.backward() to get feature sensitivity metrics.
        The importance is computed via |grad × input| (Gradient × Input attribution).
        
        Returns:
            dict mapping feature name to mean absolute importance (averaged across micro-batches)
        """
        if not hasattr(self, '_accumulated_importance') or not self._accumulated_importance:
            return {}
        
        # Average across micro-batches
        metrics = {}
        for key, values in self._accumulated_importance.items():
            if values:
                metrics[key] = sum(values) / len(values)
        
        # Clear for next mini-batch
        self._accumulated_importance = {}
        
        return metrics


