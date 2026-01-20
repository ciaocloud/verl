"""
Custom math reward with format tracking for power experiments.

Usage in config:
    custom_reward_function:
        path: verl/power/reward.py
        name: compute_score

Features:
- Uses math_verify for robust symbolic equivalence checking
  (handles: \\frac{1}{2} vs 0.5, symbolic expressions, etc.)
- Falls back to string matching if math_verify not installed
- Tracks format_ok: whether \\boxed{} was found
- Returns extracted prediction for debugging

Note on precision:
    math_verify uses SymPy for symbolic comparison, which handles:
    - Exact fractions: 1/2 == 0.5 == \\frac{1}{2}
    - Symbolic equivalence: (x+1)^2 == x^2+2x+1
    - For pure floats, it uses numerical tolerance (~1e-6)

Requirements:
    pip install math-verify
"""

from verl.utils.reward_score import math_reward
import random
import wandb

# Check if math_verify is available
try:
    from verl.utils.reward_score import math_verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("[power/reward.py] math-verify not installed. Falling back to string matching.")
    print("  For better accuracy, run: pip install math-verify")


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Compute math reward with detailed metrics.
    
    Uses math_verify for robust symbolic equivalence when available,
    falls back to string matching otherwise.
    
    Signature matches VeRL's custom reward function interface.
    
    Returns:
        dict with:
            - score: float (0.0 or 1.0) - the reward used for training
            - acc: bool - same as score, just for logging convenience
            - format_ok: bool - whether \\boxed{} was found (for diagnostics)
            - pred: str or None - the extracted prediction (for debugging)
    """
    format_ok = False
    pred = None
    score = 0.0
    
    try:
        # Extract \boxed{} for format tracking
        string_in_last_boxed = math_reward.last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            format_ok = True
            pred = math_reward.remove_boxed(string_in_last_boxed)
        
        # Check equivalence
        if MATH_VERIFY_AVAILABLE:
            # Use math_verify for robust symbolic equivalence
            score = math_verify.compute_score(solution_str, ground_truth)
        else:
            # Fallback to string matching (less robust but works without deps)
            if pred is not None and math_reward.is_equiv(pred, ground_truth):
                score = 1.0
    except Exception:
        pass

    score = float(score)
    acc = 1.0 if score >= 0.5 else 0.0        # float for accuracy
    format_ok = 1.0 if format_ok else 0.0     # float for format rate
    
    # HACK: Log 0.1% of training samples directly to WandB
    if random.random() < 0.001: 
        try:
            print(f"\n[TRAIN SAMPLE] GT: {ground_truth} | Format: {format_ok}\nOutput: {solution_str}\n")
            if wandb.run is not None:
                wandb.log({
                    "train_sample_text": wandb.Html(f"<p><b>GT:</b> {ground_truth}</p><p><b>Gen:</b> {solution_str}</p>")
                })
        except Exception:
            pass

    # Return dict with metrics
    # NOTE: If you get JSON serialization errors, remove 'trainer.rollout_data_dir' from config
    return {
        "score": score,
        "acc": acc,
        "format_ok": format_ok,
    }
