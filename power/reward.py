"""
Custom math reward with format tracking for power experiments.

Usage in config:
    custom_reward_function:
        path: verl/power/reward.py
        name: compute_score

Strategy:
1. Try math_verify (symbolic equivalence) - handles fractions, expressions, decimals
2. Fallback to math_dapo normalization - handles units, text, x=-1

Requirements:
    pip install math-verify
"""

from verl.utils.reward_score import math_dapo
import random
import os

# Debug log file (print doesn't work in Ray workers)
DEBUG_LOG_FILE = os.environ.get("REWARD_DEBUG_LOG", "/tmp/reward_debug.log")

def debug_log(msg):
    try:
        with open(DEBUG_LOG_FILE, "a") as f:
            f.write(msg + "\n")
    except:
        pass

# Use math-verify library DIRECTLY with parse() and verify() API
MATH_VERIFY_AVAILABLE = False
# try:
#     from math_verify import parse, verify
#     MATH_VERIFY_AVAILABLE = True
#     debug_log("[INIT] math-verify parse/verify API loaded")
# except ImportError as e:
#     debug_log(f"[INIT] math-verify NOT available: {e}")
# except Exception as e:
#     debug_log(f"[INIT] math-verify init error: {type(e).__name__}: {e}")

MATH_METRICS_AVAILABLE = False
try:
    from power.metric import math_metric, timeout
    debug_log("[INIT] self math_metric loaded")
    from math_verify.errors import TimeoutException
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    debug_log("[INIT] other math_verify loaded")
    MATH_METRICS_AVAILABLE = True
except ImportError as e:
    debug_log(f"[INIT] self math_metric NOT available: {e}")
except Exception as e:
    debug_log(f"[INIT] self math_metric init error: {type(e).__name__}: {e}")

def compute_score(data_source, solution_str, ground_truth, timeout_seconds=5, extra_info=None, **kwargs):
    """Compute math reward using existing VeRL modules.
    
    Strategy:
    1. math_verify: symbolic equivalence (handles 0.5 == 1/2, expressions)
    2. math_dapo: string matching fallback (handles units, text, x=-1)
    
    Uses math_dapo functions for boxed extraction (consistent with dapo normalization).
    Returns 0/1 rewards (not -1/1 like original math_dapo).
    """
    format_ok = False
    pred = None
    score = 0.0
    
    try:
        # Extract \boxed{} using math_dapo functions
        string_in_last_boxed = math_dapo.last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            format_ok = True
            pred = math_dapo.remove_boxed(string_in_last_boxed)

        if MATH_METRICS_AVAILABLE and pred is not None:
            verify_fn = math_metric(
                gold_extraction_target=(LatexExtractionConfig(),),
                pred_extraction_target=(
                    ExprExtractionConfig(),
                    LatexExtractionConfig(),
                ),
            )
            verify_fn = timeout(timeout_seconds)(verify_fn)
            try:
                gt_boxed = "\\boxed{" + ground_truth + "}"
                score, _ = verify_fn([gt_boxed], [string_in_last_boxed])
                debug_log(f"[score]: ]{score} | gt_boxed = '{gt_boxed} | string_in_last_boxed = '{string_in_last_boxed}")
            except Exception as e:
                # if random.random() < 0.01:
                debug_log(f"[math_metric ERROR] {type(e).__name__}: {e}")
                score = 0.0
            except TimeoutException:
                debug_log("[TimeoutException]")
                score = 0.0

        # Method 1: Try math-verify (symbolic equivalence) - using parse/verify API
        # Normalize GT first to handle "x = -1" -> "-1", "100 dollars" -> "100"
        if MATH_VERIFY_AVAILABLE and pred is not None:
            try:
                gt_clean = math_dapo.normalize_final_answer(ground_truth)
                # Parse raw values directly with timeout disabled for Ray threads
                # Wrap in \boxed{} to ensure LatexExtractionConfig picks it up!
                gold_parsed = parse(f"\\boxed{{{gt_clean}}}", parsing_timeout=None)
                pred_parsed = parse(f"\\boxed{{{pred}}}", parsing_timeout=None)
                
                # verify(gold, answer) - order matters!
                if verify(gold_parsed, pred_parsed):
                    score = 1.0
                else:
                     # Debug failed verification for exact matches
                     if gt_clean == pred and random.random() < 0.1:
                         debug_log(f"[VERIFY FAIL] exact match failed! gt='{gt_clean}' | gold_parsed='{gold_parsed}' | pred_parsed='{pred_parsed}'")
            except Exception as e:
                # Log actual errors (e.g. parsing failures)
                if random.random() < 0.01:
                    debug_log(f"[math_verify ERROR] {type(e).__name__}: {e}")
                score = 0.0
        
        # Method 2: Fallback to math_dapo normalization
        # Normalize both pred and GT, then compare. Essential if math_verify fails or is unavailable.
        # if score < 0.5 and pred is not None:
        #     pred_norm = math_dapo.normalize_final_answer(pred)
        #     gt_norm = math_dapo.normalize_final_answer(ground_truth)
        #     if pred_norm == gt_norm:
        #         score = 1.0
        
        # Debug logging (10% sample) - check EXACT match cases and who provided the score
        if random.random() < 0.10 and MATH_VERIFY_AVAILABLE:
            gt_norm = math_dapo.normalize_final_answer(ground_truth)
            pred_norm = math_dapo.normalize_final_answer(pred) if pred else None
            is_exact = (pred_norm == gt_norm) if pred_norm else False
            debug_log(f"[reward] score={score} | gt_norm='{gt_norm}' | pred='{pred}' | exact={is_exact} | verify_avail={MATH_VERIFY_AVAILABLE}")
                
    except Exception as e:
        if random.random() < 0.001:
            debug_log(f"[Reward Error] {type(e).__name__}: {e}")

    return {
        "score": float(score),
        "acc": 1.0 if score >= 0.5 else 0.0,
        "format_ok": 1.0 if format_ok else 0.0,
    }
