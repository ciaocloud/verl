import argparse
import json
import math
import os
import sys
import pandas as pd
import numpy as np
import glob
import re
import subprocess
import time
import torch

# ================= Configuration Defaults =================
DEFAULT_DATA = "test.parquet"
DEFAULT_N_SAMPLES = 8
DEFAULT_MAX_RESPONSE_LENGTH = 8192
DEFAULT_MAX_PROMPT_LENGTH = 2048
DEFAULT_TP_SIZE = torch.cuda.device_count() if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1
DEFAULT_GPU_UTIL = 0.95
DEFAULT_DTYPE = "auto"
DEFAULT_WANDB_PROJECT = "verl_eval"
DEFAULT_REWARD_FN = os.path.join(os.path.dirname(__file__), "reward.py")
# ==========================================================

def load_reward_fn(reward_path):
    """Load reward function from a file path."""
    if not os.path.exists(reward_path):
        raise ValueError(f"Reward file not found: {reward_path}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("reward", reward_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score

def pass_at_k(n, c, k):
    """
    Calculate unbiased pass@k.
    n: total samples
    c: correct samples
    k: k in pass@k
    """
    if n - c < k: return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def find_checkpoints(model_path):
    """
    Find all checkpoints in a directory.
    Returns sorted list of (step, path).
    """
    checkpoints = []
    if os.path.isdir(model_path):
        # Look for global_step_* subdirs
        subdirs = glob.glob(os.path.join(model_path, "global_step_*"))
        for s in subdirs:
            # extract step
            match = re.search(r"global_step_(\d+)", s)
            if match:
                step = int(match.group(1))
                # Check for actor subdir (standard verl structure)
                actor_path = os.path.join(s, "actor")
                ckpt_path = actor_path if os.path.isdir(actor_path) else s
                checkpoints.append((step, ckpt_path))
        checkpoints.sort()
    return checkpoints

def infer_step(model_path):
    """Try to infer step number from path."""
    match = re.search(r"global_step_(\d+)", model_path)
    if match:
        return int(match.group(1))
    return 0

def evaluate_model(model_path, output_file, args):
    """Run evaluation for a single model."""
    from vllm import LLM, SamplingParams
    
    # Load reward function
    compute_score = load_reward_fn(args.reward)
    print(f"Using reward function: {args.reward}")
    
    print(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    
    prompts = []
    ground_truths = []
    data_sources = []
    
    for _, row in df.iterrows():
        # prompt is list of dicts (chat format)
        msgs = row['prompt']
        prompts.append(msgs)
        
        gt = None
        if 'reward_model' in row and row['reward_model'] is not None:
             gt = row['reward_model'].get('ground_truth')
        ground_truths.append(gt)
        data_sources.append(row.get('data_source', 'unknown'))

    print(f"Loaded {len(prompts)} samples.")

    print(f"Loading model {model_path}...")
    start_time = time.time()
    llm = LLM(
        model=model_path, 
        tensor_parallel_size=args.tp, 
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_util,
        dtype=args.dtype
    )
    tokenizer = llm.get_tokenizer()
    print(f"Model loaded in {time.time() - start_time:.2f}s")
    
    formatted_prompts = []
    for msgs in prompts:
        # Standard chat template application
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(formatted)

    sampling_params = SamplingParams(n=args.n_samples, temperature=1.0, top_p=1.0, max_tokens=args.max_response_length)
    
    print(f"Generating {args.n_samples} responses per prompt...")
    gen_start_time = time.time()
    # Apply max_prompt_length truncation
    truncated_prompts = []
    for p in formatted_prompts:
        tokens = tokenizer.encode(p)
        if len(tokens) > args.max_prompt_length:
            # Truncate from the left (keep the end) to fit
            truncated_tokens = tokens[-args.max_prompt_length:]
            truncated_prompts.append(tokenizer.decode(truncated_tokens))
        else:
            truncated_prompts.append(p)
            
    outputs = llm.generate(truncated_prompts, sampling_params)
    gen_time = time.time() - gen_start_time
    print(f"Generation finished in {gen_time:.2f}s")

    print("Scoring...")
    score_start_time = time.time()
    results = []
    
    # Calculate metrics for powers of 2 up to n_samples
    k_values = []
    k = 2
    while k <= args.n_samples:
        k_values.append(k)
        k *= 2
    
    # Metrics containers
    total_metrics = {f"pass@{k}": [] for k in k_values}
    total_metrics["mean"] = []
    
    # Group by data source
    source_metrics = {}

    for i, output in enumerate(outputs):
        gt = ground_truths[i]
        ds = data_sources[i]
        
        if ds not in source_metrics:
            source_metrics[ds] = {f"pass@{k}": [] for k in k_values}
            source_metrics[ds]["mean"] = []

        scores = []
        responses = []
        for sample in output.outputs:
            response = sample.text
            responses.append(response)
            # compute_score returns dict with 'score', 'acc', 'format_ok'
            # We use 'score' which is 0.0 or 1.0
            reward_dict = compute_score(ds, response, gt)
            scores.append(reward_dict['score'])
            
        n = len(scores)
        c = sum(1 for s in scores if s >= 1.0) 
        
        mean_score = sum(scores) / n
        
        metrics = {"mean": mean_score}
        total_metrics["mean"].append(mean_score)
        source_metrics[ds]["mean"].append(mean_score)

        for k in k_values:
            pk = pass_at_k(n, c, k)
            metrics[f"pass@{k}"] = pk
            total_metrics[f"pass@{k}"].append(pk)
            source_metrics[ds][f"pass@{k}"].append(pk)

        results.append({
            "index": i,
            "data_source": ds,
            "ground_truth": gt,
            "metrics": metrics,
            "scores": scores,
            "responses": responses
        })

    # Summarize
    summary = {}
    for k, v in total_metrics.items():
        if not v: continue
        summary[f"overall/{k}"] = sum(v) / len(v)
        if k == "mean":
            summary[f"overall/{k}_std"] = np.std(v)
        
    for ds, metrics in source_metrics.items():
        for k, v in metrics.items():
            if not v: continue
            summary[f"{ds}/{k}"] = sum(v) / len(v)
            if k == "mean":
                summary[f"{ds}/{k}_std"] = np.std(v)
            
    print("Results Summary:")
    print(json.dumps(summary, indent=2))
    print(f"Scoring finished in {time.time() - score_start_time:.2f}s")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Always save details to Parquet
    if not output_file.endswith('.parquet'):
        output_file += '.parquet'
        
    df_results = pd.DataFrame(results)
    df_results.to_parquet(output_file)
    print(f"Saved detailed results to {output_file}")
    
    # Save summary to JSON sidecar
    summary_path = output_file.replace('.parquet', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    # Logging
    # Only log if directories are set and NOT "none"
    if args.tensorboard_dir and args.tensorboard_dir != "none":
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.tensorboard_dir)
        for k, v in summary.items():
            writer.add_scalar(k, v, global_step=args.step)
        writer.close()
        print(f"Logged to TensorBoard: {args.tensorboard_dir}")
        
    if args.wandb_project and args.wandb_project != "none":
        import wandb
        # Use run_name as ID to allow resuming if loop calls this script multiple times
        run_id = args.wandb_run_name if args.wandb_run_name else None
        wandb.init(
            project=args.wandb_project, 
            name=args.wandb_run_name, 
            id=run_id, 
            resume="allow"
        )
        wandb.log(summary, step=args.step)
        wandb.finish()
        print(f"Logged to WandB project: {args.wandb_project}")

def main(args):
    # --- Path Resolution & Defaults ---
    
    # 1. Detect Iteration Mode
    checkpoints = []
    if not args.force_single:
        checkpoints = find_checkpoints(args.model)
    
    # 2. Extract Experiment Name (for defaults)
    path_parts = args.model.rstrip('/').split('/')
    exp_name = path_parts[-1]
    # If we are in a subdirectory (like global_step_X), go up
    if exp_name.startswith('global_step_') or exp_name == 'actor':
        for part in reversed(path_parts):
            if not (part.startswith('global_step_') or part == 'actor'):
                exp_name = part
                break

    # 3. Resolve Output Directory
    if args.output is None:
        if len(checkpoints) > 0:
            # Loop mode: use model_dir/val
            args.output = os.path.join(args.model, "val")
        else:
            # Single mode: use exp_dir/val
            # Try to find parent experiment dir if we are deep in structure
            parent_dir = os.path.dirname(args.model.rstrip('/'))
            if os.path.basename(args.model) == 'actor': 
                parent_dir = os.path.dirname(os.path.dirname(args.model.rstrip('/')))
            # If parent_dir is a step dir, go up one more
            if os.path.basename(parent_dir).startswith('global_step_'):
                parent_dir = os.path.dirname(parent_dir)
                
            args.output = os.path.join(parent_dir, "val")

    # 4. Resolve Logging Defaults
    # TensorBoard
    if args.tensorboard_dir is None:
        # Auto-detect default
        args.tensorboard_dir = f"/workspace/tensorboard_logs/{exp_name}"
        print(f"Auto-detected TensorBoard dir: {args.tensorboard_dir}")
    elif args.tensorboard_dir == "none":
        # Explicitly disabled
        print("TensorBoard logging disabled by user.")
        # Important: Don't set to None here, keep as "none" so subprocess knows to disable it
    
    # WandB
    if args.wandb_project is None:
        args.wandb_project = DEFAULT_WANDB_PROJECT
    elif args.wandb_project == "none":
        print("WandB logging disabled by user.")
        # Important: Don't set to None here, keep as "none" so subprocess knows to disable it

    # ----------------------------------

    if len(checkpoints) > 0:
        # MANAGER MODE: Iterate through checkpoints
        print(f"Found {len(checkpoints)} checkpoints. Starting evaluation loop...")
        
        # Setup shared WandB ID if needed
        wandb_run_name = args.wandb_run_name
        if args.wandb_project and args.wandb_project != "none" and not wandb_run_name:
             wandb_run_name = f"eval-{exp_name}"
             print(f"Generated WandB Run ID: {wandb_run_name}")

        for step, ckpt_path in checkpoints:
            print(f"\n=== Processing Step {step} ===")
            
            # Construct output filename
            if os.path.isdir(args.output) or not args.output.endswith(('.json', '.parquet')):
                os.makedirs(args.output, exist_ok=True)
                out_file = os.path.join(args.output, f"eval_step_{step}.parquet")
            else:
                # If user gave a file path but we have multiple checkpoints, append step
                base, ext = os.path.splitext(args.output)
                out_file = f"{base}_step_{step}{ext}"

            if os.path.exists(out_file):
                print(f"Output {out_file} exists. Skipping.")
                continue

            # Construct command to run self in worker mode
            cmd = [
                sys.executable, __file__,
                "--model", ckpt_path,
                "--data", args.data,
                "--output", out_file,
                "--n_samples", str(args.n_samples),
                "--max_response_length", str(args.max_response_length),
                "--max_prompt_length", str(args.max_prompt_length),
                "--tp", str(args.tp),
                "--gpu_util", str(args.gpu_util),
                "--dtype", args.dtype,
                "--step", str(step),
                "--reward", args.reward,
                "--force-single" # Important: prevent recursion
            ]
            
            # Pass resolved logging args
            if args.tensorboard_dir:
                cmd.extend(["--tensorboard_dir", args.tensorboard_dir])
            
            if args.wandb_project:
                cmd.extend(["--wandb_project", args.wandb_project])
                if wandb_run_name:
                    cmd.extend(["--wandb_run_name", wandb_run_name])

            # Run subprocess
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError as e:
                print(f"Error evaluating step {step}: {e}")
                # Continue to next step?
                
    else:
        # WORKER MODE: Evaluate single model
        step = args.step if args.step is not None else infer_step(args.model)
        # Update args.step for logging
        args.step = step
        
        # Determine output filename if it's a directory
        if os.path.isdir(args.output) or not args.output.endswith(('.json', '.parquet')):
             os.makedirs(args.output, exist_ok=True)
             args.output = os.path.join(args.output, f"eval_step_{step}.parquet")
        
        # Ensure output ends with .parquet if it's a file path
        elif not args.output.endswith('.parquet'):
             args.output += '.parquet'

        # Default WandB run name for single mode if not provided
        if args.wandb_project and args.wandb_project != "none" and args.wandb_run_name is None:
             args.wandb_run_name = f"eval-{exp_name}-step-{step}"

        evaluate_model(args.model, args.output, args)

if __name__ == "__main__":
    """
    Example Usage:
    
    1. Evaluate a single model checkpoint:
       python3 verl/power/eval_checkpoint.py --model /path/to/ckpt --tp 4
       
    2. Evaluate all checkpoints in an experiment directory:
       python3 verl/power/eval_checkpoint.py --model /path/to/experiment_dir --tp 4
       
    3. Custom reward function and disable logging:
       python3 verl/power/eval_checkpoint.py --model /path/to/ckpt --reward ./my_reward.py --tensorboard_dir none --wandb_project none
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model checkpoint or directory of checkpoints")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to test.parquet")
    parser.add_argument("--output", default=None, help="Path to output file/dir. Defaults to {model_dir}/val")
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES, help="Number of samples per prompt")
    parser.add_argument("--max_response_length", type=int, default=DEFAULT_MAX_RESPONSE_LENGTH)
    parser.add_argument("--max_prompt_length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH)
    parser.add_argument("--tp", type=int, default=DEFAULT_TP_SIZE)
    parser.add_argument("--gpu_util", type=float, default=DEFAULT_GPU_UTIL)
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE)
    parser.add_argument("--step", type=int, default=None, help="Training step for logging (override)")
    parser.add_argument("--tensorboard_dir", type=str, default=None, help="Path to TB logs. Set 'none' to disable.")
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name. Set 'none' to disable.")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--reward", type=str, default=DEFAULT_REWARD_FN, help="Path to reward function file")
    parser.add_argument("--force-single", action="store_true", help="Treat input as single model even if it looks like a dir")
    args = parser.parse_args()
    main(args)
