"""
Evaluate model checkpoints on test data using vLLM.

Supported model sources:
  - HuggingFace Hub:  Qwen/Qwen2.5-0.5B-Instruct (auto-download)
  - Local path:       /path/to/checkpoint
  - GCS shorthand:    gcs:exp_name (uses default bucket)
  - GCS explicit:     gs://bucket/exp_name

Arguments and defaults (Only --model is required. All others have defaults)
  --model             (REQUIRED) Model path, HF ID, or gcs:exp_name
  --data              test.parquet
  --output            ./validation/{exp_name}
  --n_samples         8
  --max_response_length 8192
  --max_prompt_length 2048
  --tp                {GPU count} or 1
  --gpu_util          0.95
  --dtype             auto
  --step              0 (or inferred from global_step_X)
  --tensorboard_dir   /workspace/tensorboard_logs/{exp_name}
  --wandb_project     verl_eval (if API key found, else disabled)
  --wandb_run         eval-{exp_name}
  --exp_name          auto-detected from model path
  --reward            verl/power/reward.py
  --keep_downloads    False (delete GCS downloads after eval)
  --force-single      False

Examples:

  # Evaluate base model from HuggingFace
  python3 eval_checkpoint.py --model Qwen/Qwen2.5-0.5B-Instruct
  
  # Evaluate local checkpoint
  python3 eval_checkpoint.py --model /path/to/checkpoint --tp 4
  
  # Evaluate all checkpoints in local experiment dir
  python3 eval_checkpoint.py --model /path/to/experiment_dir --tp 4
  
  # Evaluate from GCS (downloads to ./checkpoints/, deletes after)
  python3 eval_checkpoint.py --model gcs:GRPO-1.5B-exp --tp 4
  
  # Keep downloaded checkpoints for inspection
  python3 eval_checkpoint.py --model gcs:GRPO-1.5B-exp --keep_downloads
  
  # Disable logging
  python3 eval_checkpoint.py --model gcs:exp --tensorboard_dir none --wandb_project none
"""

import argparse
import glob
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from vllm import LLM, SamplingParams

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ================= Configuration Defaults =================
DEFAULT_GCS_BUCKET = "gogo-verl-checkpoints"
DEFAULT_GCS_CREDENTIALS = "/wx-gcs-key.json"
DEFAULT_WANDB_KEY_FILE = "/workspace/wx-wandb-api-key.txt"

DEFAULT_N_SAMPLES = 8
DEFAULT_MAX_RESPONSE_LENGTH = 8192
DEFAULT_MAX_PROMPT_LENGTH = 2048
DEFAULT_TP_SIZE = torch.cuda.device_count() if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1
DEFAULT_GPU_UTIL = 0.95
DEFAULT_DTYPE = "auto"
DEFAULT_WANDB_PROJECT = "verl_eval"
DEFAULT_REWARD_FN = os.path.join(os.path.dirname(__file__), "reward.py")
DEFAULT_DATA = "/workspace/data/test.parquet"


# Set GCS credentials if available
if os.path.exists(DEFAULT_GCS_CREDENTIALS):
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", DEFAULT_GCS_CREDENTIALS)
    DEFAULT_GCS_BUCKET = "wx-verl-checkpoints"

# Set WandB API key if available, otherwise disable WandB by default
WANDB_AVAILABLE = False
if os.path.exists(DEFAULT_WANDB_KEY_FILE):
    with open(DEFAULT_WANDB_KEY_FILE) as f:
        os.environ.setdefault("WANDB_API_KEY", f.read().strip())
    WANDB_AVAILABLE = True
elif os.environ.get("WANDB_API_KEY"):
    WANDB_AVAILABLE = True

# ================= GCS Helpers =================
def is_gcs_path(path):
    """Check if path is a GCS path."""
    return path.startswith("gs://")

def parse_gcs_path(gcs_path):
    """Parse gs://bucket/prefix into (bucket, prefix)."""
    path = gcs_path[5:]  # Remove gs://
    parts = path.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix

def list_gcs_checkpoints(gcs_path):
    """
    List checkpoint directories in GCS.
    Returns list of (step, gcs_path).
    """
    from google.cloud import storage
    
    bucket, prefix = parse_gcs_path(gcs_path)
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    
    # List "directories" under prefix by looking for global_step_*/
    checkpoints = []
    blobs = client.list_blobs(bucket, prefix=prefix.rstrip('/') + '/', delimiter='/')
    
    # Consume iterator to get prefixes
    list(blobs)  # Need to iterate to populate prefixes
    
    for blob_prefix in blobs.prefixes:
        # blob_prefix looks like: exp_name/global_step_100/
        match = re.search(r"global_step_(\d+)", blob_prefix)
        if match:
            step = int(match.group(1))
            # Check if actor subdir exists
            actor_prefix = blob_prefix + "actor/"
            actor_blobs = list(client.list_blobs(bucket, prefix=actor_prefix, max_results=1))
            if actor_blobs:
                ckpt_gcs_path = f"gs://{bucket}/{blob_prefix}actor"
            else:
                ckpt_gcs_path = f"gs://{bucket}/{blob_prefix.rstrip('/')}"
            checkpoints.append((step, ckpt_gcs_path))
    
    checkpoints.sort()
    return checkpoints

def download_gcs_checkpoint(gcs_path, exp_name=None):
    """
    Download a checkpoint from GCS to ./checkpoints/{exp_name}/.
    Returns local path.
    """
    from google.cloud import storage
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    bucket, prefix = parse_gcs_path(gcs_path)
    
    # Determine exp_name from path if not provided
    if exp_name is None:
        parts = prefix.rstrip('/').split('/')
        exp_name = parts[0] if parts else "unknown"
    
    # Extract checkpoint name (e.g., global_step_100)
    ckpt_name = os.path.basename(prefix.rstrip('/'))
    local_ckpt_dir = os.path.join("./checkpoints", exp_name, ckpt_name)
    
    # If already downloaded, reuse
    if os.path.isdir(local_ckpt_dir) and os.listdir(local_ckpt_dir):
        log.info(f"Using cached checkpoint: {local_ckpt_dir}")
        return local_ckpt_dir
    
    os.makedirs(local_ckpt_dir, exist_ok=True)
    
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    
    # List all blobs under prefix
    blobs = list(client.list_blobs(bucket, prefix=prefix.rstrip('/') + '/'))
    
    if not blobs:
        raise ValueError(f"No files found at {gcs_path}")
    
    log.info(f"Downloading {len(blobs)} files from {gcs_path}...")
    
    def download_blob(blob):
        # Get relative path from prefix
        rel_path = blob.name[len(prefix.rstrip('/')):].lstrip('/')
        local_path = os.path.join(local_ckpt_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        return local_path
    
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(download_blob, b) for b in blobs]
        for f in as_completed(futures):
            f.result()  # Raise any exceptions
    
    log.info(f"Downloaded to {local_ckpt_dir}")
    return local_ckpt_dir
# ===============================================

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
    Find all checkpoints in a directory (local or GCS).
    Returns sorted list of (step, path).
    """
    # Handle GCS paths
    if is_gcs_path(model_path):
        return list_gcs_checkpoints(model_path)
    
    # Local path
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

# ================= Evaluation Helpers =================

def load_eval_data(data_path):
    """Load evaluation data from parquet file."""
    log.info(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    prompts = []
    ground_truths = []
    data_sources = []
    
    for _, row in df.iterrows():
        prompts.append(row['prompt'])
        gt = None
        if 'reward_model' in row and row['reward_model'] is not None:
            gt = row['reward_model'].get('ground_truth')
        ground_truths.append(gt)
        data_sources.append(row.get('data_source', 'unknown'))
    
    log.info(f"Loaded {len(prompts)} samples.")
    return prompts, ground_truths, data_sources

def generate_responses(llm, prompts, args):
    """Format prompts and generate responses."""
    tokenizer = llm.get_tokenizer()
    
    # Apply chat template
    formatted_prompts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in prompts
    ]
    
    # Truncate to max_prompt_length (left truncation)
    truncated_prompts = []
    for p in formatted_prompts:
        tokens = tokenizer.encode(p)
        if len(tokens) > args.max_prompt_length:
            truncated_tokens = tokens[-args.max_prompt_length:]
            truncated_prompts.append(tokenizer.decode(truncated_tokens))
        else:
            truncated_prompts.append(p)
    
    # Generate
    sampling_params = SamplingParams(
        n=args.n_samples, temperature=1.0, top_p=1.0, max_tokens=args.max_response_length
    )
    
    log.info(f"Generating {args.n_samples} responses per prompt...")
    start_time = time.time()
    outputs = llm.generate(truncated_prompts, sampling_params)
    log.info(f"Generation finished in {time.time() - start_time:.2f}s")
    
    return outputs

def score_responses(outputs, ground_truths, data_sources, compute_score, n_samples):
    """Score responses and compute metrics."""
    log.info("Scoring...")
    start_time = time.time()
    
    # k values for pass@k (powers of 2)
    k_values = []
    k = 2
    while k <= n_samples:
        k_values.append(k)
        k *= 2
    
    # Metrics containers
    total_metrics = {f"pass@{k}": [] for k in k_values}
    total_metrics["mean"] = []
    source_metrics = {}
    results = []

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

    # Build summary (prefix with eval/ for TensorBoard/WandB)
    summary = {}
    for k, v in total_metrics.items():
        if not v: continue
        summary[f"eval/overall/{k}"] = sum(v) / len(v)
        if k == "mean":
            summary[f"eval/overall/{k}_std"] = np.std(v)
        
    for ds, metrics in source_metrics.items():
        for k, v in metrics.items():
            if not v: continue
            summary[f"eval/{ds}/{k}"] = sum(v) / len(v)
            if k == "mean":
                summary[f"eval/{ds}/{k}_std"] = np.std(v)
    
    log.info(f"Scoring finished in {time.time() - start_time:.2f}s")
    return results, summary

def save_results(results, summary, output_file):
    """Save results to parquet and summary to JSON."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if not output_file.endswith('.parquet'):
        output_file += '.parquet'
        
    df_results = pd.DataFrame(results)
    df_results.to_parquet(output_file)
    log.info(f"Saved detailed results to {output_file}")
    
    summary_path = output_file.replace('.parquet', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {summary_path}")
    
    return output_file

def log_metrics(summary, step, args, exp_name=None):
    """Log metrics to TensorBoard and WandB."""
    if args.tensorboard_dir and args.tensorboard_dir != "none":
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.tensorboard_dir)
        for k, v in summary.items():
            writer.add_scalar(k, v, global_step=step)
        writer.close()
        log.info(f"Logged to TensorBoard: {args.tensorboard_dir}")
        
    if args.wandb_project and args.wandb_project != "none":
        import wandb
        run_name = args.wandb_run if args.wandb_run else (f"eval-{exp_name}" if exp_name else "eval")
        wandb.init(project=args.wandb_project, name=run_name, id=run_name, resume="allow")
        wandb.log(summary, step=step)
        wandb.finish()
        log.info(f"Logged to WandB project: {args.wandb_project}, run: {run_name}")

# =====================================================

def evaluate_model(model_path, output_file, args, exp_name=None):
    """Run evaluation for a single model."""
    eval_start = time.time()
    
    # Download from GCS if needed
    downloaded_path = None
    if is_gcs_path(model_path):
        log.info(f"Downloading checkpoint from GCS: {model_path}")
        start = time.time()
        downloaded_path = download_gcs_checkpoint(model_path, exp_name=exp_name)
        model_path = downloaded_path
        log.info(f"Download completed in {time.time() - start:.2f}s")
    
    # Load data and reward function
    compute_score = load_reward_fn(args.reward)
    log.info(f"Using reward function: {args.reward}")
    prompts, ground_truths, data_sources = load_eval_data(args.data)
    
    # Load model
    log.info(f"Loading model {model_path}...")
    start = time.time()
    llm = LLM(
        model=model_path, 
        tensor_parallel_size=args.tp, 
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_util,
        dtype=args.dtype
    )
    log.info(f"Model loaded in {time.time() - start:.2f}s")
    
    # Generate and score
    outputs = generate_responses(llm, prompts, args)
    results, summary = score_responses(outputs, ground_truths, data_sources, compute_score, args.n_samples)
    
    # Print summary
    log.info(f"Results Summary:\n{json.dumps(summary, indent=2)}")
    
    # Save and log
    save_results(results, summary, output_file)
    log_metrics(summary, args.step, args, exp_name)
    
    # Cleanup
    if downloaded_path and not args.keep_downloads:
        log.info(f"Cleaning up downloaded checkpoint: {downloaded_path}")
        shutil.rmtree(downloaded_path, ignore_errors=True)
    
    log.info(f"Total evaluation time: {time.time() - eval_start:.2f}s")

def main(args):
    # --- Path Resolution & Defaults ---
    
    # 0. Resolve model path (handle gcs: shorthand)
    if args.model.startswith("gcs:"):
        # gcs:exp_name → gs://{DEFAULT_GCS_BUCKET}/exp_name
        args.model = f"gs://{DEFAULT_GCS_BUCKET}/{args.model[4:]}"
        log.info(f"Using GCS path: {args.model}")
    
    # 1. Detect Iteration Mode
    checkpoints = []
    if not args.force_single:
        checkpoints = find_checkpoints(args.model)
    
    # 2. Extract Experiment Name (for defaults)
    # Handle both local and GCS paths
    model_path_str = args.model.replace("gs://", "").rstrip('/')
    path_parts = model_path_str.split('/')
    exp_name = path_parts[-1]
    # If pointing to a global_step_X subdir, go up one level
    if exp_name.startswith('global_step_'):
        exp_name = path_parts[-2] if len(path_parts) > 1 else exp_name

    # 3. Resolve Output Directory
    if args.output is None:
        args.output = f"./validation/{exp_name}"

    # 4. Resolve Logging Defaults
    # TensorBoard
    if args.tensorboard_dir is None:
        # Auto-detect default
        args.tensorboard_dir = f"/workspace/tensorboard_logs/{exp_name}"
        log.info(f"Auto-detected TensorBoard dir: {args.tensorboard_dir}")
    elif args.tensorboard_dir == "none":
        # Explicitly disabled
        log.info("TensorBoard logging disabled by user.")
        # Important: Don't set to None here, keep as "none" so subprocess knows to disable it
    
    # WandB
    if args.wandb_project is None:
        if WANDB_AVAILABLE:
            args.wandb_project = DEFAULT_WANDB_PROJECT
        else:
            args.wandb_project = "none"
            log.info("WandB disabled: no API key found.")
    elif args.wandb_project == "none":
        log.info("WandB logging disabled by user.")
        # Important: Don't set to None here, keep as "none" so subprocess knows to disable it

    # ----------------------------------

    if len(checkpoints) > 0:
        # MANAGER MODE: Iterate through checkpoints
        log.info(f"Found {len(checkpoints)} checkpoints. Starting evaluation loop...")
        loop_start = time.time()

        for step, ckpt_path in checkpoints:
            log.info(f"\n=== Processing Step {step} ===")
            
            # Construct output filename
            if os.path.isdir(args.output) or not args.output.endswith(('.json', '.parquet')):
                os.makedirs(args.output, exist_ok=True)
                out_file = os.path.join(args.output, f"eval_step_{step}.parquet")
            else:
                # If user gave a file path but we have multiple checkpoints, append step
                base, ext = os.path.splitext(args.output)
                out_file = f"{base}_step_{step}{ext}"

            if os.path.exists(out_file):
                log.info(f"Output {out_file} exists. Skipping.")
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
                if args.wandb_run:
                    cmd.extend(["--wandb_run", args.wandb_run])
            
            # Pass exp_name so subprocess uses consistent WandB run name
            cmd.extend(["--exp_name", exp_name])
            
            if args.keep_downloads:
                cmd.append("--keep_downloads")

            # Run subprocess
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError as e:
                log.info(f"Error evaluating step {step}: {e}")
        
        log.info(f"All checkpoints evaluated. Total loop time: {time.time() - loop_start:.2f}s")
                
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

        # Use provided exp_name (from manager) or extracted one
        final_exp_name = args.exp_name if args.exp_name else exp_name
        evaluate_model(args.model, args.output, args, exp_name=final_exp_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    parser.add_argument("--wandb_run", type=str, default=None, help="WandB run name (default: eval-{exp_name})")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name (auto-detected if not provided)")
    parser.add_argument("--reward", type=str, default=DEFAULT_REWARD_FN, help="Path to reward function file")
    parser.add_argument("--keep_downloads", action="store_true", help="Keep downloaded GCS checkpoints (default: delete after eval)")
    parser.add_argument("--force-single", action="store_true", help="Treat input as single model even if it looks like a dir")
    args = parser.parse_args()
    main(args)
