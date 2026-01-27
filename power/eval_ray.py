"""
Evaluate model checkpoints on test data using Ray + vLLM.

Ray handles GPU assignment and task scheduling automatically.

Supported model sources:
  - HuggingFace Hub:  Qwen/Qwen2.5-0.5B-Instruct (auto-download)
  - Local path:       /path/to/checkpoint
  - GCS shorthand:    gcs:exp_name (uses default bucket)
  - GCS explicit:     gs://bucket/exp_name

Arguments and defaults (Only --model is required. All others have defaults)
  --model             (REQUIRED) Model path, HF ID, or gcs:exp_name
  --data              /workspace/data/test.parquet (comma-separated for multiple)
  --output            ./validation/{exp_name}
  --n_samples         8
  --max_response_length 8192
  --max_prompt_length 2048
  --tensor_parallel_size  1 (TP size per eval)
  --num_gpus          {all available} (for parallel workers)
  --resume            auto | <step> (default: auto, e.g. --resume 500)
  --gpu_memory_utilization  0.95
  --dtype             auto
  --step              0 (or inferred from global_step_X)
  --tensorboard_dir   /workspace/tensorboard_logs/{exp_name}
  --wandb_project     verl_eval (if API key found, else disabled)
  --wandb_run         eval-{exp_name}
  --exp_name          auto-detected from model path
  --reward            verl/power/reward.py
  --stats_only        only save summary JSON, skip rollouts parquet
  --keep_downloads    keep downloaded GCS checkpoints after eval
  --force-single      treat input as single model even if directory

Output files:
  eval_step_{N}_summary.json  - Aggregated metrics (always saved)
  eval_step_{N}.parquet       - Detailed rollouts (default, skip with --stats_only)

Examples:

  # Evaluate base model from HuggingFace
  python3 eval_ray.py --model Qwen/Qwen2.5-0.5B-Instruct

  # Evaluate all checkpoints (auto-parallel on all GPUs)
  python3 eval_ray.py --model gcs:GRPO-1.5B-exp

  # Use 4 GPUs for parallel eval (4 checkpoints at once)
  python3 eval_ray.py --model /path/to/experiment_dir --num_gpus 4

  # Large model: use TP=4 on 4 GPUs (sequential, not parallel)
  python3 eval_ray.py --model gcs:GRPO-70B-exp --tensor_parallel_size 4 --num_gpus 1

  # Disable logging
  python3 eval_ray.py --model gcs:exp --tensorboard_dir none --wandb_project none
"""

import argparse
import importlib.util
import json
import logging
import math
import os
import re
import glob
import shutil
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

# Silence tokenizers parallelism warning (Ray uses forking)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from google.cloud import storage
import wandb
import ray
from vllm import LLM, SamplingParams

# Logging setup
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

DEFAULT_N_SAMPLES = 4
DEFAULT_MAX_RESPONSE_LENGTH = 8192
DEFAULT_MAX_PROMPT_LENGTH = 2048
DEFAULT_TP_SIZE = 1  # TP=1 is optimal for small models, use parallel workers instead
DEFAULT_NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1
DEFAULT_GPU_UTIL = 0.95
DEFAULT_DTYPE = "auto"
DEFAULT_WANDB_PROJECT = "verl_eval"
DEFAULT_REWARD_FN = os.path.join(os.path.dirname(__file__), "reward.py")
# DEFAULT_DATA = "/workspace/data/math500.parquet,/workspace/data/aime24.parquet,/workspace/data/aime25.parquet,/workspace/data/amc23.parquet,/workspace/data/olympiad_bench.parquet,/workspace/data/minerva.parquet"
DEFAULT_DATA = "/workspace/data/aime24.parquet,/workspace/data/aime25.parquet,/workspace/data/amc23.parquet"

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

# ============ GCS Utilities ============

def is_gcs_path(path: str) -> bool:
    return path.startswith("gs://") or path.startswith("gcs:")

def parse_gcs_path(gcs_path: str):
    """Parse gs://bucket/prefix into (bucket, prefix)."""
    path = gcs_path[5:]  # Remove gs://
    parts = path.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix

def list_gcs_checkpoints(gcs_path: str):
    """List checkpoint directories in GCS. Returns list of (step, gcs_path).
    
    verl checkpoint structure: global_step_X/actor/huggingface/ (contains config.json)
    We return the path to actor/ (which contains weights), and download_gcs_checkpoint
    will handle finding the huggingface/ subdir for vLLM.
    """
    bucket, prefix = parse_gcs_path(gcs_path)
    client = storage.Client()
    
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
            # Check if actor subdir exists (verl structure)
            actor_prefix = blob_prefix + "actor/"
            actor_blobs = list(client.list_blobs(bucket, prefix=actor_prefix, max_results=1))
            if actor_blobs:
                # Return path to actor/ - we'll download entire actor/ including huggingface/
                ckpt_gcs_path = f"gs://{bucket}/{blob_prefix}actor"
            else:
                ckpt_gcs_path = f"gs://{bucket}/{blob_prefix.rstrip('/')}"
            checkpoints.append((step, ckpt_gcs_path))
    
    checkpoints.sort()
    return checkpoints

def find_model_dir(base_dir: str) -> str:
    """Find the directory containing config.json (what vLLM needs).
    
    verl structure: actor/huggingface/ contains config.json
    Standard HF: config.json is in the root
    """
    # Check if config.json exists directly
    if os.path.exists(os.path.join(base_dir, "config.json")):
        return base_dir
    
    # Check huggingface/ subdir (verl structure: actor/huggingface/)
    hf_dir = os.path.join(base_dir, "huggingface")
    if os.path.isdir(hf_dir) and os.path.exists(os.path.join(hf_dir, "config.json")):
        return hf_dir
    
    # Fallback to base_dir (will likely fail, but let vLLM give the error)
    return base_dir


def download_gcs_checkpoint(gcs_path: str, exp_name: str = None):
    """Download a checkpoint from GCS to ./checkpoints/{exp_name}/. Returns local path.
    
    Returns the path to the directory containing config.json (for vLLM).
    For verl checkpoints, this is actor/huggingface/.
    """
    bucket, prefix = parse_gcs_path(gcs_path)
    
    # Determine exp_name from path if not provided
    if exp_name is None:
        parts = prefix.rstrip('/').split('/')
        exp_name = parts[0] if parts else "unknown"
    
    # Build local path preserving structure: ./checkpoints/{exp_name}/global_step_X/actor/
    # Extract the global_step_X part if present
    match = re.search(r"(global_step_\d+)", prefix)
    if match:
        step_dir = match.group(1)
        # Get remaining path after global_step_X (e.g., "actor")
        after_step = prefix[prefix.find(step_dir) + len(step_dir):].strip('/')
        local_ckpt_dir = os.path.join("./checkpoints", exp_name, step_dir, after_step) if after_step else os.path.join("./checkpoints", exp_name, step_dir)
    else:
        ckpt_name = os.path.basename(prefix.rstrip('/'))
        local_ckpt_dir = os.path.join("./checkpoints", exp_name, ckpt_name)
    
    # If already downloaded, reuse if valid
    if os.path.isdir(local_ckpt_dir) and os.listdir(local_ckpt_dir):
        model_dir = find_model_dir(local_ckpt_dir)
        if os.path.exists(os.path.join(model_dir, "config.json")):
            log.info(f"Using cached checkpoint: {model_dir}")
            return model_dir
        else:
            log.warning(f"Found cached directory {local_ckpt_dir} but config.json is missing. Re-downloading.")
            shutil.rmtree(local_ckpt_dir)
    
    os.makedirs(local_ckpt_dir, exist_ok=True)
    
    client = storage.Client()
    
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
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(download_blob, blobs))
    
    log.info(f"Downloaded to {local_ckpt_dir}")
    
    # Return the directory containing config.json
    model_dir = find_model_dir(local_ckpt_dir)
    log.info(f"Model directory (for vLLM): {model_dir}")
    return model_dir


# ============ Model/Data Utilities ============

def load_reward_function(reward_path: str):
    """Dynamically load compute_score from reward file."""
    spec = importlib.util.spec_from_file_location("reward_module", reward_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score

def pass_at_k(n, c, k):
    """
    Calculate unbiased pass@k estimator.
    n: total samples
    c: correct samples  
    k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def infer_step(model_path: str) -> int:
    """Infer step from path like .../global_step_100/..."""
    match = re.search(r'global_step_(\d+)', model_path)
    return int(match.group(1)) if match else 0


# ============ Ray Worker ============

@ray.remote
class EvalWorker:
    """Ray actor for evaluation. Holds model in GPU memory."""
    
    def __init__(self, tensor_parallel_size: int, gpu_memory_utilization: float, dtype: str):
        self.tp = tensor_parallel_size
        self.gpu_util = gpu_memory_utilization
        self.dtype = dtype
        self.llm = None
        self.tokenizer = None
        self.current_model = None
    
    def load_model(self, model_path: str):
        """Load model if different from current."""
        if model_path == self.current_model:
            return
                
        # Cleanup previous model
        if self.llm is not None:
            del self.llm
            self.llm = None
            torch.cuda.empty_cache()
        
        log.info(f"Loading model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=self.tp,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_util,
            dtype=self.dtype,
        )
        self.current_model = model_path
    
    def evaluate(
        self,
        model_path: str,
        data_path: str,
        step: int,
        n_samples: int,
        max_response_length: int,
        max_prompt_length: int,
        reward_path: str,
        data_name: str = None,
    ):
        """Run evaluation on a single checkpoint and data file. Returns (step, data_name, df_results, summary)."""
        
        eval_start = time.time()
        
        # Load model
        self.load_model(model_path)
        
        # Load data
        df = pd.read_parquet(data_path)
        log.info(f"Loaded {len(df)} prompts from {data_path}")
        
        prompts = df["prompt"].tolist()
        formatted = []
        for p in prompts:
            # Apply chat template
            text = self.tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            # Truncate if needed
            tokens = self.tokenizer.encode(text)
            if len(tokens) > max_prompt_length:
                tokens = tokens[-max_prompt_length:]
                text = self.tokenizer.decode(tokens)
            formatted.append(text)
        
        # Generate
        log.info(f"Generating {len(formatted)} × {n_samples} responses on {data_name}...")
        gen_start = time.time()
        params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=max_response_length,
            n=n_samples,
        )
        outputs = self.llm.generate(formatted, params)
        log.info(f"Generation of {len(formatted)} × {n_samples} responses on {data_name} took {time.time() - gen_start:.2f}s")
        
        # Score
        log.info(f"Scoring {len(outputs)} responses on {data_name}...")
        score_start = time.time()
        compute_score = load_reward_function(reward_path)
        
        results = []
        for i, (output, row) in enumerate(zip(outputs, df.itertuples())):
            # Extract ground_truth from reward_model dict (VeRL format from prepare_data.py)
            gt = None
            if hasattr(row, 'reward_model') and row.reward_model is not None:
                gt = row.reward_model.get('ground_truth')
            data_source = row.data_source if hasattr(row, 'data_source') else "unknown"
            
            for j, resp in enumerate(output.outputs):
                # compute_score(data_source, response, ground_truth) returns dict with 'score'
                reward_dict = compute_score(data_source, resp.text, gt)
                score = reward_dict['score'] if isinstance(reward_dict, dict) else reward_dict
                results.append({
                    "prompt_idx": i,
                    "sample_idx": j,
                    "prompt": prompts[i],
                    "response": resp.text,
                    "ground_truth": gt,
                    "score": score,
                    "data_source": data_source,
                })
        
        log.info(f"Scoring {len(outputs)} responses on {data_name} took {time.time() - score_start:.2f}s")
        
        # Compute metrics for this data file
        df_results = pd.DataFrame(results)
        
        # Metrics keyed by data_name (the file being evaluated)
        summary = {}
        summary[f"eval/{data_name}/mean"] = df_results["score"].mean()
        
        # Pass@k metrics (unbiased estimator)
        grouped = df_results.groupby("prompt_idx")["score"]
        # k_values: powers of 2 up to n_samples (2, 4, 8, 16, ...)
        k_values = [2**i for i in range(1, 10) if 2**i <= n_samples]
        
        for k in k_values:
            # For each prompt: n=total samples, c=correct samples
            pass_k_scores = []
            for prompt_idx, scores in grouped:
                n = len(scores)
                c = int(scores.sum())
                pass_k_scores.append(pass_at_k(n, c, k))
            summary[f"eval/{data_name}/pass@{k}"] = np.mean(pass_k_scores)
        
        # Best@N (max score per prompt, averaged)
        summary[f"eval/{data_name}/best@N"] = grouped.max().mean()
        
        total_time = time.time() - eval_start
        log.info(f"Step {step} on {data_name} completed in {total_time:.2f}s. Mean: {summary[f'eval/{data_name}/mean']:.4f}")
        
        # Return results for aggregation (don't write files here)
        return step, data_name, df_results, summary


# ============ Main ============

def find_checkpoints(model_path: str):
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
            match = re.search(r"global_step_(\d+)", s)
            if match:
                step = int(match.group(1))
                # Check for actor subdir (standard verl structure)
                actor_path = os.path.join(s, "actor")
                ckpt_path = actor_path if os.path.isdir(actor_path) else s
                checkpoints.append((step, ckpt_path))
        checkpoints.sort()
    return checkpoints


def get_data_name(data_path: str) -> str:
    """Extract short name from data path for output files."""
    name = os.path.basename(data_path).replace(".parquet", "")
    return name


def main(args):
    log.info(f"Starting evaluation: {args.model}")
    
    # Parse data files (comma-separated)
    data_files = [f.strip() for f in args.data.split(",")]
    log.info(f"Data files: {data_files}")
    
    # 0. Resolve model path (handle gcs: shorthand)
    if args.model.startswith("gcs:"):
        args.model = f"gs://{DEFAULT_GCS_BUCKET}/{args.model[4:]}"
        log.info(f"Using GCS path: {args.model}")
    
    # 1. Detect Iteration Mode - find checkpoints
    checkpoints = []
    if not args.force_single:
        checkpoints = find_checkpoints(args.model)
    
    # 2. Extract Experiment Name (for defaults)
    # Handle both local and GCS paths
    model_path_str = args.model.replace("gs://", "").rstrip('/')
    path_parts = model_path_str.split('/')
    exp_name = args.exp_name or path_parts[-1]
    # If pointing to a global_step_X subdir, go up one level
    if exp_name.startswith('global_step_'):
        exp_name = path_parts[-2] if len(path_parts) > 1 else exp_name
    
    # 3. Resolve Output Directory
    if args.output is None:
        args.output = f"./validation/{exp_name}"
    
    # 4. Resolve Logging Defaults
    # TensorBoard
    if args.tensorboard_dir is None:
        args.tensorboard_dir = f"/workspace/tensorboard_logs/{exp_name}"
        log.info(f"Auto-detected TensorBoard dir: {args.tensorboard_dir}")
    elif args.tensorboard_dir == "none":
        log.info("TensorBoard logging disabled by user.")
    
    # WandB
    if args.wandb_project is None:
        if WANDB_AVAILABLE:
            args.wandb_project = DEFAULT_WANDB_PROJECT
        else:
            args.wandb_project = "none"
            log.info("WandB disabled: no API key found.")
    elif args.wandb_project == "none":
        log.info("WandB logging disabled by user.")
    
    # Single model mode
    if not checkpoints:
        step = args.step if args.step is not None else infer_step(args.model)
        checkpoints = [(step, args.model)]
    
    log.info(f"Found {len(checkpoints)} checkpoint(s)")
    
    # Parse resume
    resume_from_step = None
    if args.resume != "auto":
        try:
            resume_from_step = int(args.resume)
            if resume_from_step > 0:
                log.info(f"Starting from step >= {resume_from_step}")
        except ValueError:
            log.warning(f"Invalid --resume value: {args.resume}")
    
    # Build pending tasks: (step, ckpt_path, data_files_list)
    # Group by checkpoint so same model evaluates all data files
    pending = []
    for step, ckpt_path in checkpoints: # order of pending tasks is ordered since checkpoints are sorted
        if resume_from_step is not None and step < resume_from_step:
            continue
        
        # Skip if combined summary already exists (not per-dataset, since we aggregate)
        summary_file = os.path.join(args.output, f"eval_step_{step}_summary.json")
        if resume_from_step is None and os.path.exists(summary_file):
            log.info(f"Step {step} already evaluated, skipping")
            continue
        
        # Add all data files for this checkpoint
        for data_file in data_files:
            pending.append((step, ckpt_path, data_file))
    
    if not pending:
        log.info("All checkpoints already evaluated.")
        return
    
    log.info(f"Evaluating {len(pending)} task(s) with {args.num_gpus} GPU(s)")
    
    # Initialize Ray
    num_workers = args.num_gpus // args.tensor_parallel_size
    ray.init(num_gpus=args.num_gpus)
    
    # Create worker pool
    workers = [
        EvalWorker.options(num_gpus=args.tensor_parallel_size).remote(
            args.tensor_parallel_size,
            args.gpu_memory_utilization,
            args.dtype
        )
        for _ in range(num_workers)
    ]
    
    # Group pending tasks by checkpoint
    tasks_by_ckpt = defaultdict(list)
    for step, ckpt_path, data_file in pending:
        tasks_by_ckpt[(step, ckpt_path)].append(data_file)
    
    loop_start = time.time()
    
    # Track: checkpoint -> (local_path, remaining_task_count, is_downloaded, worker_idx)
    ckpt_info = {}
    # Track: future -> (step, checkpoint_key)
    future_to_ckpt = {}
    # Track: step -> list of (data_name, df_results, summary) for aggregation
    step_results = defaultdict(list)
    
    # Submit tasks, downloading checkpoints as needed (limit to num_workers at a time)
    pending_futures = []
    all_results = []
    # Sort by step to ensure deterministic order, use deque for efficient popleft
    ckpt_queue = deque(sorted(tasks_by_ckpt.items(), key=lambda x: x[0][0]))
    worker_available = deque(range(num_workers))
    
    def download_and_submit_ckpt(ckpt_key, data_files_for_ckpt, worker_idx):
        """Download checkpoint and submit all its tasks."""
        step, ckpt_path = ckpt_key
        
        # Download from GCS if needed, or find model dir for local paths
        is_downloaded = False
        if is_gcs_path(ckpt_path):
            local_path = download_gcs_checkpoint(ckpt_path)
            is_downloaded = True
        else:
            # For local paths, find the directory with config.json
            local_path = find_model_dir(ckpt_path)
        
        # Track checkpoint info for cleanup and worker assignment
        ckpt_info[ckpt_key] = {
            'local_path': local_path,
            'remaining': len(data_files_for_ckpt),
            'is_downloaded': is_downloaded,
            'worker_idx': worker_idx,
        }
        
        # Submit tasks
        worker = workers[worker_idx]
        futures = []
        for data_file in data_files_for_ckpt:
            data_name = get_data_name(data_file)
            future = worker.evaluate.remote(
                local_path,
                data_file,
                step,
                args.n_samples,
                args.max_response_length,
                args.max_prompt_length,
                args.reward,
                data_name,
            )
            futures.append(future)
            future_to_ckpt[future] = ckpt_key
        return futures
    
    def cleanup_ckpt(ckpt_key):
        """Clean up checkpoint if downloaded and not keeping."""
        info = ckpt_info.get(ckpt_key)
        if not info or not info['is_downloaded'] or args.keep_downloads:
            return
        
        path = info['local_path']
        try:
            # Find the global_step_X directory to delete entirely
            target = path
            while target and not os.path.basename(target).startswith('global_step_'):
                target = os.path.dirname(target)
            if target and os.path.basename(target).startswith('global_step_'):
                shutil.rmtree(target)
                log.info(f"Cleaned up: {target}")
            else:
                # Fallback: just delete path
                shutil.rmtree(path)
                log.info(f"Cleaned up: {path}")
        except Exception as e:
            log.warning(f"Failed to cleanup {path}: {e}")
    
    # Initial submission: up to num_workers checkpoints
    while ckpt_queue and worker_available:
        ckpt_key, data_files_for_ckpt = ckpt_queue.popleft()
        worker_idx = worker_available.popleft()
        futures = download_and_submit_ckpt(ckpt_key, data_files_for_ckpt, worker_idx)
        pending_futures.extend(futures)
    
    # Process results as they complete (no sync barrier!)
    while pending_futures:
        # Wait for any task to complete
        done, pending_futures = ray.wait(pending_futures, num_returns=1)
        
        for future in done:
            step, data_name, df_results, summary = ray.get(future)
            
            # Store results for this step
            step_results[step].append((data_name, df_results, summary))
            
            # Get checkpoint info
            ckpt_key = future_to_ckpt.pop(future)
            info = ckpt_info[ckpt_key]
            info['remaining'] -= 1
            
            log.info(f"Step {step} / {data_name} completed,  ({info['remaining']} datasets remaining for this checkpoint)")
            
            # If all tasks for this checkpoint done, write combined files and clean up
            if info['remaining'] == 0:
                # Aggregate and write combined results for this step
                step_data = step_results[step]
                
                # Combine all DataFrames
                all_dfs = [df for _, df, _ in step_data]
                combined_df = pd.concat(all_dfs, ignore_index=True)
                
                # Combine all summaries + compute overall metrics
                combined_summary = {}
                all_scores = []
                for data_name, df, summary in step_data:
                    combined_summary.update(summary)
                    all_scores.extend(df["score"].tolist())
                
                # Add overall metrics (across all datasets)
                combined_summary["eval/overall/mean"] = np.mean(all_scores)
                
                # Overall pass@k (unbiased estimator, computed per prompt across all datasets)
                grouped = combined_df.groupby(["data_source", "prompt_idx"])["score"]
                k_values = [2**i for i in range(1, 10) if 2**i <= args.n_samples]
                for k in k_values:
                    pass_k_scores = []
                    for _, scores in grouped:
                        n = len(scores)
                        c = int(scores.sum())
                        pass_k_scores.append(pass_at_k(n, c, k))
                    combined_summary[f"eval/overall/pass@{k}"] = np.mean(pass_k_scores)
                
                # Overall best@N
                combined_summary["eval/overall/best@N"] = grouped.max().mean()
                
                # Write combined files
                os.makedirs(args.output, exist_ok=True)
                summary_path = os.path.join(args.output, f"eval_step_{step}_summary.json")
                with open(summary_path, "w") as f:
                    json.dump(combined_summary, f, indent=2)
                log.info(f"Saved combined summary to {summary_path}")
                
                if not args.stats_only:
                    parquet_path = os.path.join(args.output, f"eval_step_{step}.parquet")
                    combined_df.to_parquet(parquet_path)
                    log.info(f"Saved combined rollouts to {parquet_path}")
                
                # Store for TB/WandB logging
                all_results.append((step, combined_summary))
                
                # Get the worker that just finished this checkpoint
                finished_worker_idx = info['worker_idx']
                
                # Cleanup checkpoint
                cleanup_ckpt(ckpt_key)
                
                # Submit next checkpoint to the worker that just finished
                if ckpt_queue:
                    next_ckpt_key, next_data_files = ckpt_queue.popleft()
                    new_futures = download_and_submit_ckpt(next_ckpt_key, next_data_files, finished_worker_idx)
                    pending_futures = list(pending_futures) + new_futures
    
    results = all_results
    
    # Log to TensorBoard and WandB
    if args.tensorboard_dir and args.tensorboard_dir != "none":
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.tensorboard_dir)
        for step, summary in results:
            for k, v in summary.items():
                writer.add_scalar(k, v, global_step=step)
        writer.close()
        log.info(f"Logged to TensorBoard: {args.tensorboard_dir}")
    
    if args.wandb_project and args.wandb_project != "none" and WANDB_AVAILABLE:
        run_name = args.wandb_run or f"eval-{exp_name}"
        wandb.init(project=args.wandb_project, name=run_name, resume="allow")
        for step, summary in results:
            wandb.log(summary, step=step)
        wandb.finish()
        log.info(f"Logged to WandB: {args.wandb_project}/{run_name}")
    
    # Summary
    log.info(f"\nAll {len(results)} steps evaluated. Total time: {time.time() - loop_start:.2f}s")
    for step, summary in sorted(results, key=lambda x: x[0]):
        log.info(f"  Step {step}: overall mean = {summary.get('eval/overall/mean', 0):.4f}")
    
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to model checkpoint or directory of checkpoints")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path(s) to test parquet (comma-separated for multiple)")
    parser.add_argument("--output", default=None, help="Path to output dir. Defaults to ./validation/{exp_name}")
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES, help="Number of samples per prompt")
    parser.add_argument("--max_response_length", type=int, default=DEFAULT_MAX_RESPONSE_LENGTH)
    parser.add_argument("--max_prompt_length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH)
    parser.add_argument("--tensor_parallel_size", type=int, default=DEFAULT_TP_SIZE, help="Tensor parallel size (default: 1)")
    parser.add_argument("--num_gpus", type=int, default=DEFAULT_NUM_GPUS, help="Number of GPUs for parallel eval")
    parser.add_argument("--resume", type=str, default="auto", help="auto (skip if output exists) | <step> (redo from this step)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=DEFAULT_GPU_UTIL)
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE)
    parser.add_argument("--step", type=int, default=None, help="Training step for logging (override)")
    parser.add_argument("--tensorboard_dir", type=str, default=None, help="Path to TB logs. Set 'none' to disable.")
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name. Set 'none' to disable.")
    parser.add_argument("--wandb_run", type=str, default=None, help="WandB run name (default: eval-{exp_name})")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name (auto-detected if not provided)")
    parser.add_argument("--reward", type=str, default=DEFAULT_REWARD_FN, help="Path to reward function file")
    parser.add_argument("--stats_only", action="store_true", help="Only save summary JSON, skip rollouts parquet")
    parser.add_argument("--keep_downloads", action="store_true", help="Keep downloaded GCS checkpoints after eval")
    parser.add_argument("--force-single", action="store_true", help="Treat input as single model even if directory")
    args = parser.parse_args()
    main(args)
