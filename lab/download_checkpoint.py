#!/usr/bin/env python3
"""
Download a model checkpoint from GCS to local.

Usage:
    python download_checkpoint.py --exp_name LASER-0.5B-gsm8k-020500 --step 10
    python download_checkpoint.py --exp_name LASER-0.5B-gsm8k-020500 --step 10 --local_dir ./checkpoints
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.cloud import storage

BUCKET_NAME = "gogo-verl-checkpoints"
DEFAULT_LOCAL_DIR = "checkpoints"
DEFAULT_CREDENTIALS = "/wx-gcs-key.json"

# Set default credentials if file exists
if os.path.exists(DEFAULT_CREDENTIALS):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = DEFAULT_CREDENTIALS
    BUCKET_NAME = "wx-verl-checkpoints"


def get_latest_step(exp_name: str, bucket_name: str) -> int | None:
    """Find the largest step number for an experiment on GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # List all blobs with the exp_name prefix
    prefix = f"{exp_name}/global_step_"
    blobs = bucket.list_blobs(prefix=prefix, delimiter='/')
    
    # Extract step numbers from prefixes
    steps = set()
    for page in blobs.pages:
        for prefix in page.prefixes:
            # prefix looks like "exp_name/global_step_10/"
            try:
                step_str = prefix.rstrip('/').split('_')[-1]
                steps.add(int(step_str))
            except (ValueError, IndexError):
                continue
    
    return max(steps) if steps else None


def download_checkpoint(exp_name: str, step: int, local_dir: str, bucket_name: str, project_name: str | None = None):
    """Download a checkpoint from GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    gcs_prefix = f"{exp_name}/global_step_{step}/"
    
    # Build local path with optional project_name
    if project_name:
        local_path = Path(local_dir) / project_name / exp_name / f"global_step_{step}"
    else:
        local_path = Path(local_dir) / exp_name / f"global_step_{step}"
    local_path.mkdir(parents=True, exist_ok=True)
    
    # List all blobs with the prefix
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    if not blobs:
        print(f"[ERROR] No checkpoint found at gs://{bucket_name}/{gcs_prefix}")
        return
    
    print(f"[GCS] Downloading {len(blobs)} files from gs://{bucket_name}/{gcs_prefix}")
    
    def download_blob(blob):
        # Get relative path from prefix
        rel_path = blob.name[len(gcs_prefix):]
        if not rel_path:  # Skip directory markers
            return None
        dest = local_path / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
        return dest
    
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download_blob, b): b for b in blobs}
        downloaded = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                downloaded += 1
                if downloaded % 10 == 0:
                    print(f"[GCS] Downloaded {downloaded}/{len(blobs)} files...")
    
    print(f"[GCS] Done! Checkpoint saved to: {local_path}")


def main():
    # Get defaults from environment
    default_project = os.environ.get("PROJ_NAME")
    default_exp = os.environ.get("EXP_NAME")
    
    p = argparse.ArgumentParser(description="Download checkpoint from GCS")
    p.add_argument("--project", default=default_project, help=f"Project name (default: $PROJ_NAME)")
    p.add_argument("--exp_name", default=default_exp, help=f"Experiment name (default: $EXP_NAME)")
    p.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    p.add_argument("--local_dir", default=DEFAULT_LOCAL_DIR, help=f"Local directory (default: {DEFAULT_LOCAL_DIR})")
    p.add_argument("--bucket", default=BUCKET_NAME, help=f"GCS bucket (default: {BUCKET_NAME})")
    args = p.parse_args()
    
    if not args.exp_name:
        p.error("--exp_name required (or set EXP_NAME env var)")
    
    step = args.step
    if step is None:
        print(f"[GCS] Finding latest step for {args.exp_name}...")
        step = get_latest_step(args.exp_name, args.bucket)
        if step is None:
            print(f"[ERROR] No checkpoints found for {args.exp_name}")
            return
        print(f"[GCS] Latest step: {step}")
    
    download_checkpoint(args.exp_name, step, args.local_dir, args.bucket, args.project)


if __name__ == "__main__":
    main()
