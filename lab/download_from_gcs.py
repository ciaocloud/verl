#!/usr/bin/env python3
"""
Simple script to download a folder from GCS to local.
Reuse credentials and bucket logic from gcs_checkpoint.py.

Usage:
    python lab/download_from_gcs.py gcs/source/path /path/to/local/destination
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.cloud import storage

# Defaults from gcs_checkpoint.py
BUCKET_NAME = "gogo-verl-checkpoints"
DEFAULT_CREDENTIALS = "/Users/wangxing/rlexp/wx-rl-playground.json"

# Set default credentials if file exists
if os.path.exists(DEFAULT_CREDENTIALS):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = DEFAULT_CREDENTIALS
    BUCKET_NAME = "wx-verl-checkpoints"

def download_file(bucket, blob, local_path: Path):
    """Downloads a single file from GCS."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    # print(f"Downloaded gs://{bucket.name}/{blob.name} -> {local_path}")

def download_folder(gcs_path: str, local_dir: str, bucket_name: str = BUCKET_NAME):
    """Downloads a folder from GCS."""
    client = storage.Client()
    try:
        bucket = client.bucket(bucket_name)
    except Exception as e:
        print(f"Error accessing bucket {bucket_name}: {e}")
        sys.exit(1)

    # Ensure GCS path doesn't start with /
    gcs_prefix = gcs_path.strip("/")
    if not gcs_prefix.endswith("/"):
         gcs_prefix += "/"

    print(f"Downloading gs://{bucket_name}/{gcs_prefix} to {local_dir} ...")

    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    
    if not blobs:
        print(f"No files found at gs://{bucket_name}/{gcs_prefix}")
        sys.exit(1)

    files_to_download = []
    local_root = Path(local_dir)

    for blob in blobs:
        if blob.name.endswith("/"):
            continue # Skip directories
        
        # Calculate relative path
        rel_path = blob.name[len(gcs_prefix):] 
        local_file_path = local_root / rel_path
        files_to_download.append((blob, local_file_path))

    total_files = len(files_to_download)
    print(f"Found {total_files} files to download.")

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(download_file, bucket, blob, lp): (blob.name, lp) 
            for blob, lp in files_to_download
        }
        
        completed = 0
        for future in as_completed(futures):
            try:
                future.result()
                completed += 1
                if completed % 10 == 0 or completed == total_files:
                    print(f"Progress: {completed}/{total_files} files downloaded.", end='\r')
            except Exception as e:
                name, lp = futures[future]
                print(f"\nFailed to download {name}: {e}")

    print(f"\nDownload complete: {local_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download a folder from GCS.")
    parser.add_argument("gcs_path", help="Source path in GCS (path/to/folder).")
    parser.add_argument("local_dir", help="Local destination directory.")
    parser.add_argument("--bucket", default=BUCKET_NAME, help=f"GCS bucket name (default: {BUCKET_NAME})")

    args = parser.parse_args()

    download_folder(args.gcs_path, args.local_dir, args.bucket)

if __name__ == "__main__":
    main()
