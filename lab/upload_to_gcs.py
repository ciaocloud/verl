#!/usr/bin/env python3
"""
Simple script to upload a local folder to GCS.
Reuse credentials and bucket logic from gcs_checkpoint.py.

Usage:
    python lab/upload_to_gcs.py /path/to/local/folder [gcs/destination/path]
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.cloud import storage

# Defaults from gcs_checkpoint.py
BUCKET_NAME = "gogo-verl-checkpoints"
DEFAULT_CREDENTIALS = "/wx-gcs-key.json"

# Set default credentials if file exists
if os.path.exists(DEFAULT_CREDENTIALS):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = DEFAULT_CREDENTIALS
    BUCKET_NAME = "wx-verl-checkpoints"

def upload_file(bucket, local_path: Path, blob_path: str):
    """Uploads a single file to GCS."""
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(local_path))
    # print(f"Uploaded {local_path} -> gs://{bucket.name}/{blob_path}")

def upload_folder(local_dir: str, gcs_path: str = None, bucket_name: str = BUCKET_NAME):
    """Uploads a folder to GCS."""
    local_path = Path(local_dir).resolve()
    if not local_path.is_dir():
        print(f"Error: {local_dir} is not a directory.")
        sys.exit(1)

    if gcs_path is None:
        gcs_path = local_path.name
    
    # Remove trailing slash from GCS path if present to avoid double slashes
    gcs_path = gcs_path.rstrip('/')

    client = storage.Client()
    try:
        bucket = client.bucket(bucket_name)
    except Exception as e:
        print(f"Error accessing bucket {bucket_name}: {e}")
        sys.exit(1)

    print(f"Uploading {local_path} to gs://{bucket_name}/{gcs_path} ...")

    files_to_upload = []
    for root, _, files in os.walk(local_path):
        for file in files:
            file_path = Path(root) / file
            # Calculate relative path from the local directory root
            rel_path = file_path.relative_to(local_path)
            # Construct GCS blob path
            blob_path = f"{gcs_path}/{rel_path}"
            files_to_upload.append((file_path, blob_path))

    total_files = len(files_to_upload)
    print(f"Found {total_files} files to upload.")

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(upload_file, bucket, fp, bp): (fp, bp) 
            for fp, bp in files_to_upload
        }
        
        completed = 0
        for future in as_completed(futures):
            try:
                future.result()
                completed += 1
                if completed % 10 == 0 or completed == total_files:
                    print(f"Progress: {completed}/{total_files} files uploaded.", end='\r')
            except Exception as e:
                fp, bp = futures[future]
                print(f"\nFailed to upload {fp}: {e}")

    print(f"\nUpload complete: gs://{bucket_name}/{gcs_path}")

def main():
    parser = argparse.ArgumentParser(description="Upload a local folder to GCS.")
    parser.add_argument("local_dir", help="Path to the local directory to upload.")
    parser.add_argument("gcs_path", nargs="?", help="Destination path in GCS (optional, defaults to directory name).")
    parser.add_argument("--bucket", default=BUCKET_NAME, help=f"GCS bucket name (default: {BUCKET_NAME})")

    args = parser.parse_args()

    upload_folder(args.local_dir, args.gcs_path, args.bucket)

if __name__ == "__main__":
    main()
