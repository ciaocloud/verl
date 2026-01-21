#!/usr/bin/env python3
"""
Async GCS checkpoint uploader. Watches /dev/shm for checkpoints, uploads to GCS, deletes local.

Usage:
    export EXP_NAME="GRPO-0.5B-0121"
    python gcs_checkpoint.py                    # Uses defaults from env
    python gcs_checkpoint.py --bucket my-bucket # Override bucket
"""

import argparse
import atexit
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.cloud import storage

BUCKET_NAME = "gogo-verl-checkpoints"
# BUCKET_NAME = "wx-verl-checkpoints"
DEFAULT_WATCH_DIR = "/dev/shm/verl_ckpt"
# DEFAULT_CREDENTIALS = "/wx-gcs-key.json"

# # Set default credentials if not in env
# if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and os.path.exists(DEFAULT_CREDENTIALS):
#     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = DEFAULT_CREDENTIALS


class GCSWatcher:
    def __init__(self, local_dir: str, bucket: str):
        self.local_dir = Path(local_dir)
        self.exp_name = self.local_dir.name
        self.uploaded: set[str] = set()
        self._stop = threading.Event()
        self.bucket_name = bucket
        
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gcs")
        self.pending: dict[int, threading.Event] = {}
        self._lock = threading.Lock()
        
        atexit.register(self._wait_all)
        print(f"[GCS] Watching: {local_dir} -> gs://{bucket}/{self.exp_name}/", flush=True)

    def _upload(self, ckpt_dir: Path, step: int, event: threading.Event):
        """Upload checkpoint directory to GCS, then delete local."""
        try:
            files = [f for f in ckpt_dir.rglob("*") if f.is_file()]
            prefix = f"{self.exp_name}/{ckpt_dir.name}/"
            
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {
                    pool.submit(self._upload_file, f, f"{prefix}{f.relative_to(ckpt_dir)}"): f
                    for f in files
                }
                for future in as_completed(futures):
                    future.result()
            
            print(f"[GCS] Uploaded step {step}: {len(files)} files -> gs://{self.bucket_name}/{prefix}", flush=True)
            shutil.rmtree(ckpt_dir)
            print(f"[GCS] Deleted: {ckpt_dir}", flush=True)
        except Exception as e:
            print(f"[GCS] Failed step {step}: {e}", flush=True)
        finally:
            with self._lock:
                self.pending.pop(step, None)
            event.set()

    def _upload_file(self, local: Path, gcs_path: str):
        self.bucket.blob(gcs_path).upload_from_filename(str(local))

    def _scan(self):
        if not self.local_dir.exists():
            return
        for item in sorted(self.local_dir.iterdir()):
            if not item.is_dir() or not item.name.startswith("global_step_"):
                continue
            if item.name in self.uploaded:
                continue
            if not (item / "actor").exists():
                continue
            
            step = int(item.name.split("_")[-1])
            self.uploaded.add(item.name)
            
            event = threading.Event()
            with self._lock:
                self.pending[step] = event
            self.executor.submit(self._upload, item, step, event)

    def _wait_all(self):
        with self._lock:
            events = list(self.pending.values())
        if events:
            print(f"[GCS] Waiting for {len(events)} uploads...", flush=True)
        for e in events:
            e.wait(7200)

    def run(self, interval: float = 30):
        while not self._stop.is_set():
            self._scan()
            self._stop.wait(interval)
        self._scan()
        self._wait_all()
        print("[GCS] Done", flush=True)

    def stop(self):
        self._stop.set()


def main():
    exp_name = os.environ.get("EXP_NAME")
    default_watch = f"{DEFAULT_WATCH_DIR}/{exp_name}" if exp_name else None
    
    p = argparse.ArgumentParser()
    p.add_argument("--watch", default=default_watch, 
                   help=f"Local checkpoint dir (default: {DEFAULT_WATCH_DIR}/$EXP_NAME)")
    p.add_argument("--bucket", default=BUCKET_NAME, help=f"GCS bucket (default: {BUCKET_NAME})")
    p.add_argument("--interval", type=float, default=30, help="Poll interval seconds")
    args = p.parse_args()

    if not args.watch:
        p.error("--watch required (or set EXP_NAME env var)")

    watcher = GCSWatcher(args.watch, args.bucket)
    try:
        watcher.run(args.interval)
    except KeyboardInterrupt:
        print("\n[GCS] Stopping...", flush=True)
        watcher.stop()


if __name__ == "__main__":
    main()
