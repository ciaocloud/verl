# lab/gcp — Vertex AI training with runtime code pull

A self-contained launch setup for running verl RL jobs on Vertex AI (now branded
**Agent Platform**) CustomJob. Design: **bake dependencies into the image once,
pull the code at container startup.**

- **Code change** → `git push` + resubmit. No image rebuild.
- **Dependency change** (or new base image) → rebuild the image.

This works because the base `verlai/verl` image already ships verl's heavy deps,
and verl installs editable — so overwriting the checkout at startup needs no
reinstall of dependencies.

## Files

- `Dockerfile` — bakes deps (incl. TensorFlow/TensorBoard) + `bootstrap.sh`. Does **not** COPY the repo.
- `bootstrap.sh` — image entrypoint. Clones the fork at `GIT_REF`, `pip install -e`, execs `scripts/gcp/train.sh`.
- `build_and_push.sh` — build + push the runner image to Artifact Registry.
- `jobspec.example.yaml` — CustomJob spec template.
- `submit.sh` — thin `gcloud ai custom-jobs create --config=` wrapper.

Note: `bootstrap.sh` reuses the repo's existing `scripts/gcp/train.sh` (data
staging, checkpoint sync, Hydra arg assembly) rather than duplicating it.

## First-time GCP setup (fresh account)

1. Install tooling (macOS): `brew install --cask google-cloud-sdk docker`, then `open -a Docker`.
2. Auth: `gcloud auth login`.
3. Project + billing: `gcloud config set project PROJECT_ID`, `gcloud config set compute/region us-central1`, link a billing account.
4. Enable APIs: `gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com storage.googleapis.com compute.googleapis.com`.
5. **Request L4 GPU quota** (IAM & Admin → Quotas): "Custom model training Nvidia L4 GPUs per region" ≥ 1 in your region. Fresh accounts have 0; this gates the run and approval can lag.
6. Create a bucket: `gcloud storage buckets create gs://YOUR_BUCKET --location=us-central1`.

## Build the image (once, and on dep changes)

```sh
lab/gcp/build_and_push.sh --check          # verify docker/gcloud + resolved values
lab/gcp/build_and_push.sh --tag runner     # build linux/amd64, create repo, push
```

Produces `REGION-docker.pkg.dev/PROJECT_ID/verl-rl/verl-gcp-runner:runner`.
First build is slow (multi-GB base + amd64 emulation on Apple Silicon).

## Prepare data (once)

```sh
pip install datasets
python3 scripts/gcp/prepare_gsm8k_dataset.py --output-dir /tmp/gsm8k
gcloud storage cp /tmp/gsm8k/train.parquet gs://YOUR_BUCKET/datasets/gsm8k/train.parquet
gcloud storage cp /tmp/gsm8k/test.parquet  gs://YOUR_BUCKET/datasets/gsm8k/test.parquet
```

## Iterate (the fast loop)

```sh
# 1. edit code, then push (container only sees pushed commits)
git push origin lotis

# 2. once: copy + fill in the spec
cp lab/gcp/jobspec.example.yaml lab/gcp/jobspec.lotis-rbf.yaml
#    edit: imageUri, TRAIN_FILES/VAL_FILES/GCS_CHECKPOINT_URI, EXP_NAME

# 3. submit (repeat this per iteration)
lab/gcp/submit.sh lab/gcp/jobspec.lotis-rbf.yaml
```

Monitor:

```sh
gcloud ai custom-jobs list --region=us-central1 --limit=10
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

## Notes & gotchas

- **Push before submit.** `git clone` sees only pushed commits, not your working tree.
- **Reproducibility.** Pin `GIT_REF` to a commit SHA for runs that matter; the image tag no longer pins code.
- **Dependency changes** need an image rebuild (or set `SKIP_PIP_INSTALL=0`, which is the default, so `pip install -e` runs — but new *deps* still won't be installed since bootstrap uses the baked env). Rebuild is the reliable path.
- **Private repo.** `ciaocloud/verl` is public, so no credentials are needed. If you make it private, store a fine-grained PAT in Secret Manager and change `GIT_REMOTE` to `https://x-access-token:${GIT_TOKEN}@github.com/ciaocloud/verl.git` with `GIT_TOKEN` wired as a secret env var.
- **`us-central1`** is assumed throughout (L4 availability). Keep bucket, Artifact Registry, and the job in the same region to avoid egress cost.
