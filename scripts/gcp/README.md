# GCP serverless RL launch scripts

These scripts package this `verl` checkout into a reusable GPU training
container and submit GRPO or PPO jobs to managed GCP compute through Vertex AI
CustomJob.

## Files

- `Dockerfile`: builds a training image from a prebuilt `verlai/verl` image.
- `env.example`: configuration template.
- `build_and_push.sh`: creates an Artifact Registry repo if needed, builds, and pushes the image.
- `launch_vertex_experiment.sh`: production launcher that creates a run id, optionally builds a new image, resolves per-run GCS paths, and submits/dry-runs Vertex.
- `prepare_gsm8k_dataset.py`: creates GSM8K parquet files in verl's expected schema.
- `train.sh`: container entrypoint for configurable Qwen/GSM8K GRPO or PPO runs.
- `submit_vertex_custom_job.sh`: submits a Vertex AI CustomJob.
- `check_storage.sh`: verifies configured train/val objects and output buckets.
- `status.sh`: lists or describes Vertex AI CustomJobs.
- `tail_logs.sh`: streams Vertex AI CustomJob logs.

## Quick start

```sh
cp scripts/gcp/env.example scripts/gcp/.env
$EDITOR scripts/gcp/.env

scripts/gcp/build_and_push.sh
scripts/gcp/submit_vertex_custom_job.sh
```

To rebuild the reusable runner image after code or dependency changes:

```sh
scripts/gcp/build_and_push.sh --tag runner
```

For a production-style Vertex run with a unique run id and per-run checkpoint
paths using the configured runner image:

```sh
scripts/gcp/launch_vertex_experiment.sh
```

To review the generated Vertex jobSpec before spending GPU money:

```sh
scripts/gcp/launch_vertex_experiment.sh --dry-run --config-out /tmp/verl-vertex-job.yaml
```

Check jobs:

```sh
scripts/gcp/status.sh
scripts/gcp/status.sh CUSTOM_JOB_ID
```

Stream logs:

```sh
scripts/gcp/tail_logs.sh CUSTOM_JOB_ID
```

## Defaults

The default job uses:

- `Qwen/Qwen2.5-0.5B-Instruct`
- explicit train/validation parquet files from `TRAIN_FILES` and `VAL_FILES`
- `RL_ALGORITHM=grpo`, which maps to `algorithm.adv_estimator=grpo`
- 1 GPU
- 2 total training steps
- conservative single-L4 rollout settings:
  `ROLLOUT_AGENT_NUM_WORKERS=1`, `MAX_MODEL_LEN=1024`, `MAX_NUM_SEQS=16`
- validation capped by default with `VAL_MAX_SAMPLES=16` so smoke jobs do not
  spend most of their time evaluating the full validation set
- `DATALOADER_NUM_WORKERS=0` to avoid extra multiprocessing churn inside the
  Ray-managed Vertex container
- console logging
- local checkpoints under `/workspace/outputs/$EXP_NAME/checkpoints`
- periodic and final checkpoint upload to `GCS_CHECKPOINT_URI`

Required storage values:

```sh
TRAIN_FILES=gs://YOUR_BUCKET/datasets/gsm8k/train.parquet
VAL_FILES=gs://YOUR_BUCKET/datasets/gsm8k/test.parquet
GCS_CHECKPOINT_URI=gs://YOUR_BUCKET/checkpoints/verl-rl
```

Multiple train or validation parquet shards can be comma-separated.
These parquet files are expected to already exist. The training entrypoint only
stages them from object storage into the job container before invoking `verl`;
it does not generate datasets.

Verify storage before submitting a job:

```sh
scripts/gcp/check_storage.sh
```

Set `CHECKPOINT_SYNC_INTERVAL_SECONDS=0` to disable periodic checkpoint sync.
Set `GCS_OUTPUT_URI=gs://bucket/prefix` to also copy the full
`/workspace/outputs/$EXP_NAME` directory after training.

## Production launch flow

The normal loop for a new experiment is:

1. Change the `verl` repo code or training env settings.
2. Build and push a new immutable image tag.
3. Generate/review the Vertex jobSpec.
4. Submit the Vertex CustomJob and watch logs/checkpoints.

For a stable reusable image tag, build:

```sh
scripts/gcp/build_and_push.sh --tag runner
```

Then submit as many GRPO/PPO experiments as needed without rebuilding:

```sh
scripts/gcp/launch_vertex_experiment.sh
```

For an immutable image per run, the launcher can combine steps 2-4:

```sh
scripts/gcp/launch_vertex_experiment.sh --build
```

It derives:

- `RUN_ID=<timestamp>-<git-sha>`
- `IMAGE_TAG=$RUN_ID` when `--build` is used
- `EXP_NAME=<base-exp-name>-$RUN_ID`
- `GCS_CHECKPOINT_URI=gs://$GCS_BUCKET/checkpoints/$EXP_NAME`
- `GCS_OUTPUT_URI=gs://$GCS_BUCKET/outputs/$EXP_NAME`

Use `--reuse-env-paths` when you intentionally want the exact GCS paths from
`.env`.

For PPO-style runs, set:

```sh
RL_ALGORITHM=ppo
CRITIC_MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
```

Then tune the batch, critic, and epoch settings in `.env`. For epoch-based
runs, leave `TOTAL_TRAINING_STEPS=` empty and set `TOTAL_EPOCHS`.
For one-off Hydra overrides without changing the image, use semicolon-separated
settings:

```sh
EXTRA_HYDRA_ARGS='trainer.total_epochs=3;actor_rollout_ref.rollout.n=4'
```

## Notes

- The image build context is the repo root, so run scripts from anywhere inside
  the checkout or set `ENV_FILE=/path/to/.env`.
- The default image base is `verlai/verl:vllm012.latest`; override `BASE_IMAGE`
  if the project standardizes on another `verl` image.
- Vertex service agents need permission to pull from Artifact Registry, and
  the configured training service account needs read/write access to the GCS
  locations.
- The image is intended to be a reusable base runner. Choose GRPO/PPO and tune
  most experiment behavior with env vars or `EXTRA_HYDRA_ARGS`; rebuild the
  image only when repo code or dependencies change.
