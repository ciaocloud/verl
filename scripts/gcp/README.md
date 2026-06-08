# GCP serverless RL launch scripts

These scripts package this `verl` checkout into a reusable GPU training
container and submit GRPO or PPO jobs to managed GCP compute through Vertex AI
CustomJob.

## Files

- `Dockerfile`: builds a training image from a prebuilt `verlai/verl` image.
- `env.example`: configuration template.
- `build_and_push.sh`: creates an Artifact Registry repo if needed, builds, and pushes the image.
- `launch_vertex_experiment.sh`: production launcher that creates a run id, optionally builds a new image, resolves per-run GCS paths, creates or reuses Vertex TensorBoard, prints the per-run TensorBoard URL, and submits/dry-runs Vertex.
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
- console, TensorBoard, and JSONL file logging
- local checkpoints under `/workspace/outputs/$EXP_NAME/checkpoints`
- periodic and final checkpoint upload to `GCS_CHECKPOINT_URI`
- local metrics under `/workspace/outputs/$EXP_NAME/metrics`
- live upload to managed Vertex AI TensorBoard when enabled
- periodic and final metrics upload to `GCS_METRICS_URI` when configured

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
Set `METRICS_SYNC_INTERVAL_SECONDS=0` to disable periodic metrics sync.
Set `GCS_METRICS_URI=gs://bucket/prefix` to copy TensorBoard event files and
JSONL training metrics during and after training.
Set `GCS_OUTPUT_URI=gs://bucket/prefix` to also copy the full
`/workspace/outputs/$EXP_NAME` directory after training.

## Metrics and visualization

The Vertex entrypoint configures verl with:

- `trainer.logger=["console","tensorboard","file"]`
- `TENSORBOARD_DIR=/workspace/outputs/$EXP_NAME/metrics/tensorboard`
- `VERL_FILE_LOGGER_PATH=/workspace/outputs/$EXP_NAME/metrics/metrics.jsonl`

For PPO/GRPO jobs, verl logs training metrics once per training step. Validation
metrics are included on validation steps and at step 0 when validation-before-
train is enabled.

For managed visualization, use the production launcher:

```sh
scripts/gcp/launch_vertex_experiment.sh
```

By default it creates or reuses a Vertex AI TensorBoard named
`$VERTEX_TENSORBOARD_DISPLAY_NAME` (`verl-rl` by default), pre-creates a per-run
TensorBoard experiment named `$EXP_NAME` before submitting the CustomJob, and
prints a URL like:

```text
TensorBoard: https://REGION.tensorboard.googleusercontent.com/experiment/...
```

Open that URL immediately after submission. It should resolve before the worker
starts, though scalar charts appear only after training writes TensorBoard event
data and the background uploader sends it. During training, the container runs
`tb-gcp-uploader` against `$TENSORBOARD_DIR`, so scalar metrics arrive in
managed TensorBoard while the job is still running. The launcher also forces the
runtime logger to include TensorBoard when a managed TensorBoard resource is
configured, which prevents older console-only env files from producing a blank
board.

After training exits, the entrypoint runs one final one-shot TensorBoard upload
to flush any last scalar events. `TENSORBOARD_FINAL_UPLOAD_TIMEOUT_SECONDS`
defaults to `120` so a stuck uploader cannot keep the Vertex job alive forever.

To pin a specific managed TensorBoard instead of using display-name lookup,
set:

```sh
VERTEX_TENSORBOARD_RESOURCE_NAME=projects/PROJECT_ID/locations/REGION/tensorboards/TENSORBOARD_ID
```

The same TensorBoard files are still synced to GCS, so local visualization
remains available:

```sh
gcloud storage cp -r gs://YOUR_BUCKET/metrics/$EXP_NAME/tensorboard ./tb/$EXP_NAME
tensorboard --logdir ./tb
```

The Vertex training service account must have permission to read and write the
configured TensorBoard resource. If the in-container uploader logs
`aiplatform.tensorboards.get` permission errors, grant the service account an
appropriate Vertex AI role, for example project-level `roles/aiplatform.user`,
or an equivalent custom role scoped by your IAM policy.

A least-privilege custom role for the uploader should include at least:

```text
aiplatform.tensorboards.get
aiplatform.tensorboards.list
aiplatform.tensorboards.recordAccess
aiplatform.tensorboardExperiments.create
aiplatform.tensorboardExperiments.get
aiplatform.tensorboardExperiments.list
aiplatform.tensorboardExperiments.update
aiplatform.tensorboardExperiments.write
aiplatform.tensorboardRuns.batchCreate
aiplatform.tensorboardRuns.create
aiplatform.tensorboardRuns.get
aiplatform.tensorboardRuns.list
aiplatform.tensorboardRuns.update
aiplatform.tensorboardRuns.write
aiplatform.tensorboardTimeSeries.batchCreate
aiplatform.tensorboardTimeSeries.batchRead
aiplatform.tensorboardTimeSeries.create
aiplatform.tensorboardTimeSeries.get
aiplatform.tensorboardTimeSeries.list
aiplatform.tensorboardTimeSeries.read
aiplatform.tensorboardTimeSeries.update
aiplatform.metadataStores.create
aiplatform.metadataStores.get
aiplatform.artifacts.create
aiplatform.artifacts.get
aiplatform.artifacts.list
aiplatform.artifacts.update
aiplatform.contexts.addContextArtifactsAndExecutions
aiplatform.contexts.addContextChildren
aiplatform.contexts.create
aiplatform.contexts.get
aiplatform.contexts.list
aiplatform.contexts.queryContextLineageSubgraph
aiplatform.contexts.update
aiplatform.executions.addExecutionEvents
aiplatform.executions.create
aiplatform.executions.get
aiplatform.executions.list
aiplatform.executions.queryExecutionInputsAndOutputs
aiplatform.executions.update
aiplatform.metadataStores.list
resourcemanager.projects.get
```

The launcher checks the training service account before real submission. It
allows `roles/aiplatform.user`, `roles/aiplatform.admin`, owner/editor, or a
custom role containing the permissions above. If your organization grants an
equivalent permission path that `gcloud iam roles describe` cannot inspect, set
`SKIP_TENSORBOARD_IAM_CHECK=1`.

To create and grant the least-privilege custom role after approval:

```sh
scripts/gcp/setup_tensorboard_iam.sh --apply
```

Run it without `--apply` to print the resolved project, service account, role,
and permissions without changing IAM.

Raw per-step metrics are also available as JSONL:

```sh
gcloud storage cp gs://YOUR_BUCKET/metrics/$EXP_NAME/metrics.jsonl .
```

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
- `GCS_METRICS_URI=gs://$GCS_BUCKET/metrics/$EXP_NAME`
- `GCS_OUTPUT_URI=gs://$GCS_BUCKET/outputs/$EXP_NAME`
- `VERTEX_TENSORBOARD_EXPERIMENT_NAME=$EXP_NAME`
- `VERTEX_TENSORBOARD_EXPERIMENT_URL=<managed TensorBoard URL for this run>`
- `TENSORBOARD_FINAL_UPLOAD_TIMEOUT_SECONDS=120` unless overridden

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
