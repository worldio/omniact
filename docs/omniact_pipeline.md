# OmniAct Training Pipeline

Generate data with [omniact-molmospaces](https://github.com/worldio/omniact-molmospaces), prepare it with the scripts in [wods](https://github.com/worldio/wods), then train and evaluate with this repository. Skip any stage your data has already been through.

## Setup

```bash
git clone https://github.com/worldio/omniact.git
git clone https://github.com/worldio/wods.git
git clone https://github.com/worldio/omniact-molmospaces.git

(cd omniact && uv sync)
(cd wods && uv sync)

cd omniact-molmospaces
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[mujoco]"
```

Set these in every shell you use. Data paths can be local directories or fsspec URLs such as `gs://...`.

```bash
DATA_ROOT=/path/to/embgen_ms_1m
WODS_ROOT=$DATA_ROOT/trajectories
LIBRARY_ROOT=$DATA_ROOT/library
REPLAY_ROOT=$DATA_ROOT/replay
TEXT_WODS_ROOT=$DATA_ROOT/trajectories_with_embeds
TRAIN_WODS_ROOT=$DATA_ROOT/trajectories_with_embeds_video_chunk16
CHECKPOINT_BASE=/path/to/checkpoints
EVAL_OUT=/path/to/eval_rollouts
```

## Stage 1: Data Generation

```bash
# in omniact-molmospaces, with .venv activated; one process per RANK in 0..WORLD_SIZE-1; rerun to resume
RANK=${RANK:-0} WORLD_SIZE=${WORLD_SIZE:-1} python scripts/embodiments/generate_embodiment_dataset.py \
    --seeds 0 2000000000 \
    --library "$LIBRARY_ROOT" \
    --trajectories "$WODS_ROOT" \
    --markers "$DATA_ROOT/markers" \
    --replay-out "$REPLAY_ROOT"
```

Details: [docs/embodiment_generation.md](https://github.com/worldio/omniact-molmospaces/blob/master/docs/embodiment_generation.md) in omniact-molmospaces.

## Stage 2: Text Embeddings

```bash
# in wods; one process per RANK in 0..WORLD_SIZE-1
RANK=${RANK:-0} WORLD_SIZE=${WORLD_SIZE:-1} uv run python scripts/add_text_embed_qwen.py \
    --input "$WODS_ROOT" --output "$TEXT_WODS_ROOT" \
    --group-final-components 0 \
    --device cpu --torch-dtype bfloat16 --storage-dtype float32 --num-workers 8

# once, after all ranks finish
uv run python scripts/write_key_list.py "$TEXT_WODS_ROOT"
```

## Stage 3: Video Chunks

```bash
# in wods; one process per RANK in 0..WORLD_SIZE-1
RANK=${RANK:-0} WORLD_SIZE=${WORLD_SIZE:-1} uv run python scripts/split_wods_to_chunks.py \
    "$TEXT_WODS_ROOT" "$TRAIN_WODS_ROOT" \
    --chunk-frames 16 --crf 16 --key-workers 8

# once, after all ranks finish
uv run python scripts/write_key_list.py "$TRAIN_WODS_ROOT"
```

## Stage 4: Training

```bash
# in omniact; 8 GPUs x --batch-size 16 = effective batch size 128
uv run torchrun --standalone --nproc-per-node=8 scripts/train_omniact.py \
    --data "$TRAIN_WODS_ROOT" \
    --model-size 300M --window-size 32 --chunk-frames 16 \
    --batch-size 16 --num-workers 8 --lr 1e-3 --warmup-steps 500 \
    --max-steps 1000000 --checkpoint-interval 2000 \
    --subsample-train-embodiments 0 \
    --checkpoint-base "$CHECKPOINT_BASE" \
    --run-name omniact-300m-g8 --seed 0 --split-seed 42
```

Rerunning with the same `--run-name` resumes from its last checkpoint. `--split-seed` holds out embodiments and scenes for evaluation. Training logs to Weights & Biases; add `--no-wandb` to turn that off.

## Stage 5: Closed-Loop Evaluation

```bash
# in omniact-molmospaces, with .venv activated
python -m molmo_spaces.rollout.eval_server --port 50051

# in omniact, from a second shell
uv run python scripts/eval_omniact.py \
    --target localhost:50051 \
    --checkpoint "$CHECKPOINT_BASE/omniact-300m-g8/step_500000.pt" \
    --data "$TRAIN_WODS_ROOT" --replay-root "$REPLAY_ROOT" \
    --split-seed 42 --n-keys 50 --command pose \
    --out "$EVAL_OUT/omniact-300m-g8"
```

`--split-seed` must match training. `--split` picks `seen`, `unseen_embodiment`, `unseen_scene`, `unseen_embodiment_and_scene`, or `all` (the default); `--command` picks `joint` (the default), `pose`, or `hybrid`. Rerunning skips rollouts already saved under `--out`.

On several GPUs, start one server per GPU, each on its own port and pinned to its GPU with `CUDA_VISIBLE_DEVICES` and `MUJOCO_EGL_DEVICE_ID`, then run one client per server; the clients split the episodes by rank:

```bash
# in omniact: client i of N, connected to the server on port 50051+i
RANK=$i WORLD_SIZE=$N LOCAL_RANK=$i uv run python scripts/eval_omniact.py \
    --target localhost:$((50051 + i)) \
    --checkpoint "$CHECKPOINT_BASE/omniact-300m-g8/step_500000.pt" \
    --data "$TRAIN_WODS_ROOT" --replay-root "$REPLAY_ROOT" \
    --split-seed 42 --n-keys 50 --command pose \
    --out "$EVAL_OUT/omniact-300m-g8"

# once, after all clients finish: aggregate the success rates
uv run python scripts/eval_omniact.py \
    --target localhost:50051 \
    --checkpoint "$CHECKPOINT_BASE/omniact-300m-g8/step_500000.pt" \
    --out "$EVAL_OUT/omniact-300m-g8" --finalize
```
