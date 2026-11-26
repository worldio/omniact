<p align="center">
  <img src="assets/banner.jpg" alt="OmniAct: One Policy, One Million Embodiments" width="100%">
</p>

<p align="center">
  Xiang Fan<sup>1*</sup> · Yuheng Wang<sup>1*</sup> · Jiayuan Yan<sup>1</sup> · Bohan Fang<sup>1</sup> · Wisdom Oluchi Ikezogwo<sup>1</sup> · Ainaz Eftekhar<sup>1</sup>
  <br>
  Max Argus<sup>3</sup> · Zhongzheng Ren<sup>1,2</sup> · Jiafei Duan<sup>4</sup> · Ranjay Krishna<sup>1</sup>
  <br>
  <sup>1</sup>University of Washington · <sup>2</sup>University of North Carolina at Chapel Hill
  <br>
  <sup>3</sup>Allen Institute for Artificial Intelligence · <sup>4</sup>National University of Singapore
  <br>
  <sup>*</sup>Equal contribution
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.33158"><b>Paper</b></a> &nbsp;·&nbsp;
  <a href="https://omniact.github.io/"><b>Project page</b></a> &nbsp;·&nbsp;
  <a href="https://omniact.github.io/#real-world"><b>Videos</b></a> &nbsp;·&nbsp;
  <a href="#citation"><b>Citation</b></a>
</p>

<p align="center">
  OmniAct <b>zero-shot controls unseen robots</b> on a variety of tasks with surprisingly good robustness. It takes advantage of <b>robot-specific features</b> to maximize its task success.
</p>

## Overview

OmniAct is a cross-embodiment manipulation policy that conditions directly on each robot's kinematics in the scene. Each joint and fingertip of the robot becomes a node of a **scene-grounded kinematic graph**, placed in the 3D frame of every camera that observes it, and a flow-matching Transformer predicts a relative motion for every node. One network therefore controls arms with different numbers of joints and kinematic structures. To learn how a robot's body shapes its behavior, we procedurally generate **one million** robot embodiments, collect successful simulated trajectories for them, and train a single policy on the resulting data.

<p align="center">
  <img src="assets/overview.jpg" alt="Overview: generated 6-, 7- and 8-DoF embodiments and grippers, their successful simulated trajectories, the OmniAct policy, and its results in simulation and on real robots" width="100%">
  <br>
  <sub>(1) Generate diverse embodiments procedurally. (2) Keep their successful simulated trajectories. (3) Train a single policy, OmniAct. (4, 5) Evaluate it zero-shot in simulation and on real robots.</sub>
</p>

## Results

| Setting | OmniAct | Strongest baseline |
|---|:---:|:---:|
| **OmniAct-Bench** (sim): unseen robots in unseen scenes | **63.0%** | 57.9%&nbsp;(UMI,&nbsp;all&nbsp;views) |
| **MolmoSpaces** (sim): a Franka Panda, seen by every baseline but not OmniAct | **69.1%** | 63.0%&nbsp;(MolmoBot) |
| **Real OpenYAM**: arm and gripper zero-shot for OmniAct | **13/20** | 7/20&nbsp;(MolmoAct2) |
| **Real Franka Panda**: arm zero-shot for OmniAct | 14/20 | 15/20&nbsp;(DreamZero) |

## Training

The full pipeline, from generating data with [omniact-molmospaces](https://github.com/worldio/omniact-molmospaces) and preparing it with [wods](https://github.com/worldio/wods) to training and evaluating a policy, is in [docs/omniact_pipeline.md](docs/omniact_pipeline.md).

## Model Checkpoints

The policy trained for 520K steps is on Hugging Face at [xiangfan00/omniact](https://huggingface.co/xiangfan00/omniact):

```bash
CKPT=$(uvx --from huggingface_hub hf download xiangfan00/omniact -q)/omniact_520k.pt
```

Use `--checkpoint "$CKPT"` in the commands below.

## Evaluation

Evaluation runs closed-loop in simulation: an evaluation server from [worldio/omniact-molmospaces](https://github.com/worldio/omniact-molmospaces) rebuilds each episode and steps the physics, and `scripts/eval_omniact.py` runs the policy against it over gRPC. You need a trained checkpoint, the data root the policy was trained on (to select episodes), and the dataset's replay bundles (to rebuild them).

1. Clone this repository and omniact-molmospaces:

   ```bash
   git clone https://github.com/worldio/omniact.git
   git clone https://github.com/worldio/omniact-molmospaces.git
   ```

2. Set up omniact-molmospaces and start the evaluation server:

   ```bash
   cd omniact-molmospaces
   uv venv --python 3.11 .venv
   source .venv/bin/activate
   uv pip install -e ".[mujoco]"
   python -m molmo_spaces.rollout.eval_server --port 50051
   ```

3. In a second shell, set up this repository and run the evaluation:

   ```bash
   cd omniact
   uv sync
   uv run python scripts/eval_omniact.py \
       --target localhost:50051 \
       --checkpoint "$CKPT" \
       --data /path/to/training_data \
       --replay-root /path/to/replay \
       --split unseen_embodiment_and_scene --split-seed 42 --n-keys 50
   ```

`--split` picks the episodes to evaluate: `seen`, `unseen_embodiment`, `unseen_scene`, `unseen_embodiment_and_scene` (new robots in new scenes, the OmniAct-Bench setting), or `all` (the default). `--split-seed` must equal the training run's, or seen and unseen episodes get mixed up. Paths can be local directories or fsspec URLs such as `gs://`. The script prints the success rate per split; add `--out <dir>` to also record every rollout. See [Stage 5 of the pipeline doc](docs/omniact_pipeline.md#stage-5-closed-loop-evaluation) for the other options and for running on several GPUs.

## Real-World Deployment

A real robot runs behind a server that speaks the simulator's Rollout protocol, so `scripts/eval_omniact.py` drives it as it drives the simulator, with the task given by `--prompt`. Each robot's server is in its own repository, along with its setup, camera calibration and serve commands:

| Robot | Server |
|---|---|
| DROID: Franka Research 3, Robotiq 2F-85, ZED cameras | [worldio/droid_deploy](https://github.com/worldio/droid_deploy) |
| Anvil OpenYAM | [worldio/openyam_deploy](https://github.com/worldio/openyam_deploy) |

1. On the robot's computer, set up the server, calibrate the cameras and start the server, following that repository's README.

2. On the GPU machine, in a checkout of this repository, run the policy against it:

   ```bash
   uv sync
   uv run python scripts/eval_omniact.py \
       --target <server-ip>:50051 \
       --checkpoint "$CKPT" \
       --prompt "pick up green cube and place in white plate" \
       --command pose --max-steps 300
   ```

Each run is one episode, from the robot's start pose until `--max-steps`. `--command` picks the head that drives the arm: `joint` (the default), `pose` or `hybrid`. In the client's terminal, `p` pauses, `c` continues and `r` restarts the episode.

## Citation

```bibtex
@misc{fan2026omniactpolicymillionembodiments,
      title={OmniAct: One Policy, One Million Embodiments},
      author={Xiang Fan and Yuheng Wang and Jiayuan Yan and Bohan Fang and Wisdom Oluchi Ikezogwo and Ainaz Eftekhar and Max Argus and Zhongzheng Ren and Jiafei Duan and Ranjay Krishna},
      year={2026},
      eprint={2611.33158},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2611.33158},
}
```
