#!/usr/bin/env python
"""Closed-loop OmniAct eval: drive an omniact-molmospaces ``eval_server`` with a trained ``OmniActModel``.

Loads a checkpoint, connects to a running ``eval_server``, and for each episode runs the receding-horizon
``PolicyAdapter`` until the env reports ``done`` — the policy predicts a chunk, denorms both heads, and decodes
them into the server's contract (per-step relative joint action + ABSOLUTE TCP pose for server IK). Reports the
per-episode success and the aggregate success rate. This is the MODEL counterpart of ``replay_check.py`` (which
drives the same server with ground-truth actions to check reconstruction fidelity).

Splits
------
By default the eval runs each of the four seen/unseen SPLITS the training uses, selecting keys with the SAME
seeded axis split (``omniact.split.split_train_val_keys``): ``seen`` (the training keys), and the three
held-out categories ``unseen_embodiment`` / ``unseen_scene`` / ``unseen_embodiment_and_scene``. Pass
``--split`` to run one category and ``--n-keys`` / ``--sample-seed`` to bound the count. ``--split-seed`` MUST
match the training run's ``--split-seed``, or "seen"/"unseen" are meaningless. ``--wods-key`` overrides split
selection and evaluates a single key.

Usage
-----
Start the server (omniact-molmospaces environment), then run this (this repository's environment)::

    # omniact-molmospaces:
    python -m molmo_spaces.rollout.eval_server --port 50051
    # omniact:
    python scripts/eval_omniact.py --target localhost:50051 \
        --checkpoint <checkpoint-base>/<run-name>/step_050000.pt \
        --data <WODS root> --replay-root <replay bundle root> \
        --split-seed 42 --n-keys 50 [--split unseen_scene] [--action-horizon 8] [--out <root>]

``--data`` (the WODS root) enumerates the key universe for the split; the same root is used as ``--wods-root``
for the bundle caption unless overridden. ``--out`` records each episode's rollout to WODS.

Keys (an interactive terminal, one process): ``r`` ends the episode, saves its rollout (tagged STOPPED, never a
result) and runs the task again in a new session; ``p`` pauses (no steps are sent, so the robot holds, and no new
session starts, since a Reset moves the arm); ``c`` continues, re-planning from the newest observation.
"""
from __future__ import annotations

import argparse
import os

from omniact.split import VALIDATION_SPLITS

SEEN = "seen"
SPLITS = (SEEN, *VALIDATION_SPLITS)                     # the four seen/unseen splits


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True, help="eval_server address, host:port")
    p.add_argument("--checkpoint", required=True, help="trained OmniAct checkpoint .pt (local path or fsspec URL)")
    p.add_argument("--replay-root", default=os.environ.get("REPLAY_ROOT"),
                   help="replay bundle ROOT (any fsspec target, or $REPLAY_ROOT); bundles resolved house-aware")
    p.add_argument("--wods-root", default=os.environ.get("WODS_ROOT"),
                   help="WODS root for each bundle's authoritative caption (default: --data)")
    # Split selection (the key universe to evaluate)
    p.add_argument("--data", default=os.environ.get("WODS_ROOT"),
                   help="WODS root to enumerate the key universe for the split (any fsspec target; or $WODS_ROOT)")
    p.add_argument("--split", choices=(*SPLITS, "all"), default="all",
                   help="which seen/unseen split to evaluate (default: all four)")
    p.add_argument("--split-seed", type=int, default=42,
                   help="seed for the seen/unseen axis split — MUST match train_omniact --split-seed")
    p.add_argument("--trained-tasks", nargs="+", default=None,
                   help="task-name prefixes that defined the training split universe (default: all)")
    p.add_argument("--eval-tasks", nargs="+", default=None,
                   help="restrict the selected split keys to these task-name prefixes (default: no filter)")
    p.add_argument("--n-keys", type=int, default=0, help="max keys to eval per split (0 = all)")
    p.add_argument("--sample-seed", type=int, default=None,
                   help="if set, randomly sample the --n-keys keys (deterministic) instead of the first n")
    p.add_argument("--wods-key", nargs="+", default=None, help="evaluate ONLY these keys (overrides --split/--data)")
    # Per-episode
    p.add_argument("--episode-idx", type=int, default=0, help="episode within each resolved bundle")
    p.add_argument("--cameras", default=None, help="';'-separated camera subset (server Open option)")
    p.add_argument("--prompt", default=None,
                   help="override the bundle caption with this instruction (encoded live via Qwen; "
                        "verified to match the stored embeddings to bf16 rounding, cos 0.9999+)")
    p.add_argument("--seed", type=int, default=0,
                   help="base for the per-key RNG seed, so a rerun samples the same policy noise and two "
                        "runs are comparable key by key")
    p.add_argument("--max-steps", type=int, default=None,
                   help="steps allowed per episode (default: the task config's own cap, 500 for pick). "
                        "Raises the SERVER task horizon as well as the client loop: the task truncates "
                        "the episode itself, so capping only the client cannot lengthen one")
    p.add_argument("--finalize", action="store_true",
                   help="aggregate the result.json files already under --out and exit; run after "
                        "all ranks finish, with the same --out")
    # Model / policy
    p.add_argument("--device", default="cuda")
    p.add_argument("--action-horizon", type=int, default=8, help="steps executed per re-plan (receding horizon)")
    p.add_argument("--command", choices=("joint", "pose", "hybrid"), default="joint",
                   help="which head to drive: joint (joint head drives all actuated dims), pose (TCP-pose IK for "
                        "the arm + gripper via the joint head), or hybrid (whole joint head + TCP pose as IK ref)")
    p.add_argument("--denoise-steps", type=int, default=None, help="override the checkpoint's flow denoise steps")
    p.add_argument("--image-size", type=int, nargs=2, default=[256, 256], help="model input crop-resize (H W)")
    # Optional rollout recording
    p.add_argument("--out", help="if set, record each episode's rollout to WODS under this root")
    p.add_argument("--fps", type=float, default=30.0, help="fps for the recorded rollout (--out)")
    return p.parse_args(argv)


def select_split_keys(args) -> dict[str, list[str]]:
    """``{split_name: [keys]}`` for the requested split(s), from the SAME seeded axis split training used.
    Applies the ``--eval-tasks`` filter, then samples/truncates to ``--n-keys``."""
    import random

    from wods.v2.item import list_wods_keys

    from omniact.split import _split_key, split_train_val_keys

    if not args.data:
        raise SystemExit("--data (WODS root) is required to enumerate the split keys (or pass --wods-key)")
    split = split_train_val_keys(list_wods_keys(args.data), args.split_seed, tasks=args.trained_tasks)
    key_sets = {SEEN: set(split.train_keys), **{k: set(v) for k, v in split.validation_keys.items()}}
    if args.eval_tasks:
        prefixes = tuple(args.eval_tasks)                 # task = the config-name segment (split_train_val_keys uses it too)
        key_sets = {name: {k for k in keys if _split_key(k)[2].startswith(prefixes)}
                    for name, keys in key_sets.items()}

    chosen = list(SPLITS) if args.split == "all" else [args.split]
    out: dict[str, list[str]] = {}
    for name in chosen:
        keys = sorted(key_sets[name])
        if args.sample_seed is not None:
            random.Random(args.sample_seed).shuffle(keys)
        out[name] = keys[: args.n_keys] if args.n_keys > 0 else keys
    return out


def wods_text_embed(wods_root: str, key: str):
    """Read this key's stored Qwen conditioning from one metadata/view shard, or None if absent.

    Prefer full-trajectory meta/ shards in chunked roots; unchunked roots have bare camera shards.
    Reading only text avoids loading videos or geometry just to prime the policy's prompt cache.
    """
    from wods.v2.item import list_wods_shards, read_wods_shard_batched

    if not wods_root:
        return None
    shards = sorted(list_wods_shards(wods_root, key))
    meta = [s for s in shards if s.startswith("meta/")] or shards
    if not meta:
        return None
    off = {f"include_{name}": False for name in (
        "video", "video_raw", "timestamps", "latents", "camera", "points", "depth", "depth_raw", "mask",
        "proprio", "action", "target", "node_geometry", "node_action_geometry", "node_graph", "node_proprio",
        "node_pose", "node_action_pose")}
    return read_wods_shard_batched([wods_root], [key], meta[:1], include_text_embed=True, **off)[0].get("text_embed")


def override_prompt(obs, prompt):
    """Force ``prompt`` to be the policy's instruction for this observation.

    _text_embed prefers obs["text_embed"] and only falls back to obs["prompt"], so replacing the text
    without dropping the embedding would leave the original instruction in force.
    """
    if prompt and obs is not None:
        obs.pop("text_embed", None)
        obs["prompt"] = prompt
    return obs


class Hotkeys:
    """Single keys from the terminal, read on a thread while the eval runs (no Enter; Ctrl+C still interrupts):
    ``r`` sets ``restart``, ``p`` sets ``paused``, ``c`` clears it. The terminal is restored at exit."""

    LEGEND = "keys: r = end this episode (saved) and run the task again, p = pause, c = continue"

    def __init__(self):
        import atexit
        import sys
        import termios
        import threading
        import tty

        self.fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        atexit.register(self.close)
        self.restart, self.paused = threading.Event(), threading.Event()
        threading.Thread(target=self._read, daemon=True).start()

    @classmethod
    def start(cls):
        """Hotkeys on an interactive terminal, else None."""
        import sys

        return cls() if sys.stdin.isatty() else None

    def _read(self):
        import os

        while True:
            key = os.read(self.fd, 1).decode(errors="ignore").lower()
            if key == "r" and not self.restart.is_set():
                self.restart.set()
                print("\n[r] ending this episode: it is saved, then the task runs again"
                      + (" once you continue (c)" if self.paused.is_set() else ""), flush=True)
            elif key == "p" and not self.paused.is_set():
                self.paused.set()
                print("\n[p] paused: no steps are sent and the robot holds; c continues", flush=True)
            elif key == "c" and self.paused.is_set():
                self.paused.clear()
                print("\n[c] continuing", flush=True)

    def hold(self, until_restart=True) -> bool:
        """Wait while paused, until ``c`` (or, with ``until_restart``, ``r``). Returns whether it waited."""
        import time

        waited = False
        while self.paused.is_set() and not (until_restart and self.restart.is_set()):
            waited = True
            time.sleep(0.05)
        return waited

    def close(self):
        import termios

        termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)


def stopped_key(split: str, wods_key: str) -> str:
    """WODS key of a rollout the operator ended (``r``): tagged STOPPED, neither SUCCESS nor FAIL, so resuming and
    :func:`finalize` never take it for a result; the time tells the tries apart."""
    import time

    return f"{split}/{wods_key.replace('/', '_')}_STOPPED_{time.strftime('%Y%m%d-%H%M%S')}"


def run_episode(session, policy, obs, args, record, hotkeys=None):
    """Drive one episode to ``done`` (or ``--max-steps``, or the operator's ``r``). Returns ``(success, steps, frames,
    sent, stopped)``, ``stopped`` True when the operator ended it."""
    import numpy as np

    from omniact.rollout import last_tick

    policy.reset()
    override_prompt(obs, args.prompt)
    frames, sent = ([obs], []) if record else (None, None)   # frame i is the obs BEFORE sent[i]
    done, info, steps = False, {}, 0
    while not done:
        if hotkeys is not None:
            if hotkeys.hold():
                policy.reset()           # re-plan from the newest observation, not a chunk from before the pause
            if hotkeys.restart.is_set():
                return False, steps, frames, sent, True
        cmd = policy.act(obs)
        if record:
            sent.append(np.asarray(cmd["action"], np.float32).reshape(-1))
        # One tick per step: act re-references the gripper to each tick's live proprio.
        obs, done, info = session.step(**{key: value[None] for key, value in cmd.items()},
                                       action_reference="chunk_start")
        obs = last_tick(obs)
        override_prompt(obs, args.prompt)
        if record:
            frames.append(obs)
        steps += 1
        if args.max_steps and steps >= args.max_steps:
            break
    return bool(info.get("success", False)), steps, frames, sent, False


def episode_seed(key: str, base: int) -> int:
    """A stable per-key RNG seed.

    The flow sampler draws its noise from the global RNG, so without this every rollout of the same key is a
    fresh sample and two runs of one checkpoint disagree on ~a third of keys -- enough to swamp the effect
    sizes being measured. Derived with sha256 rather than hash(), whose salt changes per process.
    """
    import hashlib

    digest = hashlib.sha256(f"{base}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def rollout_key(split: str, wods_key: str, success: bool) -> str:
    """WODS key of one stored rollout; the SUCCESS/FAIL tag is part of the key.

    Putting the outcome in the name is what lets a directory listing alone say what happened, so an aggregation
    pass never depends on a separate index it might disagree with.
    """
    tag = "SUCCESS" if success else "FAIL"
    return f"{split}/{wods_key.replace('/', '_')}_{tag}"


def write_result(root: str, key: str, name: str, result: dict) -> None:
    """Atomically write one rollout result."""
    import json
    import os
    import posixpath

    import fsspec

    fs, path = fsspec.core.url_to_fs(f"{root}/{key}/{name}")
    fs.makedirs(posixpath.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with fs.open(temporary, "w") as file:
        json.dump(result, file, indent=2, sort_keys=True)
        file.write("\n")
    if fs.exists(path):
        fs.rm(path)        # --out takes fsspec URLs, and mv onto an existing object is not universally allowed
    fs.mv(temporary, path)


def completed_rollout(root: str, split: str, wods_key: str) -> dict | None:
    """The result of an already-stored rollout for this (split, key), or None.

    Lets a re-run resume: the key marker is written last, so a present key means the rollout finished.
    Refreshes ``result.json`` so :func:`finalize` sees rollouts published by an earlier, interrupted run.

    Resuming matters more for HANGS than for crashes here: judging success on a pick-and-place task walks every
    object's mesh AABB on every step, which can wedge a simulator for hours, and without a per-rollout record a
    restart replays the panel from its first key.
    """
    import json

    import fsspec
    from wods.v2.item import wods_key_exists

    found = [
        (success, rollout_key(split, wods_key, success))
        for success in (True, False)
        if wods_key_exists(root, rollout_key(split, wods_key, success))
    ]
    if len(found) > 1:
        raise RuntimeError(f"both SUCCESS and FAIL keys exist for {split}/{wods_key}")
    if not found:
        return None
    success, key = found[0]
    fs, path = fsspec.core.url_to_fs(f"{root}/{key}/result.json")
    if fs.exists(path):
        with fs.open(path) as file:
            return json.load(file)
    # The key is stored but its result file is not: a run died between the two. Rebuild what the key name
    # carries so :func:`finalize` still counts this rollout. Written ONLY when absent -- result.json is an
    # immutable record of a finished rollout, and rewriting it on every resumed key is pure I/O that also
    # risks dropping fields (``steps``) the key name cannot encode.
    result = {"split": split, "wods_key": wods_key, "success": success, "key": key}
    write_result(root, key, "result.json", result)
    return result


def finalize(root: str) -> int:
    """Aggregate the per-rollout result files under the output root.

    A separate pass rather than in-run bookkeeping, so N independent ranks need no coordination: each writes its
    own rollouts, and this reads whatever is on disk afterwards.
    """
    import json
    import posixpath
    from collections import defaultdict

    import fsspec

    fs, fs_root = fsspec.core.url_to_fs(root)
    paths = fs.find(fs_root) if fs.exists(fs_root) else []
    result_paths = [path for path in paths if posixpath.basename(path) == "result.json"]
    if not result_paths:
        print(f"no rollout results under {root}", flush=True)
        return 1
    by_split = defaultdict(list)
    for path in sorted(result_paths):
        with fs.open(path) as file:
            result = json.load(file)
        by_split[result.get("split", "?")].append(result)
    print(f"\n=== success rate (aggregated from {len(result_paths)} rollouts under {root}) ===", flush=True)
    total = []
    for split, results in sorted(by_split.items()):
        successes = [bool(r["success"]) for r in results]
        total += successes
        print(f"  {split:32s} {sum(successes)}/{len(successes)} = {sum(successes) / len(successes):.3f}")
    print(f"  {'OVERALL':32s} {sum(total)}/{len(total)} = {sum(total) / len(total):.3f}")
    return 0


def distributed(args):
    """Independent-process rank setup: no coordination, just a disjoint slice of the keys each.

    Resumption and parallelism are the same mechanism -- every rank filters its slice by what is already on
    disk -- so re-running the identical command picks up exactly the remainder.
    """
    import os

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if not 0 <= rank < world:
        raise ValueError(f"invalid RANK/WORLD_SIZE: {rank}/{world}")
    if args.device == "cuda" and "LOCAL_RANK" in os.environ:
        args.device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
    if world > 1:
        print(f"rank {rank}/{world}, device={args.device}", flush=True)
    return rank, world


def unevaluable(error) -> bool:
    """True when the server refused this episode as one it cannot rebuild, rather than failing.

    The server aborts with FAILED_PRECONDITION for a scene it cannot reconstruct for the recording -- blacklisted
    assets deleted from it, or a house invalid for the task. The request was well-formed and the server is
    healthy, so the panel skips that key and keeps going; every other status (a dead server, a bad argument) is
    a real failure and propagates.
    """
    import grpc

    return isinstance(error, grpc.RpcError) and error.code() is grpc.StatusCode.FAILED_PRECONDITION


def main(argv=None) -> int:
    import json

    import grpc

    import numpy as np
    import torch

    from omniact.rollout import PolicyAdapter, connect, resolve_bundle, write_rollout_wods
    from omniact.model import load_omniact_model

    args = parse_args(argv)
    if args.finalize:
        if not args.out:
            print("--finalize needs --out", flush=True)
            return 2
        return finalize(args.out)
    rank, world = distributed(args)
    wods_root = args.wods_root or args.data
    # Server-scene mode: with no --replay-root there is no bundle to ship, so the server must already hold
    # the scene (its session factory builds one from its own config and ignores an empty bundle). Nothing
    # then enumerates keys or carries a caption, so run a single unnamed episode and take the instruction
    # from --prompt. Without one the policy gets a zero text embedding, which is not what it was trained on.
    server_scene = not args.replay_root
    if server_scene:
        if args.wods_key:
            raise SystemExit("--wods-key needs --replay-root: a key is only resolvable to a bundle")
        if not args.prompt:
            print("WARNING: no --replay-root and no --prompt, so the policy gets a zero text embedding",
                  flush=True)

    model, _stats, global_step = load_omniact_model(
        args.checkpoint, device=args.device, denoise_steps=args.denoise_steps)
    policy = PolicyAdapter(model, device=args.device, action_horizon=args.action_horizon,
                           command=args.command, image_size=tuple(args.image_size))

    if server_scene:
        split_keys = {"single": ["server_scene"]}      # a name for bookkeeping, not a resolvable key
    elif args.wods_key:
        split_keys = {"single": list(args.wods_key)}
    else:
        split_keys = select_split_keys(args)
    print(f"server: {args.target}\ncheckpoint: {args.checkpoint} (step {global_step})\n"
          f"command={args.command}  action_horizon={policy.action_horizon}  "
          f"predict_horizon={policy.predict_horizon} (from ckpt)\n"
          f"split-seed {args.split_seed}  splits: "
          + ", ".join(f"{n}={len(k)}" for n, k in split_keys.items()), flush=True)

    options = {}
    if args.cameras:
        options["cameras"] = args.cameras
    if args.max_steps:
        # The task truncates at exp_config.task_horizon (500 for pick), so the client cap alone can only
        # ever shorten an episode. One flag sets both, because two that both mean "how long may this run"
        # fail silently when only one is given.
        options["task_horizon"] = str(args.max_steps)
    if args.prompt:
        # The policy gets --prompt from override_prompt; the server gets it too, so one that holds its own
        # scene (no bundle, so no caption) records the episode under it. MolmoSpaces ignores the option.
        options["prompt"] = args.prompt
    record = bool(args.out)

    def save_rollout(out_key, frames, sent):
        """An episode's ``frames`` and ``sent`` actions (the last frame's a zero hold) as ``out_key`` under --out."""
        hold = (np.zeros_like(sent[-1]) if sent else
                np.zeros(np.asarray(frames[0]["node_proprio"]).reshape(-1).shape, np.float32))
        write_rollout_wods(args.out, out_key, frames, sent + [hold], list(frames[0].get("camera_names") or []),
                           fps=args.fps)

    hotkeys = Hotkeys.start() if world == 1 else None
    if hotkeys is not None:
        print(Hotkeys.LEGEND, flush=True)
    per_split: dict[str, list[bool]] = {}
    skipped_by_split: dict[str, list[str]] = {}
    for split_name, keys in split_keys.items():
        successes: list[bool] = []
        skipped: list[str] = []
        assigned = keys[rank::world]          # disjoint slices, so ranks never contend for a key
        if world > 1:
            print(f"  [{split_name}] rank {rank}: {len(assigned)} of {len(keys)} keys", flush=True)
        for key in assigned:
            done = completed_rollout(args.out, split_name, key) if record else None
            if done is not None:
                successes.append(bool(done["success"]))
                steps = done.get("steps", "?")
                print(f"  [{split_name}] {key}: success={done['success']} steps={steps} (resumed)", flush=True)
                continue
            # Per KEY, not per run: the same key samples the same noise in every run, while different keys
            # stay independent. Seeded before the bundle is resolved so nothing upstream consumes the stream.
            torch.manual_seed(episode_seed(key, args.seed))
            if server_scene:
                bundle, emb, caption = {}, None, None
            else:
                bundle = resolve_bundle(args.replay_root, key, wods_root)
                emb = wods_text_embed(wods_root, key)
                caption = json.loads(bundle["wods_meta.json"]).get("caption")
            if args.prompt:
                # Applied per observation in run_episode, not seeded into the prompt cache: the policy
                # prefers obs["text_embed"] over obs["prompt"], so a server that sends an embedding would
                # silently outrank a cache entry. override_prompt drops that field and sets the text.
                print(f"  prompt override: {args.prompt!r}"
                      + (f" (bundle caption was {caption!r})" if caption else ""), flush=True)
            elif emb is not None and caption:
                emb = torch.as_tensor(emb).float()
                policy._prompt_cache[caption] = emb[None] if emb.ndim == 1 else emb
            try:
                while True:                 # tries: the operator's r ends one (saved as STOPPED) and starts the next
                    if hotkeys is not None:
                        hotkeys.hold(until_restart=False)   # paused: no new session yet, its Reset moves the arm
                        hotkeys.restart.clear()
                    # One handler for both points the server can refuse at: the scene is rebuilt either eagerly in
                    # Open or lazily on the first Reset, depending on the session factory.
                    session = connect(args.target, bundle=bundle, options=options)
                    try:
                        obs = session.reset(args.episode_idx)
                        success, steps, frames, sent, stopped = run_episode(session, policy, obs, args, record,
                                                                            hotkeys)
                    finally:
                        session.close()
                    if not stopped:
                        break
                    saved = ""
                    if record and frames:
                        stopped_out = stopped_key(split_name, key)
                        save_rollout(stopped_out, frames, sent)
                        write_result(args.out, stopped_out, "stopped.json",
                                     {"split": split_name, "wods_key": key, "key": stopped_out, "steps": int(steps)})
                        saved = f", saved as {stopped_out}"
                    print(f"  [{split_name}] {key}: stopped after {steps} steps{saved}; running it again", flush=True)
                    torch.manual_seed(episode_seed(key, args.seed))     # the same noise as the first try
            except grpc.RpcError as error:
                if not unevaluable(error):
                    raise
                skipped.append(key)
                print(f"  [{split_name}] {key}: SKIPPED -- {error.details()}", flush=True)
                continue
            successes.append(success)
            print(f"  [{split_name}] {key}: success={success} steps={steps}", flush=True)
            if record and frames:
                out_key = rollout_key(split_name, key, success)
                save_rollout(out_key, frames, sent)
                # After the rollout data (whose last write is the key marker), so a stored key always means a
                # finished rollout and result.json always has a complete rollout beside it.
                write_result(args.out, out_key, "result.json", {
                    "split": split_name, "wods_key": key, "key": out_key,
                    "success": bool(success), "steps": int(steps),
                })
        per_split[split_name] = successes
        skipped_by_split[split_name] = skipped

    print("\n=== success rate ===", flush=True)
    all_success: list[bool] = []
    for split_name, successes in per_split.items():
        all_success += successes
        n = len(successes)
        rate = (sum(successes) / n) if n else float("nan")
        print(f"  {split_name:32s} {sum(successes)}/{n} = {rate:.3f}"
              + (f"  ({len(skipped_by_split[split_name])} key(s) skipped as unevaluable, not in the "
                 f"denominator)" if skipped_by_split.get(split_name) else ""))
    n = len(all_success)
    print(f"  {'OVERALL':32s} {sum(all_success)}/{n} = {(sum(all_success) / n) if n else float('nan'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
