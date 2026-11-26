#!/usr/bin/env python
"""Client-side stepping-replay fidelity check: omniact client -> omniact-molmospaces ``eval_server`` over gRPC.

Connects to a running ``eval_server``, **sends it the replay bundle's content** (the client reads the bundle
-- from a local dir or ``gs://`` -- so the server needs no shared filesystem or path), then drives the
recorded trajectory with DRIFT-CORRECTED ground-truth actions and compares the server's per-step observations
against the cloud WODS recording. The server reconstructs + physics-steps from the sent bundle; this client
owns the bundle bytes + the recorded WODS (the GT actions + comparison target) + the gRPC connection.

Drift-corrected actions
-----------------------
The server rebuilds the absolute joint target from its LIVE state: ``target = node_action + current_qpos``.
In a physics replay the live proprio drifts from the recording, so sending the recorded relative
``node_action`` directly would command a drifting target. Instead we send, per node dimension::

    action_sent[k] = node_action[i][k] + node_proprio[i][k] - server_current_node_proprio[k]

so the server rebuilds the exact recorded ABSOLUTE target ``a_i`` every step, re-pinning to the recording
instead of compounding drift. (``node_proprio`` is centered, so ``captured - current`` cancels the center and
leaves the native ``q_i - q_cur`` delta; the server adds the native current qpos back.) ``node_action`` /
``node_proprio`` / the server's obs ``node_proprio`` share one node order (embodiment_wods builds both from
the same node_spec; the server's obs comes from the same representations) -- asserted at reset via node_names
and node_proprio[0].

Usage
-----
Start the server (omniact-molmospaces environment), then run this (this repository's environment)::

    # omniact-molmospaces:
    python -m molmo_spaces.rollout.eval_server --port 50051
    # omniact:
    python scripts/replay_check.py --target localhost:50051 \
        --wods-root <WODS root> --wods-key <gen_.../.../..._trajNN> \
        --replay-root <replay bundle root> \
        [--out <root>]   # optional: also record the replayed rollout to WODS

``--replay-root`` is the replay bundle root; the exact bundle is resolved house-aware from ``--wods-key``
(an embodiment spans multiple houses, so episode_idx alone is ambiguous) -- pass just the root, not a per-
trajectory path. The chosen bundle's content is read by the client and sent to the server (local dir or
``gs://``). ``--out``
additionally records the rollout in the dataset's WODS format -- per-camera video + node_action /
node_proprio / node_geometry / node_graph -- so it reads back exactly like training data (this is the
recording that ``run_episode`` used to do, now folded into the one client). Action noise should be off on
the server for a deterministic replay.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True, help="eval_server address, host:port")
    p.add_argument("--wods-key", required=True, help="WODS trajectory key of the episode to check")
    p.add_argument("--wods-root", default=os.environ.get("WODS_ROOT"),
                   help="WODS root with the recorded per-step node_action/node_proprio/node_geometry "
                        "(any fsspec target); or set $WODS_ROOT")
    p.add_argument("--replay-root", default=os.environ.get("REPLAY_ROOT"),
                   help="replay bundle ROOT (any fsspec target, e.g. a local directory or a gs:// prefix, or "
                        "$REPLAY_ROOT); the exact bundle is resolved house-aware from --wods-key")
    p.add_argument("--cameras", default=None, help="';'-separated camera subset (Open option)")
    p.add_argument("--episode-idx", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None, help="cap steps (default: whole trajectory)")
    # Reset (frame 0) is pre-noise: the reconstruction must be TIGHT. Stepping is noise-bounded (the WODS
    # node_action is pre-noise, so a noise-off replay diverges by the per-step noise), so it's a CLOSENESS
    # check, not byte-exact -- a loose bound that the drift-correction keeps the trajectory within.
    p.add_argument("--recon-tol", type=float, default=1e-3, help="tight tol for the reset reconstruction")
    p.add_argument("--proprio-tol", type=float, default=0.6, help="loose closeness bound for stepping proprio")
    p.add_argument("--geometry-tol", type=float, default=0.05, help="loose closeness bound (metres) for geometry")
    p.add_argument("--pose-tol", type=float, default=0.05, help="loose closeness bound for node_pose entries")
    p.add_argument("--command", choices=("joint", "pose", "hybrid"), default="joint",
                   help="what to drive per step: the drift-corrected joint action (default); the TCP setup -- "
                        "node_action_pose masked to the TCP row (server IK) + the joint action masked to the "
                        "gripper driver dims; or both heads (hybrid). All four StepRequest fields are always sent.")
    p.add_argument("--out", help="if set, ALSO save the replayed rollout to WODS under this root -- the same "
                                 "format the dataset uses (per-camera video + node_action/node_proprio/"
                                 "node_geometry/node_graph), read back like training data")
    p.add_argument("--out-key", help="WODS key to write under --out (default: --wods-key)")
    p.add_argument("--fps", type=float, default=None,
                   help="fps for the saved rollout (--out); default: the recorded trajectory's own rate "
                        "(from its WODS timestamps), so the video matches the GT sim-time playback")
    return p.parse_args(argv)


def load_recorded(root: str, key: str):
    """Recorded per-step ground truth from the WODS: ``(node_action (T,N), node_proprio (T,N),
    node_geometry (T,N,P,V,3), node_names, view_names)`` -- the only thing this client reads. ``view_names``
    is the shard (camera) name per V-axis slot, so the caller can align node_geometry to the server's camera
    order (the two need not agree: WODS shards are alphabetical, the env registry is config order)."""
    from wods.v2.item import list_wods_shards, read_wods_shard_batched

    shards = sorted(list_wods_shards(root, key))
    view_shards = [s for s in shards if s.startswith("meta/")] or shards
    if not view_shards:
        raise ValueError(f"no shards under {root}/{key}")
    off = dict(include_video=False, include_video_raw=False, include_timestamps=True,
               include_latents=False, include_camera=False, include_points=False, include_depth=False,
               include_mask=False, include_proprio=False, include_target=False, include_text_embed=False,
               include_node_action_geometry=False)
    by_shard = {s: read_wods_shard_batched([root], [key], [s], include_action=True, include_node_proprio=True,
                                           include_node_geometry=True, include_node_graph=True,
                                           include_node_pose=True, include_node_action_pose=True, **off)[0]
                for s in view_shards}
    prim = by_shard[view_shards[0]]                                  # node_action/node_proprio are per-key
    node_action = np.asarray(prim["action"], np.float64)            # (T, N) recorded relative command
    node_proprio = np.asarray(prim["node_proprio"], np.float64)     # (T, N) proprio right before the command
    geom = np.stack([np.asarray(by_shard[s]["node_geometry"]) for s in view_shards], axis=3)  # (T,N,P,V,3)
    node_names = list(prim["node_graph"]["node_names"]) if "node_graph" in prim else None
    view_names = [s.rsplit("/", 1)[-1] for s in view_shards]        # camera name per V slot (shard basename)
    timestamps = np.asarray(prim["timestamps"], np.float64) if prim.get("timestamps") is not None else None

    # The per-node pose tracks, read via the wods library (include_node_pose / include_node_action_pose above):
    # the label node_action_pose (T,N,3,4), one track per key (from any shard), and node_pose per camera stacked
    # to (T,N,V,3,4) like node_geometry. Absent (None) when the recording predates them.
    node_action_pose = np.asarray(prim["node_action_pose"], np.float64) if prim.get("node_action_pose") is not None else None
    node_pose = (np.stack([np.asarray(by_shard[s]["node_pose"]) for s in view_shards], axis=2)
                 if prim.get("node_pose") is not None else None)
    return (node_action, node_proprio, geom, node_names, view_names, timestamps, node_action_pose, node_pose)


def main(argv=None) -> int:
    from omniact.rollout import connect, last_tick, resolve_bundle, write_rollout_wods

    args = parse_args(argv)
    if not args.wods_root:
        raise SystemExit("--wods-root is required (pass it, or set $WODS_ROOT)")
    if not args.replay_root:
        raise SystemExit("--replay-root is required (pass it, or set $REPLAY_ROOT); the bundle is resolved from --wods-key")
    import json
    # The WODS caption rides IN the bundle (wods_meta.json) -- a logical bundle member, resolved alongside the
    # replay files, so the server reads the authoritative per-episode instruction from the self-contained bundle.
    bundle = resolve_bundle(args.replay_root, args.wods_key, args.wods_root)
    caption = json.loads(bundle["wods_meta.json"]).get("caption")
    print(f"server: {args.target}\nwods:   {args.wods_root}/{args.wods_key}\n"
          f"bundle: {len(bundle)} files  caption={caption!r}", flush=True)

    (rec_action, rec_proprio, rec_geom, rec_names, rec_views, rec_ts,
     rec_action_pose, rec_pose) = load_recorded(args.wods_root, args.wods_key)
    if args.command != "joint" and rec_action_pose is None:
        raise SystemExit(f"--command {args.command} needs the recording's node_action_pose (absent in this WODS)")
    T = len(rec_action) if args.max_steps is None else min(args.max_steps + 1, len(rec_action))
    print(f"recorded steps: {len(rec_action)}  nodes: {rec_action.shape[1]}  command: {args.command}  (checking {T-1})", flush=True)
    # Save at the RECORDED trajectory's fps (from its WODS timestamps) so the video matches the GT sim-time
    # playback; fall back to 30 only if the recording has no usable timestamps.
    out_fps = args.fps
    if out_fps is None:
        dt = float(np.median(np.diff(rec_ts))) if rec_ts is not None and len(rec_ts) > 1 else 0.0
        out_fps = (1.0 / dt) if dt > 0 else 30.0

    options = {}
    if args.cameras:
        options["cameras"] = args.cameras
    save = bool(args.out)
    frames: list = []              # accumulated obs for --out (each carries the rendered views + node fields)
    sent_actions: list = []
    session = connect(args.target, bundle=bundle, options=options)   # reads the bundle files, sends their bytes
    try:
        obs = session.reset(args.episode_idx)
        if save:
            frames.append(obs)

        # --- node-order guard: names + frame-0 proprio must match the recording ---
        srv_names = list(obs.get("node_names") or [])
        if rec_names is not None and srv_names and srv_names != rec_names:
            raise SystemExit(f"NODE ORDER MISMATCH\n server: {srv_names}\n wods:   {rec_names}")

        # --- view alignment: node_geometry's V axis is per-camera; the server's camera order (obs
        # camera_names) need not match the WODS shard order, so permute rec_geom's V axis to the server's. ---
        srv_views = list(obs.get("camera_names") or [])
        if srv_views and rec_views:
            if sorted(srv_views) != sorted(rec_views):
                raise SystemExit(f"CAMERA SET MISMATCH\n server: {srv_views}\n wods:   {rec_views}")
            perm = [rec_views.index(name) for name in srv_views]   # WODS V-index for each server camera
            rec_geom = rec_geom[:, :, :, perm, :]                  # (T,N,P,V,3) -> server camera order
            if rec_pose is not None:
                rec_pose = rec_pose[:, :, perm]                    # (T,N,V,3,4) likewise
            print(f"views aligned to server order: {srv_views}", flush=True)

        # The command's masks (all four StepRequest fields are always sent; a head not in use is zeros with an
        # all-False mask): joint = the whole joint head; pose = the TCP setup, node_action_pose masked to the TCP
        # row (server IK) + the joint action masked to the gripper driver dims; hybrid = the joint head + the
        # pose row (the server takes it as the IK posture reference).
        N = len(srv_names)
        types = np.asarray(obs["joint_type"]).reshape(-1)
        drivers = np.asarray(obs["axis_mask"], bool).reshape(-1) & np.array([n.startswith("gripper/") for n in srv_names])
        tcp_row, all_on, none_on = (types == 3), np.ones(N, bool), np.zeros(N, bool)
        act_mask, pose_mask = {"joint": (all_on, none_on), "pose": (drivers, tcp_row),
                               "hybrid": (all_on, tcp_row)}[args.command]
        zeros_pose = np.zeros((N, 3, 4), np.float64)
        if args.command != "joint":
            print(f"masks: action dims {np.flatnonzero(act_mask).tolist()}, pose row {np.flatnonzero(pose_mask).tolist()}", flush=True)

        p0 = float(np.abs(np.asarray(obs["node_proprio"], np.float64) - rec_proprio[0]).max())
        g0 = float(np.abs(np.asarray(obs["node_geometry"], np.float64) - rec_geom[0]).max())
        q0 = (float(np.abs(np.asarray(obs["node_pose"], np.float64) - rec_pose[0]).max())
              if rec_pose is not None and obs.get("node_pose") is not None else None)
        # frame 0 is before any action -> no action noise: this is the pure reconstruction check (tight).
        print(f"reset node_proprio[0] max|Δ|: {p0:.3g}   node_geometry[0] max|Δ|: {g0:.3g} m  "
              + (f"node_pose[0] max|Δ|: {q0:.3g}  " if q0 is not None else "") + "(reconstruction, pre-noise)", flush=True)

        prop_err, geom_err, pose_err = [], [], []
        first_success, info, done = None, {}, False
        for i in range(T - 1):
            cur = np.asarray(obs["node_proprio"], np.float64)                    # server live proprio (centered)
            sent = torch.as_tensor(rec_action[i] + rec_proprio[i] - cur)        # drift-corrected -> exact joint target
            pose = torch.as_tensor(rec_action_pose[i] if args.command != "joint" else zeros_pose)
            if save:
                sent_actions.append(np.asarray(sent, np.float32).reshape(-1))   # command applied at frame i
            # One tick per step: each tick's action is drift-corrected against the live proprio.
            obs, done, info = session.step(sent[None], pose[None], torch.as_tensor(act_mask)[None],
                                           torch.as_tensor(pose_mask)[None], action_reference="chunk_start")
            obs = last_tick(obs)
            if save:
                frames.append(obs)
            ep = float(np.abs(np.asarray(obs["node_proprio"], np.float64) - rec_proprio[i + 1]).max())
            prop_err.append(ep)
            eg = float(np.abs(np.asarray(obs["node_geometry"], np.float64) - rec_geom[i + 1]).max())
            geom_err.append(eg)
            eq = (float(np.abs(np.asarray(obs["node_pose"], np.float64) - rec_pose[i + 1]).max())
                  if rec_pose is not None and obs.get("node_pose") is not None else None)
            if eq is not None:
                pose_err.append(eq)
            if i < 3 or ep > args.proprio_tol or eg > args.geometry_tol or (eq is not None and eq > args.pose_tol):
                print(f"  step {i+1}/{T-1}: node_proprio Δ={ep:.3g}  node_geometry Δ={eg:.3g}"
                      + (f"  node_pose Δ={eq:.3g}" if eq is not None else "")
                      + (f"  ik={info['ik_outcome']}" if "ik_outcome" in info else "") + f"  done={done}", flush=True)
            if info.get("success") and first_success is None:
                first_success = i + 1
                print(f"  task success first reported at step {first_success}", flush=True)
            if done:
                break
        print(f"task success at end: {bool(info.get('success'))}  "
              f"(first at step {first_success}, ran {len(prop_err)} steps, done={done})", flush=True)
    finally:
        session.close()

    # --out: also SAVE the replayed rollout in the dataset's WODS format (per-camera video + node_action/
    # node_proprio/node_geometry/node_graph), read back exactly like training data. frame i is the obs
    # before sent_actions[i]; the terminal frame gets a zero hold so #frames == #actions.
    if save and frames:
        hold = (np.zeros_like(sent_actions[-1]) if sent_actions
                else np.zeros(np.asarray(frames[0]["node_proprio"]).reshape(-1).shape, np.float32))
        sent_actions.append(hold)
        cams = list(frames[0].get("camera_names") or [])
        wrote = write_rollout_wods(args.out, args.out_key or args.wods_key, frames, sent_actions, cams,
                                   fps=out_fps)
        print(f"saved rollout -> {args.out}/{wrote}  ({len(frames)} frames, {len(cams)} cameras, "
              f"{out_fps:.3g} fps)", flush=True)

    pmax = max(prop_err) if prop_err else float("nan")
    gmax = max(geom_err) if geom_err else float("nan")
    qmax = max(pose_err) if pose_err else float("nan")
    # Reconstruction (reset, pre-noise) must be tight; stepping is a closeness check (noise-bounded).
    recon_ok = p0 <= args.recon_tol and g0 <= args.recon_tol and (q0 is None or q0 <= args.recon_tol)
    step_ok = ((not prop_err or pmax <= args.proprio_tol) and (not geom_err or gmax <= args.geometry_tol)
               and (not pose_err or qmax <= args.pose_tol))
    print(f"\nreconstruction (reset): node_proprio[0]={p0:.3g} node_geometry[0]={g0:.3g}m "
          + (f"node_pose[0]={q0:.3g} " if q0 is not None else "") + f"(tol {args.recon_tol}) -> {'EXACT' if recon_ok else 'OFF'}")
    print(f"stepping closeness ({args.command}): node_proprio max|Δ|={pmax:.4g} (tol {args.proprio_tol}, noise-bounded)  "
          f"node_geometry max|Δ|={gmax:.4g}m (tol {args.geometry_tol})"
          + (f"  node_pose max|Δ|={qmax:.4g} (tol {args.pose_tol})" if pose_err else ""))
    print("RESULT:", "CLOSE" if (recon_ok and step_ok) else "FAR", flush=True)
    return 0 if (recon_ok and step_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
