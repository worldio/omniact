"""Write a closed-loop rollout to the dataset's WODS format.

Each eval episode is published as a WODS key with one shard per camera, carrying the same fields the embgen
dataset does -- ``video`` (T,H,W,3) uint8, ``action`` = node_action (T,N), ``node_proprio`` (T,N),
``node_geometry`` (T,N,2,3) per view, and the ``node_graph`` -- so eval rollouts read back exactly like the
training data (``wods.v2.item.read_wods_shard_batched``). Frame ``i`` holds the obs BEFORE ``action[i]``, and
``action[i]`` drives ``i -> i+1`` (the last action is a zero hold), matching the dataset's convention.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def write_rollout_wods(root: str, key: str, frames: list[dict[str, Any]], actions: list,
                       camera_names: list[str], *, fps: float = 30.0, crf: int = 23) -> str:
    """Publish one rollout (``frames`` obs + aligned ``actions``, ``len==len``) as a WODS key under ``root``.
    ``camera_names`` is the V-axis order of ``obs['views']``/``obs['node_geometry']`` (the server's obs order).
    The key's done-marker is written last, so a reader only ever sees a complete key."""
    import torch
    from wods.v2.item import (
        write_wods_key_done, write_wods_metadata, write_wods_shard, write_wods_shard_done,
    )

    if not frames or len(frames) != len(actions):
        raise ValueError(f"rollout needs equal, non-empty frames/actions (got {len(frames)}/{len(actions)})")
    if not camera_names:
        raise ValueError("camera_names is required (the obs['views'] V-axis order)")
    t = len(frames)
    node_proprio = torch.from_numpy(np.stack([np.asarray(f["node_proprio"], np.float32) for f in frames]))  # (T,N)
    action = torch.from_numpy(np.stack([np.asarray(a, np.float32).reshape(-1) for a in actions]))           # (T,N)
    timestamps = torch.arange(t, dtype=torch.float32) / float(fps)
    g0 = frames[0]
    node_graph = {k: g0.get(k) for k in ("edges", "axis_mask", "joint_type", "node_names")}
    caption = str(g0.get("prompt") or "")

    for v, camera in enumerate(camera_names):
        video = torch.from_numpy(np.stack([np.asarray(f["views"][v], np.uint8) for f in frames]))           # (T,H,W,3)
        node_geometry = torch.from_numpy(
            np.stack([np.asarray(f["node_geometry"], np.float32)[:, :, v, :] for f in frames]))             # (T,N,2,3)
        write_wods_shard(root, key, camera, video=video, timestamps=timestamps, action=action,
                         node_proprio=node_proprio, node_geometry=node_geometry, node_graph=node_graph, crf=crf)
    write_wods_metadata(root, key, caption=caption)
    for camera in camera_names:
        write_wods_shard_done(root, key, camera)
    write_wods_key_done(root, key)
    return key
