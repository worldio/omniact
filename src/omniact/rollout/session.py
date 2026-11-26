"""The `RolloutSession` contract — a gym-like closed-loop env wrapper, driven by the caller.

This is the interface the remote protocol mirrors: molmospaces implements it server-side over its sim +
FK (see ``rollout/proto/rollout.proto`` and ``rollout/remote.py``), and eval drives it with a local
policy."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class RolloutSession(ABC):
    """One open rollout session — gym-like env wrapper.

    Lifecycle (driven by the caller, not the embodiment):

        session = connect(...)                     # or embodiment.new_rollout(**kwargs)
        try:
            for ep in range(session.n_episodes):
                obs = session.reset(ep)
                policy.reset()
                done, info = False, {}
                while not done:
                    command = {k: v[None] for k, v in policy.act(obs).items()}     # one tick
                    obs, done, info = session.step(**command, action_reference="chunk_start")
                    obs = last_tick(obs)
                # info carries per-episode metrics (success, reward, ...)
        finally:
            session.close()

    Observation contract — each :meth:`reset` returns a dict with at least:
      - ``views``:   ``Tensor (V, H, W, C)`` in ``[-1, 1]`` channel-last, the stacked model-facing
                     cameras (rendered and resized by the server).
      - ``proprio``: ``Tensor (proprio_dim,)`` or ``None`` for obs-only embodiments.
      - ``prompt``:  ``str``, the natural-language task description.
    Extra keys (``node_geometry``, ``node_proprio``, ``edges``, ``axis_mask``, ``joint_type``,
    ``node_names``, raw multi-cam frames, ...) may appear; callers ignore what they do not need. Each
    :meth:`step` returns the same for the ticks it ran, every tensor with a leading tick dim (``node_names``,
    ``camera_names`` and ``prompt`` as they are); :func:`last_tick` takes the last.

    Info dict (returned from :meth:`step` when ``done=True``) should include at least:
      - ``success``: bool, did this episode succeed
      - ``episode_idx``: int
      - any per-prompt / per-episode metrics worth aggregating
    """

    @property
    @abstractmethod
    def n_episodes(self) -> int:
        """Number of episodes available in this session."""

    @abstractmethod
    def reset(self, episode_idx: int) -> dict[str, Any]:
        """Begin episode ``episode_idx``. Return the initial observation."""

    @abstractmethod
    def step(
        self, action: torch.Tensor, node_action_pose: torch.Tensor,
        action_mask: torch.Tensor, node_action_pose_mask: torch.Tensor, *, action_reference: str,
    ) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        """Advance k policy ticks, run in order until the episode ends. All four are required (a head you do not
        command is zeros with an all-False mask), each with a leading tick dim k: the relative joint ``action``
        (k, N), the absolute ``node_action_pose`` (k, N, 3, 4), and each head's (k, N) bool mask of the dims it
        commands. A session may reject a k it does not support. ``action_reference``, also required, is the joints
        ``action`` is relative to: "chunk_start", the joints at the step's start, for all k ticks; or "each_tick",
        each tick's own. The two agree when k = 1; a session may reject one it does not support.

        Returns:
            (observation, done, info): the observation of the ticks run, and done and info at the last of them.
        """

    def close(self) -> None:
        """Optional teardown — default no-op."""


def last_tick(obs: dict[str, Any]) -> dict[str, Any]:
    """A :meth:`RolloutSession.step` observation at its last tick, shaped as :meth:`RolloutSession.reset`'s."""
    return {key: value if key in ("node_names", "camera_names", "prompt") or value is None else value[-1]
            for key, value in obs.items()}
