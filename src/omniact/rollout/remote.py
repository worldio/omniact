"""`RemoteRolloutSession` — a `RolloutSession` backed by a gRPC `Rollout` server (molmospaces).

The driver loop is unchanged (see `RolloutSession`); only the session is remote. The policy runs
locally and never does FK — observations already carry `node_geometry`/`node_proprio` computed
server-side by the same FK that produced the training data."""
from __future__ import annotations

from typing import Any

import grpc
import torch

from . import codec
from .proto import rollout_pb2, rollout_pb2_grpc
from .session import RolloutSession

# The bundle (states.npz etc.) rides in the Open request; lift gRPC's 4 MB default well above bundle size.
_MAX_MSG = 512 * 1024 * 1024

# RolloutSession.step's action_reference -> the wire's.
_ACTION_REFERENCES = {"chunk_start": rollout_pb2.ACTION_REFERENCE_CHUNK_START,
                      "each_tick": rollout_pb2.ACTION_REFERENCE_EACH_TICK}


class RemoteRolloutSession(RolloutSession):
    def __init__(
        self,
        target: str,
        *,
        bundle: dict[str, bytes],
        options: dict[str, Any] | None = None,
        channel: grpc.Channel | None = None,
    ):
        self._channel = channel or grpc.insecure_channel(
            target, options=[("grpc.max_receive_message_length", _MAX_MSG),
                             ("grpc.max_send_message_length", _MAX_MSG)])
        self._owns_channel = channel is None
        self._stub = rollout_pb2_grpc.RolloutStub(self._channel)
        session = self._stub.Open(rollout_pb2.OpenRequest(
            bundle=bundle,
            options={k: str(v) for k, v in (options or {}).items()}))
        self._id = session.session_id
        self._n_episodes = int(session.n_episodes)

    @property
    def n_episodes(self) -> int:
        return self._n_episodes

    def reset(self, episode_idx: int) -> dict[str, Any]:
        obs = self._stub.Reset(rollout_pb2.ResetRequest(session_id=self._id, episode_idx=int(episode_idx)))
        return codec.proto_to_obs(obs)

    def step(self, action, node_action_pose, action_mask, node_action_pose_mask, *,
             action_reference: str) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        request = rollout_pb2.StepRequest(session_id=self._id, action_reference=_ACTION_REFERENCES[action_reference])
        request.action.CopyFrom(codec.tensor_to_proto(torch.as_tensor(action)))
        request.node_action_pose.CopyFrom(codec.tensor_to_proto(torch.as_tensor(node_action_pose)))
        request.action_mask.CopyFrom(codec.tensor_to_proto(torch.as_tensor(action_mask)))
        request.node_action_pose_mask.CopyFrom(codec.tensor_to_proto(torch.as_tensor(node_action_pose_mask)))
        result = self._stub.Step(request)
        return codec.proto_to_obs(result.obs), bool(result.done), codec.proto_to_info(result.info)

    def close(self) -> None:
        try:
            self._stub.Close(rollout_pb2.CloseRequest(session_id=self._id))
        finally:
            if self._owns_channel:
                self._channel.close()
