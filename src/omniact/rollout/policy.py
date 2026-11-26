"""`PolicyAdapter` — wraps a `OmniActModel` as `reset()` / `act(obs) -> step-command dict` (one tick's, keyed
by the `RolloutSession.step` arg names) for the `RolloutSession` loop.

Mirrors the dataset's per-node processing (right-align to `n_node`, model units, per-slot std, node_pose
observation) and reverses it, decoding both heads into the server's contract (per-step relative joint +
ABSOLUTE pose). The relative targets are re-referenced to the chunk start: the gripper to the latched
proprio, and the TCP pose to `P_now = c2w · node_pose` (the current world TCP pose from the observation).
The node graph comes straight from the observation, so the policy does no FK."""
from __future__ import annotations

from typing import Any

import torch

from ..dataset import (
    MAX_NODES, POSE_ACTION_ORIGIN, chunk_scale, gripper_drivers, joint_scale_factor, node_mask_for,
    pad_nodes_front, right_align_adjacency,
)
from ..joint_types import (
    JOINT_TCP, pose9_to_se3, se3_mul, se3_to_pose9,
    to_model_joint_units, to_native_joint_units, validate_joint_type,
)


class PolicyAdapter:
    COMMANDS = ("joint", "pose", "hybrid")

    def __init__(self, model, *, n_node: int | None = None, device: str = "cuda",
                 action_horizon: int = 8, command: str = "joint",
                 image_size: tuple[int, int] = (256, 256), text_encoder=None, tokenizer=None):
        self.model = model.eval()
        self.device = device
        # Which head(s) to command each step (mirrors replay_check --command): ``joint`` drives every actuated
        # dim from the joint head; ``pose`` drives the arm by server IK on the TCP pose + the gripper from the
        # joint head; ``hybrid`` sends the whole joint head AND the TCP pose (as the IK posture reference).
        if command not in self.COMMANDS:
            raise ValueError(f"command must be one of {self.COMMANDS}, got {command!r}")
        self.command = command
        self.n_node = int(n_node or getattr(model.config, "n_node", MAX_NODES))
        self.text_embed_dim = int(model.config.text_embed_dim)
        # Chunk length to denoise each re-plan is the checkpoint's TRAINED chunk length (config.action_horizon ==
        # window_size-1); not overridable. Serve at most that many steps per re-plan (can't serve more than predicted).
        self.predict_horizon = int(model.config.action_horizon)
        self.action_scale = chunk_scale(model.action_std, self.predict_horizon).to(device)   # (T, n_node) joint
        self.pose_scale = chunk_scale(model.pose_action_std, self.predict_horizon).cpu()   # (T, n_node, POSE_DIM) TCP
        self.action_horizon = min(int(action_horizon), self.predict_horizon)   # steps executed per re-plan
        self.image_size = tuple(image_size)     # what the WODS loader crop_resizes to (dataset default 256^2)
        self._text_encoder = text_encoder       # Qwen encoder+tokenizer (shared with the WODS precompute);
        self._tokenizer = tokenizer             # lazily loaded on first prompt if left None
        self._prompt_cache: dict[str, torch.Tensor] = {}   # prompt -> (L, D_text); constant within an episode
        self.plan: dict | None = None           # current chunk: joint (T,n_real) native + pose (T,n_real,3,4) world + masks
        self.plan_index = 0                     # index of the next step to serve from the plan

    def reset(self) -> None:
        """Drop the current plan so the next act() re-plans (call at each episode start)."""
        self.plan = None
        self.plan_index = 0

    def act(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """One tick -> its ``RolloutSession.step`` command as a dict keyed by the arg names (natural node order,
        n_real), without the tick dim (a caller steps it as one tick, each tensor ``[None]``): the relative joint
        ``action`` (N,), the absolute ``node_action_pose`` (N, 3, 4), and each head's (N,) bool command mask
        (``action_mask`` / ``node_action_pose_mask``). Receding horizon: re-plan every ``action_horizon`` steps,
        serve in between."""
        if self.plan is None or self.plan_index >= self.action_horizon:
            self.plan = self._predict_chunk(obs)
            self.plan_index = 0
        k = self.plan_index
        self.plan_index += 1
        p = self.plan
        action = p["joint"][k].clone()                                  # (n_real,) arm delta + gripper (target-center)
        # The gripper is referenced to the CHUNK START: the plan holds ``g[k] + centered_proprio0``, so subtracting
        # the LIVE centered proprio makes the server rebuild ``g[k] + proprio0`` — the drift-corrected step-0 target
        # (the center cancels). The arm dims stay a per-step delta the server adds live qpos to.
        live = torch.as_tensor(obs["node_proprio"]).float()            # (n_real,) native centered proprio
        action[p["gripper"]] -= live[p["gripper"]]
        return {"action": action, "node_action_pose": p["pose"][k],
                "action_mask": p["action_mask"], "node_action_pose_mask": p["pose_mask"]}

    @torch.no_grad()
    def _predict_chunk(self, obs: dict[str, Any]) -> dict:
        """Predict one chunk and decode both heads into the server's contract — relative joint + ABSOLUTE pose —
        latching the chunk-start references (gripper proprio and the world TCP pose ``P_now``). Returns a plan:
        per-step ``joint`` (T, n_real) native, ``pose`` (T, n_real, 3, 4) world, and the (n_real,) head masks."""
        # n_node = fixed frame width (model slots); n_real = this embodiment's nodes.
        device, n_node = self.device, self.n_node
        axis_mask = torch.as_tensor(obs["axis_mask"]).bool()
        joint_type = validate_joint_type(torch.as_tensor(obs["joint_type"]), axis_mask)
        node_names = list(obs["node_names"])
        edges = torch.as_tensor(obs["edges"]).long()
        n_real = axis_mask.shape[0]
        adjacency = torch.eye(n_real, dtype=torch.bool)
        adjacency[edges[:, 0], edges[:, 1]] = True
        adjacency[edges[:, 1], edges[:, 0]] = True
        joint_type_padded = pad_nodes_front(joint_type, n_node, 0).to(device)

        # Training stacks views in sorted camera-name order; permute EVERY per-view field together.
        cams = list(obs.get("camera_names") or [])
        perm = torch.tensor(sorted(range(len(cams)), key=lambda i: cams[i]), dtype=torch.long) if cams else None

        def by_view(value, axis):
            value = torch.as_tensor(value)
            return value.index_select(axis, perm.to(value.device)) if perm is not None else value

        geometry = pad_nodes_front(by_view(obs["node_geometry"], 2).float(), n_node, axis=0)
        node_proprio = pad_nodes_front(
            to_model_joint_units(torch.as_tensor(obs["node_proprio"]).float(), joint_type), n_node, axis=0)
        # node_pose observation: per-view camera-frame SE(3) -> rot6d⊕trans, right-aligned (matches the dataset).
        node_pose_cam = by_view(obs["node_pose"], 1).float()           # (n_real, V, 3, 4), sorted camera order
        node_pose = pad_nodes_front(se3_to_pose9(node_pose_cam), n_node, axis=0)   # (n_node, V, POSE_DIM)
        views = self._prep_views({**obs, "views": by_view(obs["views"], 0)})   # (V, 1, H, W, C), sorted
        text_embed, text_mask = self._text_embed(obs)                  # (1, L, D), (1, L)

        batch = {
            "views": views[None].to(device),
            "geometry": geometry[None].to(device),
            "node_id": torch.arange(n_node, device=device)[None],
            "axis_mask": pad_nodes_front(axis_mask, n_node, 0).to(device)[None],
            "joint_type": joint_type_padded[None],
            "node_proprio": node_proprio[None].to(device),
            "node_pose": node_pose[None].to(device),
            "adjacency": right_align_adjacency(adjacency, n_node).to(device)[None],
            "node_mask": node_mask_for(n_node, n_real).to(device)[None],
            "text_embed": text_embed.to(device), "text_embed_mask": text_mask.to(device),
        }
        joint, pose = self.model.predict(batch, n_steps=self.predict_horizon)   # (1,T,n_node), (1,T,n_node,POSE_DIM)

        # --- joint head: denorm -> native; arm stays a per-step delta, gripper is re-referenced to chunk step 0 ---
        T_out = joint.shape[1]
        factor = joint_scale_factor(node_names, axis_mask, joint_type, n_node).to(joint.device)   # gripper travel
        joint = to_native_joint_units(joint[0] * self.action_scale[:T_out] * factor, joint_type_padded)
        joint = joint[:, n_node - n_real:].cpu()                       # (T, n_real) natural order
        gripper = gripper_drivers(node_names, axis_mask)               # (n_real,) driver dims
        proprio0 = torch.as_tensor(obs["node_proprio"]).float()       # (n_real,) native centered proprio, LATCHED
        joint[:, gripper] += proprio0[gripper]                        # g[k] + centered_proprio0 (act subtracts live)

        # --- pose head: denorm -> relative SE(3); compose with the chunk-start world TCP pose P_now = c2w·node_pose ---
        pose = pose[0].cpu() * self.pose_scale[:T_out] + POSE_ACTION_ORIGIN
        tcp = joint_type == JOINT_TCP                                  # (n_real,) TCP row
        pose_abs = torch.zeros(pose.shape[0], n_real, 3, 4)           # (T, n_real, 3, 4) world; non-TCP zero
        if tcp.any():
            c2w0 = by_view(obs["c2w"], 0).float()[0]               # (3, 4), first SORTED camera -> world
            p_now = se3_mul(c2w0, node_pose_cam[:, 0][tcp])         # (n_tcp, 3, 4) world current TCP pose
            p_rel = pose9_to_se3(pose[:, n_node - n_real:][:, tcp])  # (T, n_tcp, 3, 4) relative to P_now
            pose_abs[:, tcp] = se3_mul(p_now[None], p_rel)          # (T, n_tcp, 3, 4) absolute world

        # --- command masks (mirrors replay_check): which joint dims + pose rows to send this step. The plan itself
        # (arm delta + step-0 gripper + absolute pose) is mode-independent; only the masks pick what the server uses.
        none = torch.zeros(n_real, dtype=torch.bool)
        action_mask, pose_mask = {
            "joint": (axis_mask, none),      # every actuated dim via the joint head; no pose
            "pose": (gripper, tcp),          # arm via server IK on the TCP pose + gripper via the joint head
            "hybrid": (axis_mask, tcp),      # whole joint head + the TCP pose as the IK posture reference
        }[self.command]
        return {"joint": joint, "pose": pose_abs, "gripper": gripper,
                "action_mask": action_mask, "pose_mask": pose_mask}

    def _prep_views(self, obs: dict[str, Any]) -> torch.Tensor:
        """Server views -> the model input, byte-identical to training: wods ``crop_resize`` to
        ``image_size`` (uint8), then ``/127.5 - 1`` (omniact/trainer.py). Returns (V, 1, H, W, C)."""
        from wods.v2.item import crop_resize

        raw = torch.as_tensor(obs["views"])                            # (V, H, W, C) uint8
        frames = []
        for v in raw:
            frame, _ = crop_resize(v[None], self.image_size, None)     # (1, H, W, C)
            frames.append(frame[0])
        views = torch.stack(frames).float() / 127.5 - 1.0              # (V, H, W, C) in [-1, 1]
        return views[:, None]                                          # (V, 1, H, W, C)

    def _text_embed(self, obs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """(1, L, D), (1, L) bool. Prefer a precomputed ``text_embed``; else encode ``prompt`` with the same
        Qwen path the WODS ``text_embed`` was built with; else a single zero token (text-free)."""
        text = obs.get("text_embed")
        if text is not None:
            text = torch.as_tensor(text).float()
            text = text[None] if text.dim() == 1 else text             # (L, D)
            return text[None], torch.ones(1, text.shape[0], dtype=torch.bool)
        prompt = obs.get("prompt")
        if prompt:
            emb = self._encode_prompt(prompt)                          # (L, D)
            return emb[None], torch.ones(1, emb.shape[0], dtype=torch.bool)
        return torch.zeros(1, 1, self.text_embed_dim), torch.zeros(1, 1, dtype=torch.bool)

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """(L, D_text) via the shared ``qwen_text`` path (raw, no z-score -- OmniActDataset reads
        text_embed raw). Cached per string; the encoder is loaded lazily on first use."""
        if prompt in self._prompt_cache:
            return self._prompt_cache[prompt]
        from ..qwen_text import encode_prompts, load_qwen_text_encoder

        if self._text_encoder is None:
            self._text_encoder, self._tokenizer = load_qwen_text_encoder(device=self.device)
        emb, _mask = encode_prompts(self._text_encoder, self._tokenizer, [prompt],
                                    device=torch.device(self.device))
        emb = emb[0].float().cpu()                                     # (L, D)
        self._prompt_cache[prompt] = emb
        return emb
