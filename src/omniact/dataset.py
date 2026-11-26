"""OmniAct WODS dataset — STANDALONE, all embodiments.

Reads a functional WODS dataset where each ``meta/<view>`` shard carries full-trajectory fields and
each ``frame_<start>/<view>`` shard carries a 16-frame video chunk. The per-node **relative** action
(the regression label, in the wods ``action`` field) remains in the metadata shard. Each sample is one
``(key, embodiment)``: the embodiment's three views, conditioned on the window's first frame, with
its relative-action chunk + node geometry at that frame + the static graph. The dataset enumerates
**every** embodiment in the dir (droid, xarm7, …) — there is no embodiment filter or discovery; the
model trains on all of them uniformly, and morphology enters only through the per-sample geometry +
graph.

See docs/omniact_plan.md.

Embodiments have a VARIABLE node count (n_arm+4 in {10,11,12}); every per-node field is RIGHT-ALIGNED into
a fixed ``n_node`` (default MAX_NODES) frame (proximal arm slots left-padded), so the shared suffix
[wrist | gripL,gripR | termL,termR] lands on fixed slots and one FIXED_ACTION_STD / id_embed table serves
all of them. ``node_mask`` marks the real (last N) slots; the model key-masks the rest.

Sample (flat, all per-node fields padded to ``n_node``):
  views       : (V, 1, H, W, C) uint8
  actions     : (T_chunk, n_node) f32  — per-node action / std (label): per-step delta for the arm, but the
                                         gripper drivers re-referenced to chunk step 0 (see _action_targets)
  geometry    : (n_node, P, V, C) f32
  node_id     : (n_node,) int          — arange(n_node); positional identity (id_embed lookup)
  adjacency   : (n_node, n_node) bool  — symmetric graph from node_graph edges (+ self-loops), block bottom-right
  axis_mask   : (n_node,) bool         — actuated (has direction point) vs terminal/padded
  joint_type  : (n_node,) long         — terminal=0, revolute=1, prismatic=2
  node_proprio: (n_node,) f32          — centered radians or decimetres at the input frame
  node_pose   : (n_node, V, POSE_DIM) f32 — per-view TCP SE(3) as rot6d⊕trans at the input frame (raw metres, camera
                                            frame; unposed/padded rows 0, gated by pose_mask in the model)
  node_action_pose: (T_chunk, n_node, POSE_DIM) f32 — TCP pose target, rot6d⊕trans, z-scored; RE-REFERENCED to the
                                            chunk-start world TCP pose (P_now⁻¹·pose) so it is relative, not absolute
                                            (non-TCP rows are masked out of the flow loss in the model)
  node_mask   : (n_node,) bool         — True for the real (non-padded) nodes
  action_mag_mask : (T_chunk, n_node) bool — KEEP mask over real action nodes (axis_mask or TCP; gripper drivers never filtered); if action_mag_mask_sigma
                                         set, nodes moving > that many σ are filtered OUT (False)
  text_embed / text_embed_mask : (L, D_text) f32 / (L,) bool
  key : str ; embodiment : str
"""
from __future__ import annotations

import io

import torch
from loguru import logger
from torch.utils.data import Dataset
import torchvision.transforms as T

from .joint_types import (
    JOINT_TCP, POSE_DIM,
    se3_inverse, se3_mul, se3_to_pose9,
    to_model_joint_units, validate_joint_type,
)
from .fsspec_retry import (
    FsspecRetryConfig,
    close_fsspec_filesystem,
    fsspec_storage_options,
    retry_fsspec_operation_forever,
)
# The fixed node-graph frame. Every embodiment's nodes are ordered [arm(n_arm) | gripL,gripR | TCP | termL,termR];
# n_arm in {6,7,8} (droid/xarm7 = 7) -> 11/12/13 nodes. MAX_NODES is the 8-arm ceiling. Variable-N samples are
# RIGHT-ALIGNED into this frame (proximal arm slots left-padded), so the shared suffix [grip | TCP | term]
# lands on fixed slots across embodiments (see pad_nodes_front / node_mask_for).
MAX_NODES = 13

# Fixed text-token frame: the collate pads (or truncates) to this length instead of the batch max, so the
# token count is constant across batches. Needed for torch.compile.
TEXT_PAD_LEN = 32

# Action normalization: each label is divided by its entry in a per-step scale table, row k-1 = chunk step k
# (k=1..T_chunk). Rows are physical chunk steps, so a chunk shorter than MAX_ACTION_STEPS uses the first T_chunk
# rows and a longer chunk is refused. The values are RMS about fixed origins (joint zero, identity TCP pose), not
# empirical-mean-centred standard deviations. The tables are copied into the model's checkpoint buffers; these
# defaults initialize datasets and new models, and the policy decodes with the model's saved buffers.
MAX_ACTION_STEPS = 31

# Joint scales, (MAX_ACTION_STEPS, MAX_NODES), right-aligned. Arm slots: one pooled per-step scale at every step,
# fit on 620,694 readable full windows from the 626,718-key embgen_ms_1m root, window 32 (ramp_fit_full.log;
# 2026-09-10). Gripper driver slots are in units of the gripper family's travel: each sample multiplies them by
# gripper_travel (see below). The RMS of label / travel over 4,000 all-key training chunks is 0.483 with no clear
# step dependence, so 0.48 at every step gives ~unit RMS and a full open<->close switch is ~ +/-2.1 on every family.
FIXED_ACTION_STD = torch.tensor([
    0.035564, 0.041106, 0.043455, 0.045707, 0.046202, 0.048713, 0.053092, 0.064875,  # arm (pooled)
    0.48, 0.48,                                                                # gripper (x family travel)
    1.0, 0.001, 0.001,                                                        # masked TCP/terminals
]).repeat(MAX_ACTION_STEPS, 1)

# Gripper driver travel ``q_closed - q_open`` (native radians / metres) per gripper family, keyed by driver node
# name. From the replay bundles' node_spec (gripper_qpos_at_closed - gripper_qpos_at_open): identical on every
# sampled embodiment of each family (400/400), and the node names identify the family exactly. Signed, so a full
# close is +1 after dividing on every family (robotiq closes toward +q, rum and linear_4310 toward -q; lite6's two
# slides are mirrored, the left closing toward +q and the right toward -q). lite6_gripper_wide (the real Lite6's
# gripper, served by lite6-deploy) is from its molmospaces node_spec: open [-0.0081, 0.0081], closed [-1e-5, 1e-5].
GRIPPER_TRAVEL = {
    "gripper/left_driver": 0.821178, "gripper/right_driver": 0.821178,       # robotiq_2f85 (revolute)
    "gripper/left_finger": -0.786205, "gripper/right_finger": -0.786205,     # rum (revolute)
    "gripper/tip_left": -0.04695, "gripper/tip_right": -0.04695,             # linear_4310 (prismatic)
    "gripper/gripper_left_finger": 0.00809, "gripper/gripper_right_finger": -0.00809,   # lite6_gripper_wide (prismatic)
}

# TCP pose origin: identity rot6d + zero translation, so no motion normalizes to zero at every chunk step.
POSE_ACTION_ORIGIN = torch.tensor([1., 0., 0., 0., 1., 0., 0., 0., 0.])   # (POSE_DIM,)
# TCP pose scales, (MAX_ACTION_STEPS, MAX_NODES, POSE_DIM). Only the TCP row (third from the end) has a pose target;
# other rows keep neutral, masked placeholders. The TCP row is the RMS of (pose9 - origin) at each chunk step,
# measured over all 626,718 keys, one chunk per key with the training sampler (random chunk start, tail padding
# held); the two halves of the keys agree within 1.7%. The label is a commanded pose already one step ahead of the
# observed chunk-start pose, so it does not vanish at k=1: the previous full-window scale times a (k/T_chunk)^p
# power law gave normalized RMS 2.1 at k=1 on r1/r5 (1.3 on translation) and 0.85-0.88 at k=31.
FIXED_POSE_ACTION_STD = torch.ones(MAX_ACTION_STEPS, MAX_NODES, POSE_DIM)
FIXED_POSE_ACTION_STD[:, MAX_NODES - 3] = torch.tensor([
    [0.015024, 0.054107, 0.047409, 0.054124, 0.014290, 0.049074, 0.014322, 0.014917, 0.016435],  # k=1
    [0.016867, 0.060221, 0.054865, 0.060239, 0.016398, 0.056830, 0.016591, 0.017258, 0.019209],  # k=2
    [0.019507, 0.067184, 0.063393, 0.067241, 0.019163, 0.065500, 0.019280, 0.020018, 0.022445],  # k=3
    [0.023398, 0.074388, 0.072235, 0.074439, 0.022912, 0.074645, 0.022218, 0.023062, 0.025906],  # k=4
    [0.027942, 0.081638, 0.081179, 0.081763, 0.027256, 0.083746, 0.025270, 0.026166, 0.029455],  # k=5
    [0.032461, 0.088808, 0.090014, 0.089045, 0.031920, 0.092733, 0.028407, 0.029342, 0.033018],  # k=6
    [0.037050, 0.095756, 0.098603, 0.096107, 0.036782, 0.101466, 0.031553, 0.032521, 0.036552],  # k=7
    [0.042254, 0.102507, 0.106921, 0.102978, 0.042355, 0.109929, 0.034697, 0.035690, 0.040028],  # k=8
    [0.047321, 0.108915, 0.114956, 0.109489, 0.047565, 0.118110, 0.037804, 0.038851, 0.043434],  # k=9
    [0.052766, 0.115071, 0.122654, 0.115755, 0.053053, 0.125923, 0.040881, 0.041964, 0.046773],  # k=10
    [0.057971, 0.120940, 0.130034, 0.121735, 0.058464, 0.133385, 0.043938, 0.045032, 0.050008],  # k=11
    [0.063206, 0.126469, 0.137087, 0.127359, 0.063846, 0.140541, 0.046928, 0.048048, 0.053179],  # k=12
    [0.068256, 0.131735, 0.143785, 0.132749, 0.069097, 0.147350, 0.049859, 0.051004, 0.056232],  # k=13
    [0.073534, 0.136800, 0.150150, 0.137898, 0.074506, 0.153848, 0.052744, 0.053905, 0.059194],  # k=14
    [0.078620, 0.141578, 0.156207, 0.142788, 0.079747, 0.159999, 0.055564, 0.056756, 0.062079],  # k=15
    [0.083650, 0.146083, 0.161935, 0.147410, 0.084944, 0.165859, 0.058310, 0.059528, 0.064849],  # k=16
    [0.088777, 0.150387, 0.167399, 0.151817, 0.090346, 0.171420, 0.061001, 0.062237, 0.067521],  # k=17
    [0.093652, 0.154455, 0.172570, 0.155998, 0.095413, 0.176640, 0.063625, 0.064859, 0.070103],  # k=18
    [0.098511, 0.158291, 0.177470, 0.159948, 0.100411, 0.181632, 0.066197, 0.067420, 0.072592],  # k=19
    [0.103299, 0.161880, 0.182124, 0.163643, 0.105333, 0.186377, 0.068686, 0.069922, 0.074995],  # k=20
    [0.107816, 0.165247, 0.186531, 0.167144, 0.110076, 0.190857, 0.071099, 0.072342, 0.077298],  # k=21
    [0.112209, 0.168464, 0.190708, 0.170488, 0.114671, 0.195105, 0.073451, 0.074696, 0.079506],  # k=22
    [0.116517, 0.171508, 0.194626, 0.173663, 0.119167, 0.199105, 0.075727, 0.076973, 0.081622],  # k=23
    [0.120615, 0.174348, 0.198323, 0.176639, 0.123434, 0.202847, 0.077932, 0.079177, 0.083657],  # k=24
    [0.124626, 0.176998, 0.201843, 0.179423, 0.127573, 0.206405, 0.080069, 0.081315, 0.085598],  # k=25
    [0.128488, 0.179500, 0.205177, 0.182055, 0.131574, 0.209743, 0.082133, 0.083381, 0.087454],  # k=26
    [0.132289, 0.181901, 0.208338, 0.184613, 0.135532, 0.212900, 0.084128, 0.085380, 0.089223],  # k=27
    [0.135947, 0.184160, 0.211347, 0.187028, 0.139352, 0.215888, 0.086055, 0.087299, 0.090916],  # k=28
    [0.139437, 0.186318, 0.214189, 0.189330, 0.143020, 0.218734, 0.087913, 0.089150, 0.092526],  # k=29
    [0.142815, 0.188334, 0.216884, 0.191516, 0.146586, 0.221407, 0.089703, 0.090925, 0.094050],  # k=30
    [0.146008, 0.190247, 0.219448, 0.193588, 0.149938, 0.223944, 0.091431, 0.092636, 0.095510],  # k=31
])
NORMALIZATION_FIELDS = ("action_std", "pose_action_std")


def default_action_normalization(n_node: int) -> dict[str, torch.Tensor]:
    """Independent, right-aligned per-step scale tables for a fresh dataset or model."""
    if not 1 <= n_node <= MAX_NODES:
        raise ValueError(f"n_node must be in [1, {MAX_NODES}], got {n_node}")
    return {
        "action_std": FIXED_ACTION_STD[:, -n_node:].clone(),
        "pose_action_std": FIXED_POSE_ACTION_STD[:, -n_node:].clone(),
    }


def gripper_drivers(node_names, axis_mask) -> torch.Tensor:
    """(N,) bool: the actuated ``gripper/`` driver nodes, in natural node order."""
    return torch.as_tensor(axis_mask).bool() & torch.tensor([n.startswith("gripper/") for n in node_names])


def gripper_travel(node_names, axis_mask) -> torch.Tensor:
    """(N,) signed native driver travel for the actuated ``gripper/`` nodes (0 elsewhere). Raises on an actuated
    gripper node whose family is not in GRIPPER_TRAVEL rather than silently mis-scaling it."""
    out = torch.zeros(len(node_names))
    for i, (name, driver) in enumerate(zip(node_names, gripper_drivers(node_names, axis_mask).tolist())):
        if driver:
            if name not in GRIPPER_TRAVEL:
                raise KeyError(f"no gripper travel for driver node {name!r}; add its family to GRIPPER_TRAVEL")
            out[i] = GRIPPER_TRAVEL[name]
    return out


def joint_scale_factor(node_names, axis_mask, joint_type, n_node: int) -> torch.Tensor:
    """(n_node,) right-aligned per-sample multiplier on the joint scale table: the gripper family's signed travel in
    model joint units on the gripper driver slots, 1 elsewhere (including padded slots)."""
    travel = to_model_joint_units(gripper_travel(node_names, axis_mask), torch.as_tensor(joint_type))
    travel = pad_nodes_front(travel, n_node, axis=0)
    return torch.where(travel != 0, travel, torch.ones_like(travel))


def chunk_scale(table: torch.Tensor, t_chunk: int) -> torch.Tensor:
    """The first ``t_chunk`` rows of a per-step scale table (row k-1 is chunk step k)."""
    if not 1 <= t_chunk <= table.shape[0]:
        raise ValueError(f"t_chunk must be in [1, {table.shape[0]}] for the per-step scale tables, got {t_chunk}")
    return table[:t_chunk]


def _key_embodiment(key: str) -> str:
    """Embodiment id — the first segment (``gen_<seed>``) of a molmospaces WODS key
    ``gen_<seed>/<task>/<scene>_house_<h>_traj<NN>``."""
    return key.split("/", 1)[0]


def _key_task(key: str) -> str:
    """Task/config name — the second segment of a molmospaces WODS key."""
    parts = key.split("/")
    return parts[1] if len(parts) > 1 else ""


def pad_nodes_front(x: torch.Tensor, n_node: int, axis: int) -> torch.Tensor:
    """Left-pad ``x`` with zeros along ``axis`` up to ``n_node`` (real nodes right-aligned into the frame)."""
    n = x.shape[axis]
    if n == n_node:
        return x
    shape = list(x.shape)
    shape[axis] = n_node - n
    return torch.cat([x.new_zeros(shape), x], dim=axis)


def right_align_adjacency(adj: torch.Tensor, n_node: int) -> torch.Tensor:
    """(N, N) bool adjacency -> (n_node, n_node): the real block bottom-right, padded nodes as self-loops only
    (so shortest_path leaves them at max distance — they are key-masked anyway)."""
    n = adj.shape[0]
    if n == n_node:
        return adj
    out = torch.eye(n_node, dtype=adj.dtype, device=adj.device)
    out[n_node - n:, n_node - n:] = adj
    return out


def node_mask_for(n_node: int, n: int) -> torch.Tensor:
    """(n_node,) bool: the last ``n`` (real) slots True, leading (padded) slots False."""
    m = torch.zeros(n_node, dtype=torch.bool)
    m[n_node - n:] = True
    return m


def _decode_frame_at(raw: bytes, idx: int, size) -> torch.Tensor:
    """Decode ONLY frame ``idx`` from raw mp4 bytes via PyAV seek, then ``crop_resize`` — one frame, not
    the whole clip, since the model conditions only on the window's first frame."""
    import av
    from wods.v2.item import crop_resize

    container = av.open(io.BytesIO(raw), format="mp4")
    stream = container.streams.video[0]
    fps = float(stream.average_rate) or 1.0
    target = idx / fps
    container.seek(int(target / stream.time_base), stream=stream, backward=True)
    out = None
    for f in container.decode(stream):
        if f.time is not None and f.time + 1e-6 >= target:
            out = f.to_ndarray(format="rgb24")
            break
    if out is None:  # seek/pts edge case — fall back to a full decode of this one clip
        container.close()
        container = av.open(io.BytesIO(raw), format="mp4")
        frames = [fr.to_ndarray(format="rgb24") for fr in container.decode(video=0)]
        out = frames[min(idx, len(frames) - 1)]
    container.close()
    frame, _ = crop_resize(torch.from_numpy(out)[None], tuple(size), None)  # (1,H,W,C)
    return frame[0]


class OmniActDataset(Dataset):
    """Every ``(key, embodiment)`` of a functional OmniAct WODS dir. Reads the ``action`` field
    directly (already the per-node relative label — NOT via ``ecfg.extract_action``, which would slice
    it to the absolute width). All embodiments are used; node geometry is assumed on every shard."""

    def __init__(
        self,
        path: str,
        image_size: tuple[int, int] | None = (256, 256),
        window_size: int = 8,
        normalize_actions: bool = True,
        text_embed_dim: int = 3584,
        tasks: list[str] | None = None,
        read_text: bool = True,
        clip_sigma: float | None = None,
        n_node: int = MAX_NODES,
        chunk_frames: int = 16,
        use_image_augmentation: bool = True,
        shared_view_geometry: bool = False,
        action_mag_mask_sigma: float | None = None,
    ):
        from wods.v2.item import list_wods_keys, list_wods_shards

        self.path = path
        self.image_size = image_size
        if not 2 <= window_size <= MAX_ACTION_STEPS + 1:
            raise ValueError(f"window_size must be in [2, {MAX_ACTION_STEPS + 1}], got {window_size}")
        self.window_size = int(window_size)
        # Every sample carries an `action_mag_mask` (T_chunk, n_node) bool KEEP mask over real ACTION nodes (actuated
        # or TCP). When action_mag_mask_sigma is set, nodes whose z-scored relative target exceeds this many standard
        # deviations (moving too much) are FILTERED OUT (False); None => no motion filter (all action nodes True).
        self.action_mag_mask_sigma = None if action_mag_mask_sigma is None else float(action_mag_mask_sigma)
        if chunk_frames < 1:
            raise ValueError(f"chunk_frames must be >= 1, got {chunk_frames}")
        self.chunk_frames = int(chunk_frames)
        self.use_image_augmentation = bool(use_image_augmentation)
        self.shared_view_geometry = bool(shared_view_geometry)
        # Fixed node-graph frame width: every per-node field is right-aligned into n_node slots (default
        # MAX_NODES). Eval passes the checkpoint's n_node so its padding matches the trained id_embed table.
        self.n_node = int(n_node)
        if self.n_node > MAX_NODES:
            raise ValueError(f"n_node={self.n_node} exceeds MAX_NODES={MAX_NODES} (raise MAX_NODES + std)")
        self.text_embed_dim = text_embed_dim
        self.read_text = read_text
        self.clip_sigma = clip_sigma
        self._fsspec_retry_config = FsspecRetryConfig.from_env()

        logger.info(f"Discovering keys at {path} ...")
        self.keys = sorted(list_wods_keys(path))
        if tasks:
            keep = tuple(tasks)
            self.keys = [k for k in self.keys if _key_task(k).startswith(keep)]
            logger.info(f"Task filter {list(tasks)}: {len(self.keys)} keys")
        if not self.keys:
            raise ValueError(f"No keys at {path}" + (f" for tasks {tasks}" if tasks else ""))

        # molmospaces layout: each key is "gen_<seed>/<task>/<scene>_house_<h>_traj<NN>" — the embodiment
        # (gen_<seed>) is the FIRST segment, the task the second, and a key's SHARDS are the camera VIEWS.
        # One sample == one key. ``view_shards`` is the per-key metadata-shard list (the views); a view's
        # video lives in its ``frame_<NN>/<view>`` chunks (produced by molmospaces' video-split step).
        self.view_shards = sorted(
            shard.removeprefix("meta/")
            for shard in list_wods_shards(path, self.keys[0])
            if shard.startswith("meta/")
        )
        self.embodiments = sorted({_key_embodiment(k) for k in self.keys})
        self.items = list(self.keys)
        logger.info(
            f"OmniActDataset: {len(self.items)} (traj,emb) keys, "
            f"{len(self.embodiments)} embodiments {self.embodiments}, views {self.view_shards}"
        )

        self.normalize_actions = bool(normalize_actions)
        for name, value in default_action_normalization(self.n_node).items():
            setattr(self, name, value)

    # ------------------------------------------------------------------

    @property
    def action_scale(self) -> torch.Tensor:
        """(T_chunk, n_node) joint scales for this dataset's chunk length."""
        return chunk_scale(self.action_std, self.window_size - 1)

    @property
    def pose_scale(self) -> torch.Tensor:
        """(T_chunk, n_node, POSE_DIM) TCP pose scales for this dataset's chunk length."""
        return chunk_scale(self.pose_action_std, self.window_size - 1)

    def update_action_std(self, keys: list[str]) -> torch.Tensor:
        """Refit the joint/pose per-step scale tables on training keys, about the fixed origins.

        Reparameterize one sampled chunk per key exactly as training does, divide gripper drivers by their family
        travel, subtract the fixed pose origin, and take the RMS at every chunk step and slot, so dividing by the table yields unit RMS at each step. Only actuated
        joint slots and TCP pose rows contribute; masked slots, slots with no samples, and rows past the chunk
        length retain their existing scales. Fresh-run defaults come from the recorded full-dataset fits (see
        docs/omniact_pipeline.md).
        """
        if not keys:
            raise ValueError("cannot compute action statistics from an empty key list")
        nn, Tc = self.n_node, self.window_size - 1
        action_squared_sum = torch.zeros(Tc, nn, dtype=torch.float64)
        pose_squared_sum = torch.zeros(Tc, nn, POSE_DIM, dtype=torch.float64)
        action_count = torch.zeros(nn, dtype=torch.float64)   # chunks per slot; every chunk fills all Tc steps
        pose_count = torch.zeros(nn, dtype=torch.float64)
        n_read = 0
        for idx, key in enumerate(keys):
            by_shard = self._read_meta(key, idx)
            if by_shard is None:
                continue
            primary = by_shard[self.view_shards[0]]
            graph = primary["node_graph"]
            axis_mask = graph["axis_mask"].bool()
            joint_type = validate_joint_type(graph["joint_type"], axis_mask)
            T = primary["action"].shape[0]
            if T < 2:
                continue
            # Sample a chunk and reparameterize it EXACTLY as __getitem__ does (gripper + TCP pose re-referenced to
            # step 0), so the fitted stats match the targets the model actually regresses.
            t_start = int(torch.randint(0, T - 1, (1,)).item())
            actions, pose = self._action_targets(primary, t_start, axis_mask, joint_type, graph["node_names"])
            joint_rows = pad_nodes_front(axis_mask, nn, axis=0).double()                       # (nn,)
            factor = joint_scale_factor(graph["node_names"], axis_mask, joint_type, nn).double()  # gripper travel
            actions = pad_nodes_front(actions, nn, axis=1).double() / factor
            action_squared_sum += actions.square() * joint_rows
            action_count += joint_rows
            pose = pad_nodes_front(pose, nn, axis=1).double() - POSE_ACTION_ORIGIN
            pose_rows = pad_nodes_front(joint_type == JOINT_TCP, nn, axis=0).double()        # (nn,)
            pose_squared_sum += pose.square() * pose_rows[:, None]
            pose_count += pose_rows
            n_read += 1
        if n_read == 0:
            raise ValueError("no readable items to compute per-node action std")
        action_rms = (action_squared_sum / action_count.clamp(min=1)).sqrt()                   # (Tc, nn)
        pose_rms = (pose_squared_sum / pose_count.clamp(min=1)[:, None]).sqrt()                # (Tc, nn, POSE_DIM)
        self.action_std = self.action_std.clone()
        self.action_std[:Tc] = torch.where(action_count > 0, action_rms, self.action_std[:Tc]).float().clamp(min=1e-3)
        self.pose_action_std = self.pose_action_std.clone()
        self.pose_action_std[:Tc] = torch.where(
            (pose_count > 0)[:, None], pose_rms, self.pose_action_std[:Tc]).float().clamp(min=1e-3)
        logger.info(f"Per-step joint RMS scale from {n_read} training keys, k=1 {self.action_std[0].tolist()}, "
                    f"k={Tc} {self.action_std[Tc - 1].tolist()}")
        logger.info(f"Per-step TCP pose RMS scale over {int((pose_count > 0).sum())} TCP slot(s), "
                    f"k=1 {self.pose_action_std[0, pose_count > 0].tolist()}, "
                    f"k={Tc} {self.pose_action_std[Tc - 1, pose_count > 0].tolist()}")
        return self.action_std

    def _get_fs(self):
        import os

        import fsspec

        pid = os.getpid()
        if not hasattr(self, "_fs") or self._fs is None or getattr(self, "_fs_pid", None) != pid:
            self._fs, _, _ = fsspec.get_fs_token_paths(self.path, storage_options=fsspec_storage_options())
            self._fs_pid = pid
        return self._fs

    def _reset_fs(self) -> None:
        close_fsspec_filesystem(getattr(self, "_fs", None))
        self._fs = None
        self._fs_pid = None

    def __len__(self) -> int:
        return len(self.items)

    def _read_meta(self, key: str, idx: int) -> dict | None:
        from wods.v2.item import read_wods_shard_batched

        shards = self.view_shards            # a key's shards are its camera views
        meta_shards = [f"meta/{shard}" for shard in shards]

        def _read():
            fs = self._get_fs()
            off = dict(
                include_video=False, include_video_raw=False,
                include_timestamps=False, include_latents=False,
                include_points=False, include_depth=False, include_mask=False,
                include_proprio=False, include_target=False,
                include_node_action_geometry=False, fs=fs,
            )
            results = read_wods_shard_batched(
                [self.path] * len(shards), [key] * len(shards), meta_shards,
                include_action=False, include_node_geometry=True, include_node_graph=False,
                include_node_proprio=False, include_text_embed=False, include_camera=False,
                include_node_pose=True, include_node_action_pose=False,
                **off,
            )
            results[0].update(read_wods_shard_batched(
                [self.path], [key], meta_shards[:1],
                include_action=True, include_node_geometry=False, include_node_graph=True,
                include_node_proprio=True,     # per-node centered proprio (precomputed per-spec)
                include_text_embed=self.read_text,
                include_camera=True,           # c2w (view 0): world-frame reference for the relative pose target
                include_node_pose=False, include_node_action_pose=True,
                **off,
            )[0])
            return results

        results = retry_fsspec_operation_forever(
            _read, reset=self._reset_fs, description=f"omniact idx={idx} key={key}",
            config=self._fsspec_retry_config,
        )
        by_shard = dict(zip(shards, results))
        if any("node_geometry" not in by_shard.get(s, {}) for s in shards):
            return None
        return by_shard

    def _read_video_chunks(self, key: str, t_start: int, idx: int):
        from wods.v2.item import read_wods_shard_batched

        chunk, frame = divmod(t_start, self.chunk_frames)
        shards = [f"frame_{chunk * self.chunk_frames:05d}/{view}" for view in self.view_shards]

        def _read():
            return read_wods_shard_batched(
                [self.path] * len(shards), [key] * len(shards), shards,
                include_video=False, include_video_raw=True,
                include_timestamps=False, include_latents=False, include_camera=False,
                include_points=False, include_depth=False, include_mask=False,
                include_proprio=False, include_action=False, include_target=False,
                include_text_embed=False, include_node_geometry=False,
                include_node_action_geometry=False, include_node_graph=False,
                include_node_proprio=False, include_node_pose=False,
                include_node_action_pose=False, fs=self._get_fs(),
            )

        results = retry_fsspec_operation_forever(
            _read, reset=self._reset_fs, description=f"omniact idx={idx} key={key} video chunk",
            config=self._fsspec_retry_config,
        )
        if any("video_raw" not in result for result in results):
            return None
        return [result["video_raw"] for result in results], frame

    def _text(self, primary: dict):
        te = primary.get("text_embed") if self.read_text else None
        if te is not None:
            if te.dim() == 1:
                te = te.unsqueeze(0)
            return te.float(), torch.ones(te.shape[0], dtype=torch.bool)
        return torch.zeros(1, self.text_embed_dim), torch.zeros(1, dtype=torch.bool)

    def _apply_augmentation(self, image: torch.Tensor, rng=None):
            """Apply stochastic augmentations before normalization and resizing."""
            if not self.use_image_augmentation:
                return image
    
            # Color jitter
            color_jitter = T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.05
            )
            image = color_jitter(image)
    
            # Gaussian blur
            gaussian_blur = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))
            image = gaussian_blur(image)
    
            # Random posterize with different bits
            posterize_7 = T.RandomPosterize(bits=7, p=0.2)
            image = posterize_7(image)

            posterize_6 = T.RandomPosterize(bits=6, p=0.2)
            image = posterize_6(image)
    
            posterize_5 = T.RandomPosterize(bits=5, p=0.2)
            image = posterize_5(image)
    
            posterize_4 = T.RandomPosterize(bits=4, p=0.2)
            image = posterize_4(image)
    
            # Random adjust sharpness
            adjust_sharpness = T.RandomAdjustSharpness(sharpness_factor=2, p=0.5)
            image = adjust_sharpness(image)
    
            grayscale = T.RandomGrayscale(p=0.05)
            aug_img = grayscale(image)
    
            return aug_img

    @staticmethod
    def _hold_pad(x: torch.Tensor, length: int, fill: str) -> torch.Tensor:
        """Pad a ``(<=length, …)`` time chunk up to ``length`` along axis 0 (truncate if longer). ``fill='zero'``
        for a relative action (a zero delta = no move past the trajectory end); ``fill='last'`` for an absolute
        state/pose (hold the final frame)."""
        n = x.shape[0]
        if n >= length:
            return x[:length]
        tail = x[-1:].expand(length - n, *x.shape[1:]) if fill == "last" else x.new_zeros(length - n, *x.shape[1:])
        return torch.cat([x, tail], dim=0)

    def _action_targets(self, primary: dict, t_start: int, axis_mask: torch.Tensor,
                        joint_type: torch.Tensor, node_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """The chunk's flow-matching targets in natural node order, both RE-REFERENCED to the chunk's step-0
        state so the model predicts motion relative to where it starts (not absolute world coordinates). Shared
        by ``__getitem__`` and ``update_action_std`` so training and the stats fit reparameterize identically.

        Joint action ``(T_chunk, N)`` — the per-node relative action, left as-is for the arm (already a per-step
        delta) but the GRIPPER driver dims re-referenced to step 0: ``action + node_proprio − node_proprio[t0]``.
        (At eval the arm is driven by IK on the pose, the gripper by the joint head, so only the gripper needs
        to share the pose's chunk-start reference.)

        TCP pose ``(T_chunk, N, POSE_DIM)`` — the world-frame ``node_action_pose`` re-expressed in the chunk-start
        TCP frame: ``P_now⁻¹ · node_action_pose``, with ``P_now = c2w[t0] · node_pose[t0]`` (view 0, camera→world)
        the current world TCP pose the observation reports; non-TCP rows stay zero. Both are in model joint units /
        raw rot6d⊕trans — the caller pads to ``n_node`` and normalizes."""
        Tc = self.window_size - 1
        units = lambda x: to_model_joint_units(x.float(), joint_type)

        actions = self._hold_pad(units(primary["action"][t_start:t_start + Tc]), Tc, "zero")      # (Tc, N)
        proprio = self._hold_pad(units(primary["node_proprio"][t_start:t_start + Tc]), Tc, "last")  # (Tc, N)
        driver = axis_mask & torch.tensor([n.startswith("gripper/") for n in node_names])          # (N,) gripper dims
        actions[:, driver] += proprio[:, driver] - units(primary["node_proprio"][t_start])[driver]

        tcp = joint_type == JOINT_TCP                                                               # (N,) the TCP row
        action_pose = self._hold_pad(primary["node_action_pose"][t_start:t_start + Tc].float(), Tc, "last").clone()  # (Tc,N,3,4)
        p_now = se3_mul(primary["c2w"][t_start].float(), primary["node_pose"][t_start].float()[tcp])  # (n_tcp,3,4) world
        action_pose[:, tcp] = se3_mul(se3_inverse(p_now)[None], action_pose[:, tcp])                # relative to P_now (in place on our copy)
        return actions, se3_to_pose9(action_pose)                                                   # (Tc,N), (Tc,N,POSE_DIM)

    def _action_mag_mask(self, actions_z: torch.Tensor, pose_z: torch.Tensor, axis_mask: torch.Tensor,
                           joint_type: torch.Tensor, node_mask: torch.Tensor, gripper: torch.Tensor) -> torch.Tensor:
        """(T_chunk, n_node) bool KEEP mask — always emitted. True for a real node that carries an action
        (actuated ``axis_mask`` or the ``JOINT_TCP`` pose node) AND, when ``action_mag_mask_sigma`` is set, does NOT
        move too much: its z-scored relative target is within that many σ (``|joint action|`` for actuated nodes,
        the largest ``|channel|`` of the relative TCP pose (rot6d⊕trans) for the pose node — the direct analog of
        the joint scalar). Nodes moving > sigma are FILTERED OUT (False); without a sigma nothing is filtered.

        ``gripper`` (n_node,) marks the gripper driver nodes, which are never filtered: an open<->close switch is
        a full-travel step that is the supervision the gripper needs, not an outlier."""
        tcp = joint_type == JOINT_TCP                                    # (n_node,) the pose node
        has_action = node_mask & (axis_mask | tcp)                      # (n_node,) node carries an action
        if self.action_mag_mask_sigma is None:
            keep = torch.ones_like(actions_z, dtype=torch.bool)         # (T_chunk, n_node) no motion filter
        else:
            mag = torch.where(tcp, pose_z.abs().amax(-1), actions_z.abs())  # (T_chunk, n_node) in σ units
            keep = mag <= self.action_mag_mask_sigma                       # filter OUT nodes that move too much
            keep = keep | gripper[None]                                    # gripper drivers are never σ-filtered
        return keep & has_action                                        # (T_chunk, n_node)

    def __getitem__(self, idx: int) -> dict:
        key = self.items[idx]                # "gen_<seed>/<task>/<scene>_house_<h>_traj<NN>"
        by_shard = self._read_meta(key, idx)
        assert by_shard is not None, f"OmniAct key {key!r} incomplete — fix the data."

        shards = self.view_shards
        primary = by_shard[shards[0]]
        T = primary["action"].shape[0]                    # trajectory length
        if T < 2:
            raise ValueError(f"Trajectory too short: T={T} (key={key})")
        t_start = int(torch.randint(0, T - 1, (1,)).item())

        video_chunks = self._read_video_chunks(key, t_start, idx)
        assert video_chunks is not None, f"OmniAct key {key!r} has a missing video chunk"
        video_raw, frame = video_chunks
        frames = [_decode_frame_at(raw, frame, self.image_size) for raw in video_raw]
        views = torch.stack(frames, dim=0)       # (V, H, W, C) uint8
        views = views.permute(0, 3, 1, 2)        # VCHW
        views = self._apply_augmentation(views)  # VCHW
        views = views.permute(0, 2, 3, 1)[:, None]        # V1HWC

        # node_geometry is per-view (T, N, P, 3); stack the V views -> (T, N, P, V, 3), index t_start.
        geometry = torch.stack([by_shard[s]["node_geometry"] for s in shards], dim=3)[t_start].float()
        if self.shared_view_geometry:
            # Ablation: give every view the SAME node coordinates (view 0's), so the geometry is no longer
            # registered to the pixels of the view it accompanies. Coordinate statistics are untouched -- the
            # only thing removed is the per-view correspondence -- which isolates camera grounding from
            # having geometry at all (--no-geometry removes both).
            geometry = geometry[:, :, :1].expand_as(geometry).contiguous()
        # node_pose is the per-VIEW camera-frame TCP pose SE(3) [R|t] (T, N, 3, 4); stack views -> (T, N, V, 3, 4),
        # index t_start, then rot6d⊕translation -> (N, V, POSE_DIM). RAW metres/rot6d (fourier-encoded in the
        # model, tied to the geometry PE scale); only the TCP node carries a frame, others zero, and the model
        # gates this branch by pose_mask so the padded/unposed rows never contribute.
        node_pose = torch.stack([by_shard[s]["node_pose"] for s in shards], dim=2)[t_start].float()  # (N,V,3,4)
        node_pose = se3_to_pose9(node_pose)                           # (N, V, POSE_DIM)
        graph = primary["node_graph"]                     # static graph + per-node joint semantics
        edges = graph["edges"].long()                     # (E, 2) directed parent→child id pairs (natural order)
        axis_mask = graph["axis_mask"].bool()             # (N,) actuated vs terminal
        joint_type = validate_joint_type(graph["joint_type"], axis_mask)
        N = axis_mask.shape[0]
        adjacency = torch.eye(N, dtype=torch.bool)        # (N, N) symmetric + self-loops
        adjacency[edges[:, 0], edges[:, 1]] = True
        adjacency[edges[:, 1], edges[:, 0]] = True
        # Per-node CENTERED proprio at the input frame — always the precomputed per-spec field
        # (add_transfer scatters raw qpos - pcenter onto nodes; correct for any joint count).
        node_proprio = to_model_joint_units(
            primary["node_proprio"][t_start].float(), joint_type
        )                                                             # centered radians/decimetres

        # The chunk's flow-matching targets, both re-referenced to the chunk-start state (gripper dims + TCP pose;
        # see _action_targets). Natural node order -> right-align, then z-score like every other per-node field.
        actions, action_pose = self._action_targets(primary, t_start, axis_mask, joint_type, graph["node_names"])
        actions = pad_nodes_front(actions, self.n_node, axis=1)             # (T_chunk, n_node)
        gripper = pad_nodes_front(gripper_drivers(graph["node_names"], axis_mask), self.n_node, axis=0)  # (n_node,)
        factor = joint_scale_factor(graph["node_names"], axis_mask, joint_type, self.n_node)   # gripper travel
        action_pose = pad_nodes_front(action_pose, self.n_node, axis=1)     # (T_chunk, n_node, POSE_DIM)
        axis_mask = pad_nodes_front(axis_mask, self.n_node, axis=0)         # (n_node,) padded slots False (reused below)
        joint_type = pad_nodes_front(joint_type, self.n_node, axis=0)       # (n_node,) padded terminal=0 (reused below)
        node_mask = node_mask_for(self.n_node, N)                          # (n_node,) real (last N) slots (reused below)

        # z-score to σ units — the per-(step, slot) joint scale (origin 0) and the per-(step, slot, channel) pose scale
        # about the fixed pose origin. These are the OUTPUT targets when normalize_actions is on, and always the
        # magnitudes the σ mask thresholds.
        actions_z = actions / (self.action_scale * factor)
        pose_z = (action_pose - POSE_ACTION_ORIGIN) / self.pose_scale
        action_mag_mask = self._action_mag_mask(actions_z, pose_z, axis_mask, joint_type, node_mask, gripper)
        if self.normalize_actions:
            actions = actions_z if self.clip_sigma is None else actions_z.clamp(-self.clip_sigma, self.clip_sigma)
            action_pose = pose_z

        text_embed, text_mask = self._text(primary)
        return {
            "views": views,
            "actions": actions,                                       # (T_chunk, n_node) right-aligned
            "geometry": pad_nodes_front(geometry, self.n_node, axis=0),        # (n_node, P, V, 3)
            "node_id": torch.arange(self.n_node),                     # (n_node,) positional → id_embed
            "adjacency": right_align_adjacency(adjacency, self.n_node),        # (n_node, n_node)
            "axis_mask": axis_mask,                                   # (n_node,) padded slots False
            "joint_type": joint_type,                                 # (n_node,) padded terminal=0
            "node_proprio": pad_nodes_front(node_proprio, self.n_node, axis=0),  # (n_node,)
            "node_pose": pad_nodes_front(node_pose, self.n_node, axis=0),       # (n_node, V, POSE_DIM) raw obs pose
            "node_action_pose": action_pose,                          # (T_chunk, n_node, POSE_DIM) z-scored target
            "node_mask": node_mask,                                   # (n_node,) real (last N) slots True
            "action_mag_mask": action_mag_mask,                               # (T_chunk, n_node) keep mask: action nodes minus big movers
            "text_embed": text_embed,
            "text_embed_mask": text_mask,
            "key": key,
            "embodiment": _key_embodiment(key),
        }


_text_trunc_warned = False


def _warn_text_truncated(length: int) -> None:
    """Warn once — a longer instruction than the fixed frame loses its tail tokens."""
    global _text_trunc_warned
    if not _text_trunc_warned:
        _text_trunc_warned = True
        logger.warning(f"text_embed of length {length} truncated to TEXT_PAD_LEN={TEXT_PAD_LEN}; "
                       "raise TEXT_PAD_LEN if this dataset's instructions are longer.")


def collate(batch: list[dict]) -> dict:
    """Stack the flat OmniAct items; zero-pad the text embedding to the fixed ``TEXT_PAD_LEN``."""
    pe, pm = [], []
    for s in batch:
        e, m = s["text_embed"], s["text_embed_mask"]
        pad = TEXT_PAD_LEN - e.shape[0]
        if pad < 0:
            _warn_text_truncated(e.shape[0])
            e, m = e[:TEXT_PAD_LEN], m[:TEXT_PAD_LEN]
        elif pad > 0:
            e = torch.cat([e, e.new_zeros(pad, e.shape[1])], dim=0)
            m = torch.cat([m, m.new_zeros(pad, dtype=torch.bool)], dim=0)
        pe.append(e)
        pm.append(m)
    # Every per-node field is already right-aligned to n_node, so the stacks are uniform (mixed-embodiment
    # batches no longer clash). ``edges`` is dropped — nothing downstream reads it (the model uses adjacency).
    out = {
        "views": torch.stack([s["views"] for s in batch], dim=0),          # (B, V, 1, H, W, C) uint8
        "actions": torch.stack([s["actions"] for s in batch], dim=0),      # (B, T_chunk, n_node)
        "geometry": torch.stack([s["geometry"] for s in batch], dim=0),    # (B, n_node, P, V, C)
        "node_id": torch.stack([s["node_id"] for s in batch], dim=0),      # (B, n_node)
        "adjacency": torch.stack([s["adjacency"] for s in batch], dim=0),  # (B, n_node, n_node)
        "axis_mask": torch.stack([s["axis_mask"] for s in batch], dim=0),  # (B, n_node)
        "joint_type": torch.stack([s["joint_type"] for s in batch], dim=0),  # (B, n_node)
        "node_proprio": torch.stack([s["node_proprio"] for s in batch], dim=0),  # (B, n_node) rad/dm at t_start
        "node_pose": torch.stack([s["node_pose"] for s in batch], dim=0),   # (B, n_node, V, POSE_DIM) obs pose
        "node_action_pose": torch.stack([s["node_action_pose"] for s in batch], dim=0),  # (B, T_chunk, n_node, POSE_DIM)
        "node_mask": torch.stack([s["node_mask"] for s in batch], dim=0),  # (B, n_node) real-node mask
        "action_mag_mask": torch.stack([s["action_mag_mask"] for s in batch], dim=0),  # (B, T_chunk, n_node) keep mask
        "text_embed": torch.stack(pe, dim=0),
        "text_embed_mask": torch.stack(pm, dim=0),
        "key": [s["key"] for s in batch],
        "embodiment": [s["embodiment"] for s in batch],
    }
    return out
