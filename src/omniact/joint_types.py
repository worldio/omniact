"""Shared node-joint type codes and model-space unit conventions."""

import torch

JOINT_TERMINAL = 0
JOINT_REVOLUTE = 1
JOINT_PRISMATIC = 2
JOINT_TCP = 3
NUM_JOINT_TYPES = 4

# Per-node pose target/observation (node_pose / node_action_pose) representation for the TCP node:
# rot6d (Zhou et al. 2019 — the two rotation columns, Gram-Schmidt'd back to SO(3); continuous, and any
# 6-vector decodes to a valid rotation) plus a 3-D translation.
ROT6D_DIM = 6
POSE_DIM = ROT6D_DIM + 3        # 9

# Raw simulator qpos stays in SI units. Model-facing prismatic proprio/actions use decimetres
# so their numeric scale is comparable to revolute values in radians.
METERS_TO_DECIMETERS = 10.0


def validate_joint_type(joint_type: torch.Tensor, axis_mask: torch.Tensor) -> torch.Tensor:
    """Validate and return terminal/revolute/prismatic/TCP type IDs.

    ``axis_mask`` marks the actuated nodes that carry a direction point — exactly the revolute and
    prismatic joints. Terminals (no joint) and the TCP node (a pose-only frame, ``JOINT_TCP``) are both
    False, so the invariant is ``axis_mask == (revolute | prismatic)``, not ``joint_type != terminal``.
    """
    joint_type = joint_type.long()
    if joint_type.shape != axis_mask.shape:
        raise ValueError(f"joint_type shape {joint_type.shape} != axis_mask shape {axis_mask.shape}")
    if not ((joint_type >= 0) & (joint_type < NUM_JOINT_TYPES)).all():
        raise ValueError(f"invalid joint_type values: {joint_type.tolist()}")
    actuated = (joint_type == JOINT_REVOLUTE) | (joint_type == JOINT_PRISMATIC)
    if not torch.equal(actuated, axis_mask):
        raise ValueError("axis_mask must equal (joint_type is revolute or prismatic)")
    return joint_type


def _scale_prismatic(values, joint_type, scale: float):
    out = values.clone() if torch.is_tensor(values) else values.copy()
    out[..., joint_type == JOINT_PRISMATIC] *= scale
    return out


def to_model_joint_units(values, joint_type):
    """Native radians/metres -> model radians/decimetres."""
    return _scale_prismatic(values, joint_type, METERS_TO_DECIMETERS)


def to_native_joint_units(values, joint_type):
    """Model radians/decimetres -> native radians/metres."""
    return _scale_prismatic(values, joint_type, 1.0 / METERS_TO_DECIMETERS)


def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    """SO(3) rotation matrix ``(..., 3, 3)`` -> rot6d ``(..., 6)``: its first two COLUMNS concatenated
    ``[col0 | col1]`` (Zhou et al. 2019). Inverse of ``rot6d_to_matrix``; the identity -> ``[1,0,0, 0,1,0]``."""
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """rot6d ``(..., 6)`` -> SO(3) rotation matrix ``(..., 3, 3)``: Gram-Schmidt the two 3-vectors into an
    orthonormal basis and place them (+ their cross product) as the matrix COLUMNS. Any 6-vector with
    linearly independent halves decodes to a valid rotation. Inverse of ``matrix_to_rot6d``. (The all-zero
    6-vector — an unposed node — is degenerate and must be masked out, never decoded.)"""
    a1, a2 = rot6d[..., :3], rot6d[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)          # b1,b2,b3 are the columns


def se3_to_pose9(se3: torch.Tensor) -> torch.Tensor:
    """Repack a pose from its WODS SE(3) matrix form ``[R | t]`` ``(..., 3, 4)`` into the model's 9-vector form
    ``(..., POSE_DIM=9)`` = ``rot6d(R) ⊕ t`` (same pose, matrix rotation -> rot6d). An unposed node (all-zero
    ``[R | t]``) maps to the all-zero 9-vector — masked out downstream by pose_mask."""
    return torch.cat([matrix_to_rot6d(se3[..., :3, :3]), se3[..., :3, 3]], dim=-1)


def pose9_to_se3(pose9: torch.Tensor) -> torch.Tensor:
    """Repack a pose from the model's 9-vector form ``(..., POSE_DIM=9)`` = ``rot6d ⊕ t`` back into the WODS
    SE(3) matrix form ``[R | t]`` ``(..., 3, 4)`` (rot6d -> rotation matrix). Inverse of ``se3_to_pose9`` up to
    the Gram-Schmidt projection of the rotation (must not be called on unposed zeros)."""
    R = rot6d_to_matrix(pose9[..., :ROT6D_DIM])                 # (..., 3, 3)
    return torch.cat([R, pose9[..., ROT6D_DIM:, None]], dim=-1)  # (..., 3, 4)


def se3_inverse(se3: torch.Tensor) -> torch.Tensor:
    """Invert a rigid SE(3) transform ``[R | t]`` ``(..., 3, 4)`` -> ``[Rᵀ | -Rᵀ t]``. Broadcasts over leading
    dims. (A rigid inverse — assumes ``R`` is orthonormal, as it is for a pose/camera matrix.)"""
    R, t = se3[..., :3, :3], se3[..., :3, 3]
    Rt = R.transpose(-1, -2)
    return torch.cat([Rt, -(Rt @ t.unsqueeze(-1))], dim=-1)     # (..., 3, 4)


def se3_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compose two SE(3) transforms ``[Ra|ta] ∘ [Rb|tb]`` ``(..., 3, 4)`` -> ``[Ra·Rb | Ra·tb + ta]`` (apply
    ``b`` then ``a``). Leading dims broadcast."""
    Ra, ta = a[..., :3, :3], a[..., :3, 3]
    Rb, tb = b[..., :3, :3], b[..., :3, 3]
    t = ta + (Ra @ tb.unsqueeze(-1)).squeeze(-1)                # (..., 3)
    return torch.cat([Ra @ Rb, t.unsqueeze(-1)], dim=-1)       # (..., 3, 4)


def driver_qpos_in_node_order(driver_links, driver_qpos, qpos_names) -> tuple:
    """Align left/right driver qpos with node order; retain positional order when names have no side."""
    driver_qpos = tuple(driver_qpos)

    def side(name):
        return "left" if "left" in name else ("right" if "right" in name else None)

    link_sides = [side(name) for name in driver_links]
    joint_sides = [side(qpos_names[i]) for i in driver_qpos]
    if all(link_sides) and all(joint_sides) and set(link_sides) == set(joint_sides):
        by_side = dict(zip(joint_sides, driver_qpos))
        return tuple(by_side[s] for s in link_sides)
    return driver_qpos


def node_proprio_from_qpos(qpos, center, n_arm: int, driver_qpos, n_nodes: int) -> torch.Tensor:
    """Scatter centered native qpos onto arm/driver nodes; terminal nodes remain zero."""
    qpos = torch.as_tensor(qpos, dtype=torch.float32)
    centered = qpos - torch.as_tensor(center, dtype=torch.float32, device=qpos.device)
    out = centered.new_zeros(*centered.shape[:-1], n_nodes)
    out[..., :n_arm] = centered[..., :n_arm]
    for node, qpos_index in enumerate(driver_qpos, start=n_arm):
        out[..., node] = centered[..., qpos_index]
    return out
