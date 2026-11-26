"""OmniAct policy — a morphology-agnostic flow-matching VLA over a robot scene-graph.

Compared with a flow-matching VLA that encodes proprioception as one token:

  * ``proprio_proj`` (one token) -> :class:`NodeEncoder` (N*V tokens): one token per (node, view) =
    the sincos PE of the node's metric anchor points expressed in that camera frame (bands tied to
    the model width) + the node's raw-qpos proprio scalar (fourier -> MLP, shared across views) + a
    per-node ``id_embed`` + joint-type embedding + a per-VIEW embedding SHARED with the image tokens
    (so a node in camera frame v aligns with image-of-view-v). Views as SEPARATE tokens ->
    camera-count-agnostic + attention does
    the cross-view fusion; the axis point is masked for terminal nodes.
  * attention over node + action tokens carries an **edge bias** from the graph's shortest-path
    distance, on every node-bearing pair (node<->node, action->node, action<->action) — the only
    place the kinematic structure enters (Graphormer-style, one learned bias per hop-distance shared
    across heads + relationships). Folded into the existing block-causal additive mask, so
    :class:`BaselineDiTBlock` is reused unchanged.
  * the action is **per-node, per-step** tokens: each (node, chunk-step) of the relative-joint-coordinate
    label ``(B, T, N)`` becomes a token = scalar-proj + a learned ``action_id_embed[node]`` (a
    DISTINCT set from the node id_embed — the action role vs the state role) + a step embedding.
    A SHARED scalar proj/head (1<->D, any N) keeps it morphology-agnostic; the embodiment enters
    only through the node geometry + graph, never a fixed-N projection.

Token layout in the core: ``[image | text | nodes | action]``. Trunk (blocks 0..q) encodes
``[image | text | nodes]`` (marker-free, for the optional invariance/cross-decode of P2); the head
(q+1..L-1) adds the noised action chunk. Flow-matching velocity target, Euler-ODE ``predict``.
The flow time conditions the HEAD only, so one observation encoding serves any number of noise
draws: ``flow_noise_samples`` draws C of them per sample in training (the trunk runs once, the head
C times), and ``predict`` encodes once for the whole ODE instead of once per denoise step.

See docs/omniact_plan.md.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import get_1d_sincos_pos_embed_from_grid

from .blocks import BaselineDiTBlock, SiglipTokenEncoder
from .timestep_embedding import TimestepEmbedding, Timesteps
from .dataset import (   # one-way: dataset doesn't import model
    MAX_NODES, NORMALIZATION_FIELDS, default_action_normalization)
from .joint_types import NUM_JOINT_TYPES, JOINT_TCP, ROT6D_DIM, POSE_DIM


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class OmniActModelConfig:
    policy_dim: int = 768
    policy_blocks: int = 12
    policy_heads: int = 12
    dropout: float = 0.0

    # Trunk / head split. The head -- the last fifth of the blocks, 4 of the 300M model's 24 -- is the only part
    # that sees the noised action, and the part re-run per noise draw and per denoise step. None => that fifth.
    align_start: int | None = None
    align_end: int | None = None

    # Node encoder
    n_node: int = MAX_NODES    # node-graph frame width; sizes id_embed / action_id_embed / action_std. Variable-N
                               # embodiments (10/11/12) are right-aligned into this frame by the dataset.
    n_points: int = 2          # P: anchor points per node (center, axis_tip)
    fourier_bands: int = 8     # FourierPE bands per coordinate
    node_mlp_hidden: int = 512
    max_spd: int = 8           # shortest-path distance cap for the edge bias
    use_geometry: bool = True  # ablation: encode per-node geometry (off => proprio + id_embed only)

    # SigLIP v2 vision backbone
    siglip_model: str = "google/siglip2-base-patch16-naflex"
    freeze_siglip: bool = True
    siglip_random_init: bool = False
    siglip_test_config: dict = field(default_factory=lambda: {
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
        "intermediate_size": 128, "num_patches": 16, "patch_size": 16,
    })

    # Flow matching
    denoise_steps: int = 10
    flow_noise_samples: int = 8   # noise/flow-time draws per sample; the trunk is encoded once and shared
    flow_beta_alpha: float = 1.0
    flow_beta_beta: float = 1.5
    flow_cutoff: float = 0.999

    max_views: int = 3
    max_text_len: int = 2048
    max_action_len: int = 64   # T_chunk cap
    action_horizon: int = 31   # default predict() chunk = window_size-1 (CLI default window 32); CLI overrides
    action_clip_sigma: float | None = None
    text_embed_dim: int = 3584

    def resolve_pq(self) -> tuple[int, int]:
        L = self.policy_blocks
        q = self.align_end if self.align_end is not None else (4 * L) // 5
        p = self.align_start if self.align_start is not None else L // 3
        q = max(0, min(q, L - 2))
        p = max(0, min(p, q))
        return p, q


# ---------------------------------------------------------------------------
# Geometry encoding
# ---------------------------------------------------------------------------


COORD_PE_SCALE = 256.0  # scale metric camera-frame XYZ before the diffusers-style sincos encoding


def fourier_encode(geometry: torch.Tensor, bands: int) -> torch.Tensor:
    """(..., 3) metric camera-frame XYZ -> (..., 3*2*bands) sincos features.

    Uses diffusers' sincos directly. ``get_3d_sincos_pos_embed`` itself can't take our coordinates —
    it builds an *integer grid* internally (``arange`` of the spatial/temporal sizes) — so we call the
    same 1-D building block it composes per axis, ``get_1d_sincos_pos_embed_from_grid`` (geometric
    ``omega = 1/10000**(k/bands)``), on each of the 3 axes and concatenate. Coordinates are scaled by
    256 before encoding; this is a frequency scale for metre-valued XYZ, not a pixel projection or
    normalization to image bounds."""
    *lead, C = geometry.shape
    pos = (geometry.float() * COORD_PE_SCALE).reshape(-1, C)                       # (M, C), metres * 256
    axes = [get_1d_sincos_pos_embed_from_grid(2 * bands, pos[:, c], output_type="pt") for c in range(C)]
    return torch.cat(axes, dim=-1).reshape(*lead, C * 2 * bands).to(geometry.dtype)  # (..., C*2*bands)


class NodeEncoder(nn.Module):
    """geometry + masks/types/IDs + per-node model-space proprio ->
    per-(node, view) tokens (B, N*V, D), node-major (token n*V+v = node n in view v).

    Each token sums (a) the sincos PE of node n's P anchor points (center, axis_tip) expressed in
    view v — bands tied to the model width (diffusers-style, ``dim // (2*P*3)``, so the PE ~fills D)
    then projected to D (ONLY when ``use_geometry``), (b) the per-node raw-qpos proprio scalar
    (fourier -> MLP, shared across views), and (c) a per-node ``id_embed``. Proprio + id_embed are
    UNCONDITIONAL; only (a) is gated. The per-VIEW embedding (SHARED with the image tokens) is added
    in the core. Views are SEPARATE tokens, so the encoder is camera-count-agnostic and attention does
    the cross-view fusion; the axis point is zeroed for terminal nodes."""

    def __init__(self, dim: int, n_node: int, n_points: int,
                 fourier_bands: int = 8, node_mlp_hidden: int = 512,
                 use_geometry: bool = True):
        super().__init__()
        self.n_points = n_points

        self.use_geometry = bool(use_geometry)
        # #frequencies tied to the hidden width (diffusers ties #freqs to embed_dim): each (node,view)
        # token packs P*3 coords, so split dim across them -> bands = dim // (2 * P * 3).
        self.bands = max(1, dim // (2 * n_points * 3))
        if self.use_geometry:
            self.proj = nn.Linear(n_points * 3 * 2 * self.bands, dim)

        self.proprio_bands = int(fourier_bands)      # per-node raw-qpos scalar -> fourier -> MLP
        self.proprio_mlp = nn.Sequential(
            nn.Linear(2 * self.proprio_bands, node_mlp_hidden), nn.GELU(),
            nn.Linear(node_mlp_hidden, dim),
        )

        # TCP-pose conditioning: metric translation -> fourier (like geometry, tied to self.bands) ⊕ rot6d
        # rotation raw (already a trig encoding) -> shared MLP (mirrors proprio_mlp). Gated to pose nodes.
        self.pose_mlp = nn.Sequential(
            nn.Linear(3 * 2 * self.bands + ROT6D_DIM, node_mlp_hidden), nn.GELU(),
            nn.Linear(node_mlp_hidden, dim),
        )

        self.id_embed = nn.Embedding(n_node, dim)
        self.joint_type_embed = nn.Embedding(NUM_JOINT_TYPES, dim)

    def forward(self, geometry: torch.Tensor, axis_mask: torch.Tensor, joint_type: torch.Tensor,
                node_id: torch.Tensor, node_proprio: torch.Tensor,
                node_pose: torch.Tensor, pose_mask: torch.Tensor) -> torch.Tensor:
        # Shards always store P=2 anchor points (center, axis_tip); ``n_points`` sizes ``proj``.
        # Keep the leading points so --n-points 1 ablates the axis tip instead of failing the matmul.
        if geometry.shape[2] > self.n_points:
            geometry = geometry[:, :, : self.n_points]
        B, N, P, V, C = geometry.shape
        if self.use_geometry:
            feat = fourier_encode(geometry, self.bands)           # (B, N, P, V, C*2*bands)
            if P > 1:  # zero the axis_tip point for terminal nodes (axis_mask False) — its tip == center
                keep = axis_mask.view(B, N, 1, 1, 1).to(feat.dtype)
                axis_keep = torch.ones(P, device=feat.device, dtype=feat.dtype)
                axis_keep[1:] = 0.0                                # point >=1 is the axis tip
                gate = axis_keep.view(1, 1, P, 1, 1) + keep * (1 - axis_keep.view(1, 1, P, 1, 1))
                feat = feat * gate                                 # center always kept; direction only if actuated
            feat = feat.permute(0, 1, 3, 2, 4).reshape(B, N * V, P * C * 2 * self.bands)   # (B, N*V, ·) node-major
            tok = self.proj(feat)                                 # (B, N*V, D)
        else:
            tok = geometry.new_zeros(B, N * V, self.id_embed.embedding_dim)   # geometry off: proprio + id only
        pfeat = fourier_encode(node_proprio[..., None].float(), self.proprio_bands)    # (B, N, 2*bands)
        ptok = self.proprio_mlp(pfeat)                            # (B, N, D) per-node proprio, shared over views
        ptok = ptok * axis_mask[..., None].to(ptok.dtype)        # zeroed where there is no proprio scalar (terminals / TCP)
        tok = tok + ptok.unsqueeze(2).expand(B, N, V, -1).reshape(B, N * V, -1)
        # TCP pose: fourier the metric translation (like geometry) ⊕ raw rot6d -> pose_mlp; only pose nodes.
        rot6d, trans = node_pose[..., :ROT6D_DIM], node_pose[..., ROT6D_DIM:]        # (B,N,V,6), (B,N,V,3)
        pose_feat = torch.cat([fourier_encode(trans, self.bands), rot6d], dim=-1)    # (B, N, V, 3*2*bands + 6)
        pgate = pose_mask.view(B, N, 1, 1).to(tok.dtype)
        tok = tok + (self.pose_mlp(pose_feat) * pgate).reshape(B, N * V, -1)
        id_e = self.id_embed(node_id).unsqueeze(2).expand(B, N, V, -1).reshape(B, N * V, -1)
        type_e = self.joint_type_embed(joint_type).unsqueeze(2)
        type_e = type_e.expand(B, N, V, -1).reshape(B, N * V, -1)
        return tok + id_e + type_e                                # per-view embed added in the core


def shortest_path(adjacency: torch.Tensor, max_spd: int) -> torch.Tensor:
    """(B, N, N) bool adjacency (symmetric, self-loops) -> (B, N, N) long shortest-path hop distance,
    capped at ``max_spd`` (unreachable -> max_spd). Min-plus matrix squaring, O(log N) iters."""
    B, N, _ = adjacency.shape
    INF = max_spd + 1
    eye = torch.eye(N, dtype=torch.bool, device=adjacency.device)
    d = torch.where(adjacency & ~eye, 1, INF).to(torch.long)
    d = d.masked_fill(eye.unsqueeze(0), 0)
    for _ in range(max(1, math.ceil(math.log2(N + 1)))):
        d = torch.minimum(d, (d.unsqueeze(3) + d.unsqueeze(1)).amin(dim=2)).clamp(max=INF)
    return d.clamp(max=max_spd)


# ---------------------------------------------------------------------------
# Core transformer (trunk + head, with the node-node edge bias)
# ---------------------------------------------------------------------------


@dataclass
class TrunkContext:
    tokens: torch.Tensor       # (B, V*P + L + N*V, D) trunk output: [image | text | node-view]
    edge_bias: torch.Tensor    # (B, N, N) additive node-node attention bias
    V: int
    P: int
    L: int
    N: int
    text_mask: torch.Tensor    # (B, L)
    node_mask: torch.Tensor    # (B, N) True=real node, False=padding (key-masked -> padding-invariant)
    action_joint_mask: torch.Tensor  # (B, N) True=actuated joint node (joint scalar head; else key-masked)
    action_pose_mask: torch.Tensor   # (B, N) True=TCP/pose node (pose head)


class GraphCoreTransformer(nn.Module):
    """Block-causal DiT over ``[image | text | nodes | action]`` with a learned shortest-path edge
    bias on every node-bearing pair (node<->node, action->node, action<->action) and key-masking for
    padded nodes. Trunk = blocks 0..q on ``[image|text|nodes]``; head = blocks q+1..L-1 with the
    per-node-per-step action chunk appended."""

    def __init__(self, policy_dim, siglip_dim, text_embed_dim, num_blocks, num_heads, max_views,
                 align_start, align_end, max_spd, max_text_len=2048, max_action_len=64, dropout=0.0,
                 time_freq_dim=256):
        super().__init__()
        self.policy_dim = policy_dim
        self.num_blocks = num_blocks
        self.max_views = max_views
        self.max_text_len = max_text_len
        self.max_action_len = max_action_len
        self.q = align_end
        self.align_layers = list(range(align_start, align_end + 1))

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=policy_dim)
        self.trunk_cond = nn.Parameter(torch.zeros(1, policy_dim))   # the trunk's AdaLN conditioning, learned

        self.image_proj = nn.Linear(siglip_dim, policy_dim)
        self.text_input_norm = nn.RMSNorm(text_embed_dim)
        self.text_proj = nn.Linear(text_embed_dim, policy_dim)

        def _param(*shape):
            t = nn.Parameter(torch.empty(*shape)); nn.init.normal_(t, std=0.02); return t

        self.type_image = _param(1, 1, policy_dim)
        self.type_text = _param(1, 1, policy_dim)
        self.type_node = _param(1, 1, policy_dim)
        self.type_action = _param(1, 1, policy_dim)
        self.view_embed = _param(1, max_views, 1, policy_dim)
        self.text_pos_embed = _param(1, max_text_len, policy_dim)
        # shortest-path -> scalar bias (shared across heads); zero-init so it starts as plain attention.
        self.spd_bias = nn.Embedding(max_spd + 1, 1)
        nn.init.zeros_(self.spd_bias.weight)

        self.blocks = nn.ModuleList([BaselineDiTBlock(policy_dim, num_heads, dropout) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(policy_dim, elementwise_affine=False)
        self.final_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(policy_dim, 2 * policy_dim))
        nn.init.zeros_(self.final_adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_adaLN_modulation[-1].bias)

    def edge_bias(self, adjacency: torch.Tensor, max_spd: int) -> torch.Tensor:
        """(B, N, N) per-node-pair attention bias from shortest-path distance; ``_mask`` scatters it
        onto every node-bearing token pair (node + action), shared across the three relationships."""
        spd = shortest_path(adjacency, max_spd)                   # (B, N, N) long
        return self.spd_bias(spd).squeeze(-1)                     # (B, N, N)

    def _mask(self, V, P, L, N, text_mask, n_action, edge_bias, dtype, node_mask,
              action_latent_mask=None):
        """Block-causal additive (B, T, T) mask over [image+text (0) | nodes (1) | action (2)].

        The SPD edge bias lands on EVERY node-bearing pair via a per-token node id (image/text =
        none): node<->node, action->node, and action<->action all come from one gather. node->action
        is killed by the block-causal rule so its bias stays -inf; in the trunk (n_action=0) this
        reduces to node<->node. Padded nodes (``node_mask`` False) are key-masked at BOTH their node
        and per-step action tokens, so real queries never attend to them -> exact padding-invariance."""
        device = text_mask.device
        B = text_mask.shape[0]
        m_ctx = V * P + L + N * V          # node block = one token per (node, view), node-major
        T = m_ctx + n_action
        block = torch.zeros(T, dtype=torch.long, device=device)
        block[V * P + L : m_ctx] = 1   # nodes
        block[m_ctx:] = 2              # action
        can_see = block.unsqueeze(1) >= block.unsqueeze(0)        # (T, T) query block >= key block
        mask = torch.zeros(B, T, T, device=device, dtype=dtype)
        mask = mask.masked_fill(~can_see.unsqueeze(0), float("-inf"))
        Ta = n_action // N if n_action else 0                    # action chunk steps (one action token per node)
        # key_valid[b, j]: may KEY token j be attended to? False = padding (padded text / padded
        # nodes / non-actuated or padded action tokens). Masked keys become -inf for EVERY query.
        key_valid = torch.ones(B, T, dtype=torch.bool, device=device)
        key_valid[:, V * P : V * P + L] = text_mask                    # text: drop padded positions
        key_valid[:, V * P + L : m_ctx] = node_mask.repeat_interleave(V, dim=1)   # padded node-view KEYS
        if n_action:
            assert action_latent_mask is not None and action_latent_mask.shape == node_mask.shape
            key_valid[:, m_ctx:] = action_latent_mask.repeat(1, Ta)   # action tokens with a latent (joint | pose)
        mask = mask.masked_fill(~key_valid.unsqueeze(1), float("-inf"))
        # per-token node id: node tokens 0..N-1; action tokens reuse their node ((t,n) order); image/
        # text = -1. Gather the (B, N, N) bias onto all node-bearing pairs.
        node_of = torch.full((T,), -1, dtype=torch.long, device=device)
        node_of[V * P + L : m_ctx] = torch.arange(N, device=device).repeat_interleave(V)   # node-major
        if n_action:
            node_of[m_ctx:] = torch.arange(N, device=device).repeat(Ta)
        valid = node_of >= 0
        idx = node_of.clamp(min=0)
        bias = edge_bias[:, idx][:, :, idx] * (valid[:, None] & valid[None, :]).to(dtype)
        return mask + bias                                       # node->action / image-text stay -inf / 0

    def encode_trunk(self, img_feats, node_tok, text_embed, text_mask, edge_bias, node_mask,
                     action_joint_mask, action_pose_mask):
        B, V, P, _ = img_feats.shape
        L = text_embed.shape[1]
        assert node_tok.shape[1] % V == 0, (                      # geometry views must match image views
            f"node tokens ({node_tok.shape[1]}) not divisible by V={V}: one node-view token per camera")
        N = node_tok.shape[1] // V                                # node_tok is (B, N*V, D), node-major
        c = self.trunk_cond.expand(B, -1)          # the observation encoding does not depend on the flow time
        img_tok = (self.image_proj(img_feats) + self.view_embed[:, :V] + self.type_image).reshape(B, V * P, -1)
        text_tok = self.text_proj(self.text_input_norm(text_embed)) + self.type_text + self.text_pos_embed[:, :L]
        node_view_emb = self.view_embed[:, :V, 0].repeat(1, N, 1)  # (1, N*V, D) shared view embed (node-major)
        node_tok = node_tok + self.type_node + node_view_emb
        x = torch.cat([img_tok, text_tok, node_tok], dim=1)       # [image | text | node-view]
        mask = self._mask(V, P, L, N, text_mask, 0, edge_bias, x.dtype, node_mask)
        trunk_hidden = {}
        for l in range(self.q + 1):
            x, _ = self.blocks[l](x, c, mask)
            if l in self.align_layers:
                trunk_hidden[l] = x
        return TrunkContext(x, edge_bias, V, P, L, N, text_mask, node_mask,
                            action_joint_mask, action_pose_mask), trunk_hidden

    def flow_cond(self, s):
        """(B, 1) flow time -> (B, D) AdaLN conditioning for the head blocks."""
        return self.time_embedder(self.timesteps_proj(s.squeeze(-1)))

    def decode_action(self, ctx: TrunkContext, action_tok, cond, marker=None, draws=1):
        """Head over [context | action]. ``ctx`` holds B samples, ``action_tok`` holds B*draws rows: the context is
        encoded once per sample -- it never attends to the action and carries no flow-time conditioning -- and only
        the action tokens are duplicated, attending to the shared context at every block."""
        n_act = action_tok.shape[1]   # per-(node, step) action tokens (T*N); scalar/id/view/time by the model
        x = ctx.tokens if marker is None else ctx.tokens + marker.view(1, 1, -1)
        act_tok = action_tok + self.type_action
        action_latent_mask = ctx.action_joint_mask | ctx.action_pose_mask   # joint or pose token carries a latent
        full_mask = self._mask(ctx.V, ctx.P, ctx.L, ctx.N, ctx.text_mask, n_act, ctx.edge_bias, act_tok.dtype,
                               ctx.node_mask, action_latent_mask)
        n_ctx = x.shape[1]
        ctx_mask = full_mask[:, :n_ctx, :n_ctx]                        # context self-attention, B rows
        act_mask = full_mask[:, n_ctx:, :].repeat_interleave(draws, 0)  # action queries over [context | action]
        ctx_cond = self.trunk_cond.expand(x.shape[0], -1)               # the context sees no flow time
        for l in range(self.q + 1, self.num_blocks):
            x, _, pre = self.blocks[l](x, ctx_cond, ctx_mask, return_pre=True)
            act_tok, _ = self.blocks[l](act_tok, cond, act_mask, kv_pre=pre.repeat_interleave(draws, 0))
        out = act_tok
        shift, scale = self.final_adaLN_modulation(cond).chunk(2, dim=-1)
        return self.final_norm(out) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class OmniActModel(nn.Module):
    """Morphology-agnostic OmniAct policy. Shared (not per-embodiment) node encoder + action
    proj/head; the embodiment enters only through the per-sample geometry + graph."""

    def __init__(self, config: OmniActModelConfig):
        super().__init__()
        self.config = config
        self.p, self.q = config.resolve_pq()

        self.siglip = SiglipTokenEncoder(
            model_name=config.siglip_model, freeze=config.freeze_siglip,
            random_init=config.siglip_random_init, test_config=config.siglip_test_config,
        )
        if not config.freeze_siglip:
            try:
                self.siglip.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            except Exception:
                pass

        self.node_encoder = NodeEncoder(
            config.policy_dim, config.n_node, config.n_points,
            fourier_bands=config.fourier_bands, node_mlp_hidden=config.node_mlp_hidden,
            use_geometry=config.use_geometry,
        )

        # Per-(node, step) action tokens (shared, embodiment-agnostic; the scalar relative action, A=1):
        # in-proj(A) + per-node action id (distinct from the node id) + time embedding.

        self.action_id_embed = nn.Embedding(config.n_node, config.policy_dim)
        self.action_time_pos = nn.Parameter(torch.empty(1, config.max_action_len, config.policy_dim))
        nn.init.normal_(self.action_time_pos, std=0.02)

        A = 1
        self.action_in_proj = nn.Linear(A, config.policy_dim)
        self.action_out_head = nn.Linear(config.policy_dim, A)
        nn.init.zeros_(self.action_out_head.weight)
        nn.init.zeros_(self.action_out_head.bias)

        # TCP pose head: shares the action tokens' id_embed / time_pos. The noised node_action_pose latent
        # projects in (direct linear, like action_in_proj); the head regresses rot6d+trans (POSE_DIM).
        self.pose_in_proj = nn.Linear(POSE_DIM, config.policy_dim)
        self.pose_out_head = nn.Linear(config.policy_dim, POSE_DIM)
        nn.init.zeros_(self.pose_out_head.weight)
        nn.init.zeros_(self.pose_out_head.bias)

        self.core = GraphCoreTransformer(
            policy_dim=config.policy_dim, siglip_dim=self.siglip.hidden_dim,
            text_embed_dim=config.text_embed_dim, num_blocks=config.policy_blocks,
            num_heads=config.policy_heads, max_views=config.max_views,
            align_start=self.p, align_end=self.q, max_spd=config.max_spd,
            max_text_len=config.max_text_len, max_action_len=config.max_action_len, dropout=config.dropout,
        )

        # Carrier buffers for dataset normalization / policy denormalization; never read in forward/predict.
        # Stored per-step scale tables mean changing fresh-run defaults cannot change a checkpoint's decoded actions.
        for name, value in default_action_normalization(config.n_node).items():
            self.register_buffer(name, value)

    def _sample_flow_s(self, B, device):
        beta = torch.distributions.Beta(
            torch.tensor(self.config.flow_beta_alpha, device=device),
            torch.tensor(self.config.flow_beta_beta, device=device))
        return self.config.flow_cutoff * beta.sample((B, 1))

    def _img_feats(self, views):
        B, V = views.shape[:2]
        imgs = views[:, :, 0].flatten(0, 1)                       # (B*V, H, W, C) at t0
        cm = torch.no_grad() if self.config.freeze_siglip else nullcontext()
        with cm:
            feats = self.siglip(imgs)
        img = feats.reshape(B, V, feats.shape[1], self.siglip.hidden_dim)
        return img.detach() if self.config.freeze_siglip else img

    def _node_mask(self, batch):
        """(B, N) bool — True for real nodes. All-real when the batch carries no padding mask (the
        same-N case); set per-sample by the collate when padding a mixed-N batch up to max-N."""
        nm = batch.get("node_mask")
        if nm is None:
            B, N = batch["node_id"].shape
            nm = torch.ones(B, N, dtype=torch.bool, device=batch["node_id"].device)
        return nm

    def action_joint_mask(self, batch):
        """Actuated real nodes (revolute/prismatic); the joint scalar head predicts these."""
        return batch["axis_mask"] & self._node_mask(batch)

    def action_pose_mask(self, batch):
        """TCP/pose nodes (joint_type == JOINT_TCP); the pose head predicts these. Disjoint from the joint mask."""
        return (batch["joint_type"] == JOINT_TCP) & self._node_mask(batch)

    def encode(self, batch):
        """Observation (image + text + node graph, incl. the TCP node_pose) -> (TrunkContext, trunk hiddens)."""
        img = self._img_feats(batch["views"])
        node_tok = self.node_encoder(
            batch["geometry"].float(), batch["axis_mask"], batch["joint_type"],
            batch["node_id"], batch["node_proprio"],
            batch["node_pose"], self.action_pose_mask(batch))
        edge_bias = self.core.edge_bias(batch["adjacency"], self.config.max_spd)
        return self.core.encode_trunk(
            img, node_tok, batch["text_embed"], batch["text_embed_mask"], edge_bias,
            self._node_mask(batch), self.action_joint_mask(batch), self.action_pose_mask(batch))

    def _action_latent_tokens(self, joint_latent, pose_latent, joint_mask, pose_mask) -> torch.Tensor:
        """Per-(node, step) action tokens (B, T*N, D), ordered (t, n) = index t*N+n. Each node carries either the
        noised joint scalar (kept by ``joint_mask``) or the noised TCP pose (kept by ``pose_mask``) — the masks
        are disjoint — plus the shared per-node ``action_id_embed`` and the step's time embed. Masks (B, T*N, 1)."""
        B, T, N = joint_latent.shape[:3]
        dev = joint_latent.device
        n_idx = torch.arange(N, device=dev).repeat(T)                        # (T*N,) node id per token
        t_idx = torch.arange(T, device=dev).repeat_interleave(N)             # (T*N,) chunk step per token
        tok = (self.action_in_proj(joint_latent.reshape(B, T * N, 1)) * joint_mask
               + self.pose_in_proj(pose_latent.reshape(B, T * N, POSE_DIM)) * pose_mask)
        return tok + self.action_id_embed(n_idx) + self.action_time_pos[:, t_idx]

    @staticmethod
    def _flow(target, mask, s_b):
        """Rectified-flow noising: return (noised latent, velocity target), both masked to ``mask``."""
        target = target * mask
        eps = torch.randn_like(target) * mask
        return (1 - s_b) * eps + s_b * target, target - eps

    def decode(self, ctx, target_joint, target_pose, s, marker=None, draws=1):
        """Flow-match the per-node joint action ``(B, T, N)`` and the TCP pose ``(B, T, N, POSE_DIM)`` from
        ``ctx`` at flow time ``s``. Joint (actuated) nodes and pose (TCP) nodes are disjoint. ``ctx`` holds one row
        per sample; the targets hold ``draws`` rows per sample, sharing their sample's context."""
        B, T, N = target_joint.shape
        joint_mask_grid = ctx.action_joint_mask.repeat_interleave(draws, 0)[:, None, :, None].to(target_joint.dtype)
        pose_mask_grid = ctx.action_pose_mask.repeat_interleave(draws, 0)[:, None, :, None].to(target_joint.dtype)
        with torch.no_grad():
            s_b = s.view(B, 1, 1, 1)
            joint_noised, v_joint = self._flow(target_joint[..., None], joint_mask_grid, s_b)   # (B, T, N, 1)
            pose_noised, v_pose = self._flow(target_pose, pose_mask_grid, s_b)                  # (B, T, N, POSE_DIM)
        joint_mask_flat = joint_mask_grid.expand(B, T, N, 1).reshape(B, T * N, 1)
        pose_mask_flat = pose_mask_grid.expand(B, T, N, 1).reshape(B, T * N, 1)
        out = self.core.decode_action(
            ctx, self._action_latent_tokens(joint_noised, pose_noised, joint_mask_flat, pose_mask_flat),
            self.core.flow_cond(s), marker, draws)
        return {"v_pred":      self.action_out_head(out).reshape(B, T, N, 1) * joint_mask_grid, "v_target": v_joint,
                "v_pred_pose": self.pose_out_head(out).reshape(B, T, N, POSE_DIM) * pose_mask_grid, "v_target_pose": v_pose}

    def forward(self, batch: dict) -> dict:
        """P1: encode the sample's graph, self-decode its action target via flow matching.
        (P2 invariance/cross-decode reuse ``encode``/``decode`` with paired batches + markers.)"""
        c = max(1, int(self.config.flow_noise_samples))
        s = self._sample_flow_s(batch["views"].shape[0] * c, batch["views"].device)
        ctx, trunk_hidden = self.encode(batch)                       # once per sample, whatever c is
        out = self.decode(ctx, batch["actions"].repeat_interleave(c, 0),
                          batch["node_action_pose"].repeat_interleave(c, 0), s, draws=c)
        out["trunk_hidden"] = trunk_hidden
        return out

    @staticmethod
    def action_loss(v_pred, v_target, node_mask=None):
        """Flow MSE over ``(B, T, N, ...)``. ``node_mask`` is a per-(node, step) ``(B, T, N)`` keep mask (e.g. the
        dataset's ``action_mag_mask`` split to one head — actuated or TCP, minus big movers); masked entries are
        excluded, broadcast over the trailing (view/feature) dims. ``None`` => plain MSE."""
        if node_mask is None:
            return F.mse_loss(v_pred, v_target)
        m = node_mask
        while m.dim() < v_pred.dim():
            m = m.unsqueeze(-1)                                 # broadcast over any trailing (view/feature) dims
        m = m.to(v_pred.dtype).expand_as(v_pred)
        denom = m.sum().clamp(min=1.0)
        return ((v_pred - v_target) ** 2 * m).sum() / denom

    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, batch: dict, n_steps: int | None = None):
        """Euler-ODE denoise to the joint-action chunk ``(B, T, N)`` and the TCP pose ``(B, T, N, POSE_DIM)``."""
        device = batch["views"].device
        B = batch["views"].shape[0]
        T = n_steps if n_steps is not None else self.config.action_horizon
        N = batch["geometry"].shape[1]            # padded batch N (not config.n_node)
        img = self._img_feats(batch["views"])
        joint_mask = self.action_joint_mask(batch)          # (B, N)
        pose_mask = self.action_pose_mask(batch)            # (B, N)
        node_tok = self.node_encoder(
            batch["geometry"].float(), batch["axis_mask"], batch["joint_type"],
            batch["node_id"], batch["node_proprio"], batch["node_pose"], pose_mask)
        edge_bias = self.core.edge_bias(batch["adjacency"], self.config.max_spd)
        node_mask = self._node_mask(batch)
        joint_mask_grid = joint_mask[:, None, :, None].to(node_tok.dtype)     # (B, 1, N, 1)
        pose_mask_grid = pose_mask[:, None, :, None].to(node_tok.dtype)
        joint_mask_flat = joint_mask_grid.expand(B, T, N, 1).reshape(B, T * N, 1)
        pose_mask_flat = pose_mask_grid.expand(B, T, N, 1).reshape(B, T * N, 1)
        num = self.config.denoise_steps
        dt = 1.0 / num
        joint = torch.randn(B, T, N, 1, device=device) * joint_mask_grid
        pose = torch.randn(B, T, N, POSE_DIM, device=device) * pose_mask_grid
        ctx, _ = self.core.encode_trunk(                              # time-independent: encode once, not per step
            img, node_tok, batch["text_embed"], batch["text_embed_mask"], edge_bias,
            node_mask, joint_mask, pose_mask)
        for i in range(num):
            s = torch.full((B, 1), i * dt, device=device)
            out = self.core.decode_action(ctx, self._action_latent_tokens(joint, pose, joint_mask_flat, pose_mask_flat),
                                          self.core.flow_cond(s))
            joint = (joint + self.action_out_head(out).reshape(B, T, N, 1) * dt) * joint_mask_grid
            pose = (pose + self.pose_out_head(out).reshape(B, T, N, POSE_DIM) * dt) * pose_mask_grid
        return joint.reshape(B, T, N), pose        # (B, T, N) joint action, (B, T, N, POSE_DIM) TCP pose

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def checkpoint_config(ckpt: dict) -> dict:
    """The model config dict stored in a training checkpoint. Checkpoints written before the package was renamed
    from ``transfer_query`` to ``omniact`` store it under ``transfer_query_config``."""
    return ckpt["omniact_config"] if "omniact_config" in ckpt else ckpt["transfer_query_config"]


def load_omniact_model(checkpoint=None, device="cpu", *, denoise_steps=None, random_config=None):
    """Load a trained ``OmniActModel`` + its per-slot denorm stats from a checkpoint.

    ``checkpoint`` is a local path or fsspec URL to a training ``.pt`` (carrying ``omniact_config`` +
    ``model_state_dict``); ``None`` + ``random_config`` builds a random-init model (smoke only). Returns
    ``(model.eval() on device, stats, global_step)`` with ``stats`` carrying the joint denorm (``action_mean``
    0, ``action_std`` the per-(step, slot) scale table) and the TCP pose scale table ``pose_action_std`` (about the
    fixed ``POSE_ACTION_ORIGIN``). The config and complete model state are loaded strictly.
    Joint denorm is identical for every embodiment (mean-0 relative coords; per-slot scale buffers)."""
    import os

    dev = torch.device(device)
    if checkpoint is not None:
        if os.path.exists(str(checkpoint)):
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        else:
            import fsspec
            with fsspec.open(checkpoint, "rb") as f:
                ckpt = torch.load(f, map_location="cpu", weights_only=False)
        cfg = OmniActModelConfig(**checkpoint_config(ckpt))
        if denoise_steps is not None:
            cfg.denoise_steps = denoise_steps
        model = OmniActModel(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        global_step = ckpt.get("global_step")
    else:
        cfg = random_config or OmniActModelConfig(siglip_random_init=True, denoise_steps=denoise_steps or 10)
        model = OmniActModel(cfg)
        global_step = None
    stats = {"action_mean": 0.0, **{name: getattr(model, name).to(dev) for name in NORMALIZATION_FIELDS}}
    return model.to(dev).eval(), stats, global_step
