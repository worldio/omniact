"""Transformer building blocks for OmniActModel: the SigLIP token encoder, the DiT self-attention
block + core transformer, and the width-multiplier param-scaling helpers."""
from __future__ import annotations

import math
from contextlib import nullcontext
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .timestep_embedding import TimestepEmbedding, Timesteps

_HEAD_DIM = 128

def _patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert ``(B, H, W, C)`` images into the patchified shape that
    ``Siglip2VisionEmbeddings.forward`` expects:
    ``(B, num_patches, C · patch_size · patch_size)``.

    Matches :func:`transformers.models.siglip2.image_processing_siglip2.convert_image_to_patches`.
    """
    B, H, W, C = imgs.shape
    assert H % patch_size == 0 and W % patch_size == 0, (
        f"(H, W) = ({H}, {W}) must be divisible by patch_size={patch_size}"
    )
    num_h = H // patch_size
    num_w = W // patch_size
    x = imgs.reshape(B, num_h, patch_size, num_w, patch_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.reshape(B, num_h * num_w, C * patch_size * patch_size)

class SiglipTokenEncoder(nn.Module):
    """Thin wrapper around ``transformers.Siglip2VisionModel``:

    * Resizes ``(B, H, W, C)`` inputs in ``[-1, 1]`` to the model's native
      spatial size (exactly ``num_patches`` after patchify), so we can use a
      constant ``attention_mask`` / ``spatial_shapes`` and skip the
      ``Siglip2ImageProcessor`` at train time.
    * Returns the per-patch ``last_hidden_state`` (no pooling head output).
    """

    def __init__(
        self,
        model_name: str,
        freeze: bool,
        random_init: bool = False,
        test_config: dict | None = None,
    ) -> None:
        super().__init__()
        from transformers import Siglip2VisionConfig, Siglip2VisionModel

        if random_init:
            cfg = Siglip2VisionConfig(**(test_config or {}))
            self.model = Siglip2VisionModel(cfg)
        else:
            self.model = Siglip2VisionModel.from_pretrained(model_name)
        # We only consume ``last_hidden_state`` — drop the MAP pooling head so
        # it doesn't waste compute or get counted as trainable when unfrozen.
        self.model.vision_model.use_head = False
        if hasattr(self.model.vision_model, "head"):
            del self.model.vision_model.head
        self.hidden_dim = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.num_patches = self.model.config.num_patches

        side = int(round(math.sqrt(self.num_patches)))
        assert side * side == self.num_patches, (
            f"SigLIP num_patches={self.num_patches} is not a perfect square"
        )
        self._side_patches = side
        self.native_side = side * self.patch_size  # H=W in pixels

        self.freeze = freeze
        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):  # keep frozen SigLIP in eval
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Args:
            imgs: ``(B, H, W, C)`` in ``[-1, 1]`` — matches SigLIP's expected
                normalisation (mean=0.5, std=0.5 applied to [0, 1] inputs).

        Returns:
            ``(B, num_patches, hidden_dim)`` patch feature tensor.
        """
        B = imgs.shape[0]
        if imgs.shape[1] != self.native_side or imgs.shape[2] != self.native_side:
            x = imgs.permute(0, 3, 1, 2)
            x = F.interpolate(
                x,
                size=(self.native_side, self.native_side),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            imgs = x.permute(0, 2, 3, 1).contiguous()

        pixel_values = _patchify(imgs, self.patch_size)
        device = pixel_values.device
        spatial_shapes = torch.tensor(
            [[self._side_patches, self._side_patches]] * B,
            device=device,
            dtype=torch.long,
        )
        attention_mask = torch.ones(B, self.num_patches, device=device, dtype=torch.long)

        out = self.model(
            pixel_values=pixel_values,
            pixel_attention_mask=attention_mask,
            spatial_shapes=spatial_shapes,
        )
        return out.last_hidden_state

class BaselineSelfAttention(nn.Module):
    """Multi-head self-attention with QK-RMSNorm and an additive attention mask.

    No RoPE (baseline uses learnable pos embeds added to tokens before the
    transformer).  Works with an external ``(B, N, N)`` additive mask where
    ``0`` means attend and ``-inf`` means block.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.attn_dropout = dropout
        self.norm_q = nn.RMSNorm(self.head_dim)
        self.norm_k = nn.RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None,
                kv: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        if kv is not None:                                    # cross-attention: keys/values from another sequence
            _, k, v = self.to_qkv(kv).reshape(B, kv.shape[1], 3, self.num_heads, self.head_dim).unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = self.norm_q(q)
        k = self.norm_k(k)
        mask = attn_mask.unsqueeze(1) if attn_mask is not None else None
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)

class BaselineDiTBlock(nn.Module):
    """DiT block with AdaLN-Zero conditioning on the flow-matching timestep."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = BaselineSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: torch.Tensor | None,
        kv_pre: torch.Tensor | None = None,
        return_pre: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """``kv_pre``: this block's pre-attention hidden for a shared prefix, prepended to the keys/values, so the
        prefix is encoded once and the queries here are only the suffix. ``return_pre`` hands that hidden back."""
        if c.dim() == 2:
            c = c.unsqueeze(1)
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN_modulation(c).chunk(6, dim=-1)

        h = self.norm1(x) * (1 + scale1) + shift1
        pre = h
        x = x + gate1 * self.attn(h, attn_mask, None if kv_pre is None else torch.cat([kv_pre, h], dim=1))

        h = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.ffn(h)

        extras = {
            "gate_attn": gate1.detach().abs().mean(),
            "gate_ffn": gate2.detach().abs().mean(),
        }
        return (x, extras, pre) if return_pre else (x, extras)

class BaselineCoreTransformer(nn.Module):
    """Core bidirectional DiT over the assembled multi-modal token sequence.

    Token sequence layout (per sample):
        [image_0..(V·P-1), text_0..(L-1), proprio_0, action_0..(N-1)]

    All tokens see all other tokens (subject only to text-padding masking).
    The final AdaLN + output projection (here) applies only to the action
    slots; the per-embodiment action-dim linear lives on ``BaselineModel``.
    """

    def __init__(
        self,
        policy_dim: int,
        siglip_dim: int,
        text_embed_dim: int,
        num_blocks: int,
        num_heads: int,
        max_views: int,
        max_text_len: int = 2048,
        max_action_len: int = 64,
        dropout: float = 0.0,
        time_freq_dim: int = 256,
    ) -> None:
        super().__init__()
        self.policy_dim = policy_dim
        self.max_views = max_views
        self.max_text_len = max_text_len
        self.max_action_len = max_action_len

        # Timestep conditioning.
        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0,
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=policy_dim,
        )

        # Modality projections.
        self.image_proj = nn.Linear(siglip_dim, policy_dim)
        # We extract Qwen ``hidden_states[-3]`` at dataset-prep time, so the
        # text embeddings have not passed through Qwen's final RMSNorm.  Apply
        # an RMSNorm here to standardise per-token L2 magnitudes (preserves
        # outlier-channel identity, unlike per-dim z-scoring); mirrors what
        # PaliGemma's internal pre-norm provides for pi0.5.
        self.text_input_norm = nn.RMSNorm(text_embed_dim)
        self.text_proj = nn.Linear(text_embed_dim, policy_dim)

        # Learnable type embeddings (image / text / proprio / action).
        def _param(*shape: int) -> nn.Parameter:
            p = nn.Parameter(torch.empty(*shape))
            nn.init.normal_(p, std=0.02)
            return p

        self.type_image = _param(1, 1, policy_dim)
        self.type_text = _param(1, 1, policy_dim)
        self.type_proprio = _param(1, 1, policy_dim)
        self.type_action = _param(1, 1, policy_dim)

        # Per-view embedding added to image tokens.
        self.view_embed = _param(1, max_views, 1, policy_dim)

        # Learnable position embeddings for text + action tokens.
        self.text_pos_embed = _param(1, max_text_len, policy_dim)
        self.action_pos_embed = _param(1, max_action_len, policy_dim)

        self.blocks = nn.ModuleList([
            BaselineDiTBlock(policy_dim, num_heads, dropout) for _ in range(num_blocks)
        ])

        # Final AdaLN (acts on action slots only in forward()).
        self.final_norm = nn.LayerNorm(policy_dim, elementwise_affine=False)
        self.final_adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(policy_dim, 2 * policy_dim),
        )
        nn.init.zeros_(self.final_adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_adaLN_modulation[-1].bias)

    def forward(
        self,
        img_feats: torch.Tensor,     # (B, V, P, siglip_dim)
        proprio_tok: torch.Tensor,   # (B, policy_dim)
        text_embed: torch.Tensor,    # (B, L, text_embed_dim)
        text_mask: torch.Tensor,     # (B, L) bool, True = valid
        action_tok: torch.Tensor,    # (B, N, policy_dim)
        s: torch.Tensor,             # (B, 1) flow-matching timestep
    ) -> tuple[torch.Tensor, dict]:
        B, V, P, _ = img_feats.shape
        L = text_embed.shape[1]
        N = action_tok.shape[1]
        device = img_feats.device

        assert L <= self.max_text_len, f"text seq len {L} exceeds max_text_len {self.max_text_len}"
        assert N <= self.max_action_len, f"action chunk len {N} exceeds max_action_len {self.max_action_len}"
        assert V <= self.max_views, f"got {V} views but max_views={self.max_views}"

        # Timestep conditioning (shared across all tokens).
        c = self.time_embedder(self.timesteps_proj(s.squeeze(-1)))  # (B, policy_dim)

        # Image tokens.
        img_tok = self.image_proj(img_feats)                       # (B, V, P, D)
        img_tok = img_tok + self.view_embed[:, :V] + self.type_image
        img_tok = img_tok.reshape(B, V * P, -1)

        # Text tokens.
        text_tok = self.text_proj(self.text_input_norm(text_embed)) + self.type_text + self.text_pos_embed[:, :L]

        # Proprio token (single).
        prop_tok = (proprio_tok + self.type_proprio.squeeze(1)).unsqueeze(1)  # (B, 1, D)

        # Action tokens.
        act_tok = action_tok + self.type_action + self.action_pos_embed[:, :N]

        tokens = torch.cat([img_tok, text_tok, prop_tok, act_tok], dim=1)
        total = tokens.shape[1]

        # Block-causal attention mask (pi0.5 style):
        #   block 0 = [image + text]   — prefix (VLM expert)
        #   block 1 = [proprio]        — state block (attends to prefix + self)
        #   block 2 = [action]         — action expert (attends to prefix + proprio + self)
        # Prefix tokens MUST NOT attend to action tokens (which are noisy during
        # training).  Within each block, attention is bidirectional.
        # Ref: openpi pi0.py:125,133,182; pi0.5 paper §III.
        block_id = torch.zeros(total, dtype=torch.long, device=device)
        block_id[V * P + L : V * P + L + 1] = 1      # proprio
        block_id[V * P + L + 1 :] = 2                 # actions
        # can_see[q, k] = True iff query's block >= key's block
        can_see = block_id.unsqueeze(1) >= block_id.unsqueeze(0)  # (total, total)
        # Text-padding: mask out padded text positions as keys for all queries.
        kp_valid = torch.ones(B, total, dtype=torch.bool, device=device)
        kp_valid[:, V * P : V * P + L] = text_mask
        attn_mask = torch.zeros(B, total, total, device=device, dtype=tokens.dtype)
        attn_mask = attn_mask.masked_fill(~can_see.unsqueeze(0), float("-inf"))
        attn_mask = attn_mask.masked_fill(~kp_valid.unsqueeze(1), float("-inf"))

        block_extras: list[dict] = []
        for block in self.blocks:
            tokens, bx = block(tokens, c, attn_mask)
            block_extras.append(bx)

        # Final AdaLN on action slots only.
        action_out = tokens[:, -N:]
        shift, scale = self.final_adaLN_modulation(c).chunk(2, dim=-1)
        action_out = self.final_norm(action_out) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        extras: dict = {}
        if block_extras:
            ga = torch.stack([bx["gate_attn"] for bx in block_extras])
            gf = torch.stack([bx["gate_ffn"] for bx in block_extras])
            extras["gate_attn_mean"] = ga.mean()
            extras["gate_ffn_mean"] = gf.mean()
            extras["gate_attn_first"] = block_extras[0]["gate_attn"]
            extras["gate_attn_last"] = block_extras[-1]["gate_attn"]
            extras["gate_ffn_first"] = block_extras[0]["gate_ffn"]
            extras["gate_ffn_last"] = block_extras[-1]["gate_ffn"]
        return action_out, extras

def parse_param_count(s: str) -> int:
    s = s.strip().upper()
    if s.endswith("B"):
        return int(float(s[:-1]) * 1e9)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1e6)
    if s.endswith("K"):
        return int(float(s[:-1]) * 1e3)
    return int(s)

def _dims_from_multiplier(m: float) -> tuple[int, int, int]:
    policy_heads = max(4, round(m / 512) * 4)
    policy_dim = policy_heads * _HEAD_DIM
    raw_blocks = 147 * policy_dim / (3 * policy_dim + 3456)
    policy_blocks = max(4, round(raw_blocks / 4) * 4)
    return policy_dim, policy_blocks, policy_heads

def _count_core_params(
    policy_dim: int,
    policy_blocks: int,
    policy_heads: int,
    siglip_dim: int,
    text_embed_dim: int,
    max_views: int,
    max_text_len: int,
    max_action_len: int,
) -> int:
    with torch.device("meta"):
        core = BaselineCoreTransformer(
            policy_dim=policy_dim,
            siglip_dim=siglip_dim,
            text_embed_dim=text_embed_dim,
            num_blocks=policy_blocks,
            num_heads=policy_heads,
            max_views=max_views,
            max_text_len=max_text_len,
            max_action_len=max_action_len,
        )
    return sum(p.numel() for p in core.parameters())


class ScaledDims(NamedTuple):
    policy_dim: int
    policy_blocks: int
    policy_heads: int


def scale_transfer_config(
    target_params,
    *,
    siglip_dim: int = 768,
    text_embed_dim: int = 3584,
    max_views: int = 3,
    max_text_len: int = 2048,
    max_action_len: int = 64,
    **kwargs,
) -> ScaledDims:
    """Auto-scale the core transformer to a target param count -> (policy_dim, policy_blocks, policy_heads).
    Extra kwargs are ignored."""
    if isinstance(target_params, str):
        target_params = parse_param_count(target_params)

    def _count(m: float) -> int:
        return _count_core_params(
            *_dims_from_multiplier(m),
            siglip_dim, text_embed_dim, max_views, max_text_len, max_action_len,
        )

    lo, hi = 128.0, 32768.0
    while _count(hi) < target_params:
        hi *= 2
    for _ in range(64):
        mid = (lo + hi) / 2
        if _count(mid) < target_params:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1.0:
            break
    best_m = lo if abs(_count(lo) - target_params) <= abs(_count(hi) - target_params) else hi
    return ScaledDims(*_dims_from_multiplier(best_m))
