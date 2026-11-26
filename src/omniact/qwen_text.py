"""Qwen text encoding — byte-identical to the WODS precompute path.

Used by both the WODS dataset side (when recomputing text embeds) and
the rollout-policy side (when consuming a fresh prompt at eval).  The
constants and logic here must stay in lockstep with the offline
``add_text_embed_qwen.py`` script that built the WODS ``text_embed``
shards: any change here would put eval-time embeddings out of
distribution relative to training-time ones.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


# Qwen chat-template system message.  Backslash continuation produces
# nine spaces between numbered items (one trailing on the first line,
# eight leading on the next).
# fmt: off
SYSTEM_MESSAGE = "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video."
# fmt: on
CROP_START = 108
MAX_LENGTH = 1000 + CROP_START


def load_qwen_text_encoder(
    text_model: str = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Any, Any]:
    """Load the Qwen text encoder + tokenizer used at precompute time."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        text_model, subfolder="tokenizer", trust_remote_code=True,
    )
    text_encoder = (
        AutoModel.from_pretrained(
            text_model, subfolder="text_encoder",
            dtype=dtype, trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    return text_encoder, tokenizer


@torch.no_grad()
def encode_prompts(
    text_encoder: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    device: torch.device,
    mean: Tensor | None = None,
    std: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Encode a batch of prompts into float32 hidden states.

    Args:
        text_encoder: Qwen encoder (Hunyuan-bundled umt5-xxl by default).
        tokenizer: Matching tokenizer.
        prompts: Natural-language prompts.  Empty strings are replaced
            with a single space (matches the precompute's null guard).
        device: Target device.
        mean / std: Optional per-feature z-score stats (raw division —
            the stored ``std`` is pre-clamped at compute time, matching
            ``WodsWindowDataset._build_sample``'s text-embed branch).

    Returns:
        ``(embed, mask)`` where
            ``embed``: ``(B, L_max, D)`` float32, zero-padded to the
                longest real-token length across the batch.
            ``mask``:  ``(B, L_max)`` bool, True on real tokens.
    """
    if not prompts:
        raise ValueError("encode_prompts requires at least one prompt")

    conversations = [
        [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": p if p else " "},
        ]
        for p in prompts
    ]
    inputs = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    device_type = device.type if isinstance(device, torch.device) else (
        "cuda" if str(device).startswith("cuda") else "cpu"
    )
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    hidden = outputs.hidden_states[-3][:, CROP_START:]
    attention_mask = attention_mask[:, CROP_START:]

    stripped: list[Tensor] = []
    for i in range(hidden.size(0)):
        seq_len = int(attention_mask[i].sum())
        stripped.append(hidden[i, :seq_len])                                     # (L_i, D)

    max_len = max(t.size(0) for t in stripped)
    D = hidden.size(-1)
    result = hidden.new_zeros(len(stripped), max_len, D)
    mask = torch.zeros(len(stripped), max_len, dtype=torch.bool, device=device)
    for i, t in enumerate(stripped):
        result[i, : t.size(0)] = t
        mask[i, : t.size(0)] = True

    result = result.float()
    if mean is not None:
        result = (result - mean) / std
    return result, mask


def pad_text_context(
    embed: Tensor, mask: Tensor, length: int
) -> tuple[Tensor, Tensor]:
    """Truncate / zero-pad text embeddings + mask to a fixed ``length`` along the
    token axis (Wan-2.1's fixed-length zero-padded text context: pad with zeros,
    no attention mask). Operates on the trailing token dim, so it accepts both
    per-sample ``(L, D)`` / ``(L,)`` and batched ``(B, L, D)`` / ``(B, L)``.
    """
    L = embed.shape[-2]
    if L == length:
        return embed, mask
    if L > length:
        return embed[..., :length, :], mask[..., :length]
    pad = length - L
    embed = torch.cat([embed, embed.new_zeros(*embed.shape[:-2], pad, embed.shape[-1])], dim=-2)
    mask = torch.cat([mask, mask.new_zeros(*mask.shape[:-1], pad)], dim=-1)
    return embed, mask
