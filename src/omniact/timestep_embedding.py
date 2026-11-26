"""Sinusoidal timestep embeddings for flow matching.

Adapted from diffusers (Apache 2.0 license). Provides:
  - get_timestep_embedding: pure function, 1-D timesteps → sinusoidal features
  - Timesteps: nn.Module wrapper around get_timestep_embedding
  - TimestepEmbedding: learned MLP that projects sinusoidal features to a target dim

Typical usage (following Wan / diffusion transformer convention):
    self.timesteps_proj = Timesteps(freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
    self.time_embedder = TimestepEmbedding(freq_dim, embed_dim)
    ...
    temb = self.time_embedder(self.timesteps_proj(s))   # (B,) -> (B, embed_dim)
"""

import math

import torch
import torch.nn as nn


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings (DDPM-style).

    Args:
        timesteps: 1-D tensor of N scalar timesteps (may be fractional).
        embedding_dim: Dimension of the output embeddings.
        flip_sin_to_cos: If True, output order is [cos, sin] instead of [sin, cos].
        downscale_freq_shift: Controls delta between frequency bands.
        scale: Scaling factor applied before sin/cos.
        max_period: Controls maximum frequency of the embeddings.

    Returns:
        (N, embedding_dim) sinusoidal positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb


class Timesteps(nn.Module):
    """Sinusoidal timestep projection: scalar → frequency features.

    Args:
        num_channels: Output dimension (number of sinusoidal features).
        flip_sin_to_cos: Whether to flip sin/cos order.
        downscale_freq_shift: Frequency shift parameter.
    """

    def __init__(self, num_channels: int = 256, flip_sin_to_cos: bool = True, downscale_freq_shift: float = 0):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )


class TimestepEmbedding(nn.Module):
    """Learned projection: sinusoidal features → target embedding dim.

    Architecture: Linear → SiLU → Linear (matching diffusion transformer convention).

    Args:
        in_channels: Input dimension (from Timesteps output).
        time_embed_dim: Output embedding dimension.
    """

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample
