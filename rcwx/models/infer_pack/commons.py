"""Common utilities for RVC models."""

from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """Initialize conv layer weights."""
    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding for 'same' convolution."""
    return int((kernel_size * dilation - dilation) / 2)


def kl_divergence(
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    m_q: torch.Tensor,
    logs_q: torch.Tensor,
) -> torch.Tensor:
    """Calculate KL divergence between two Gaussian distributions."""
    kl = (logs_q - logs_p) - 0.5
    kl += 0.5 * (torch.exp(2.0 * logs_p) + (m_p - m_q) ** 2) * torch.exp(-2.0 * logs_q)
    return kl


def slice_segments(
    x: torch.Tensor,
    ids_str: torch.Tensor,
    segment_size: int = 4,
) -> torch.Tensor:
    """Slice segments from tensor."""
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i].item()
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def slice_segments2(
    x: torch.Tensor,
    ids_str: torch.Tensor,
    segment_size: int = 4,
) -> torch.Tensor:
    """Slice segments from 2D tensor."""
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i].item()
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]
    return ret


def rand_slice_segments(
    x: torch.Tensor,
    x_lengths: Optional[torch.Tensor] = None,
    segment_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random slice segments for training."""
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).long()
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def sequence_mask(
    length: torch.Tensor,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    """Create a sequence mask."""
    if max_length is None:
        max_length = length.max().item()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    n_channels: int,
) -> torch.Tensor:
    """Fused gating operation for WaveNet."""
    n_channels_int = n_channels
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def subsequent_mask(length: int) -> torch.Tensor:
    """Create a causal mask."""
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


def generate_path(
    duration: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Generate alignment path from durations."""
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, dim=-1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).float()
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, (0, 0, 1, 0, 0, 0))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def convert_pad_shape(pad_shape: List[List[int]]) -> List[int]:
    """Convert pad shape from [[l1,r1],[l2,r2],...] to [l_n,r_n,...,l1,r1]."""
    result = []
    for item in reversed(pad_shape):
        result.extend(item)
    return result
