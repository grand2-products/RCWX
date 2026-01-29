"""Audio resampling utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly


def resample(
    audio: NDArray[np.float32],
    orig_sr: int,
    target_sr: int,
) -> NDArray[np.float32]:
    """
    Resample audio to target sample rate.

    Args:
        audio: Input audio array (1D)
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio

    # Find GCD for efficient resampling
    from math import gcd

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g

    return resample_poly(audio, up, down).astype(np.float32)


def resample_to_16k(
    audio: NDArray[np.float32],
    orig_sr: int,
) -> NDArray[np.float32]:
    """
    Resample audio to 16kHz (RVC internal sample rate).

    Args:
        audio: Input audio array
        orig_sr: Original sample rate

    Returns:
        Audio resampled to 16kHz
    """
    return resample(audio, orig_sr, 16000)


def resample_from_16k(
    audio: NDArray[np.float32],
    target_sr: int,
) -> NDArray[np.float32]:
    """
    Resample audio from 16kHz to target sample rate.

    Args:
        audio: Input audio array at 16kHz
        target_sr: Target sample rate

    Returns:
        Resampled audio
    """
    return resample(audio, 16000, target_sr)
