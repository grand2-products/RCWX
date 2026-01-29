"""RMVPE F0 extraction model.

Matches original RVC WebUI architecture exactly:
https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/infer/lib/rmvpe.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BiGRU(nn.Module):
    """Bidirectional GRU layer."""

    def __init__(self, input_features: int, hidden_features: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    """Convolutional block with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class ResEncoderBlock(nn.Module):
    """Encoder block: conv blocks followed by optional pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Optional[tuple],
        n_blocks: int = 1,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor):
        for conv in self.conv:
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            return x


class ResDecoderBlock(nn.Module):
    """Decoder block: transposed conv, concat, then conv blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple,
        n_blocks: int = 1,
        momentum: float = 0.01,
    ):
        super().__init__()
        # Output padding depends on stride orientation
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        # After concat with skip connection: 2*out_channels -> out_channels
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x: torch.Tensor, concat_tensor: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        # Handle size mismatch from odd-sized inputs during encoding
        # Pad or crop x to match concat_tensor's spatial dimensions
        if x.shape[2] != concat_tensor.shape[2] or x.shape[3] != concat_tensor.shape[3]:
            diff_h = concat_tensor.shape[2] - x.shape[2]
            diff_w = concat_tensor.shape[3] - x.shape[3]
            x = F.pad(x, (0, diff_w, 0, diff_h))
        x = torch.cat((x, concat_tensor), dim=1)
        for conv in self.conv2:
            x = conv(x)
        return x


class Encoder(nn.Module):
    """Encoder with initial BatchNorm and ResEncoderBlock layers."""

    def __init__(
        self,
        in_channels: int,
        in_size: int,
        n_encoders: int,
        kernel_size: tuple,
        n_blocks: int,
        out_channels: int = 16,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []

        ch_in = in_channels
        ch_out = out_channels
        size = in_size

        for i in range(n_encoders):
            self.layers.append(
                ResEncoderBlock(ch_in, ch_out, kernel_size, n_blocks, momentum=momentum)
            )
            self.latent_channels.append([ch_out, size])
            ch_in = ch_out
            ch_out *= 2
            size //= 2

        self.out_size = size
        self.out_channel = ch_out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, List[torch.Tensor]]:
        concat_tensors = []
        x = self.bn(x)
        for layer in self.layers:
            skip, x = layer(x)
            concat_tensors.append(skip)
        return x, concat_tensors


class Intermediate(nn.Module):
    """Intermediate section with ResEncoderBlocks (no pooling)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_inters: int,
        n_blocks: int,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        # First layer: in_channels -> out_channels
        self.layers.append(
            ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)
        )
        # Remaining layers: out_channels -> out_channels
        for i in range(n_inters - 1):
            self.layers.append(
                ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    """Decoder with ResDecoderBlock layers."""

    def __init__(
        self,
        in_channels: int,
        n_decoders: int,
        stride: tuple,
        n_blocks: int,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders

        ch_in = in_channels
        for i in range(n_decoders):
            ch_out = ch_in // 2
            self.layers.append(
                ResDecoderBlock(ch_in, ch_out, stride, n_blocks, momentum)
            )
            ch_in = ch_out

    def forward(
        self, x: torch.Tensor, concat_tensors: List[torch.Tensor]
    ) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    """Deep U-Net for RMVPE."""

    def __init__(
        self,
        kernel_size: tuple,
        n_blocks: int,
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ):
        super().__init__()

        # Encoder: processes mel spectrogram (128 mel bins)
        self.encoder = Encoder(
            in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels
        )

        # Intermediate: encoder.out_channel/2 -> encoder.out_channel
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )

        # Decoder: uses encoder's output channel count
        self.decoder = Decoder(
            self.encoder.out_channel, en_de_layers, kernel_size, n_blocks
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class E2E(nn.Module):
    """End-to-end network for RMVPE."""

    def __init__(
        self,
        n_blocks: int,
        n_gru: int,
        kernel_size: tuple,
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ):
        super().__init__()
        self.unet = DeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))

        if n_gru > 0:
            self.fc = nn.Sequential(
                BiGRU(3 * 128, 256, n_gru),
                nn.Linear(512, 360),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * 128, 360),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, 1, frames, mel_bins] (transposed in mel2hidden)
        x = self.unet(x)
        x = self.cnn(x)
        # x: [B, 3, frames, mel_bins]
        # Original RVC: x.transpose(1, 2).flatten(-2)
        x = x.transpose(1, 2).flatten(-2)  # [B, frames, 3*128]
        x = self.fc(x)
        return x


class MelSpectrogram(nn.Module):
    """Mel spectrogram extractor for RMVPE.

    Uses librosa-compatible mel filterbank with htk=True to match original RVC.
    """

    def __init__(
        self,
        is_half: bool,
        n_mel_channels: int,
        sampling_rate: int,
        win_length: int,
        hop_length: int,
        n_fft: Optional[int] = None,
        mel_fmin: float = 0,
        mel_fmax: Optional[float] = None,
        clamp: float = 1e-5,
    ):
        super().__init__()
        n_fft = n_fft or win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp
        self.is_half = is_half

        mel_basis = self._mel_basis(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax or sampling_rate / 2
        )
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("hann_window", torch.hann_window(win_length))

    def _mel_basis(self, sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> torch.Tensor:
        """Compute mel filterbank using librosa-compatible method with htk=True."""
        try:
            # Only import the specific function we need (avoids numba issues)
            from librosa.filters import mel as librosa_mel
            # Use librosa for correct mel filterbank (htk=True matches original RVC)
            mel_basis = librosa_mel(
                sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=True
            )
            return torch.from_numpy(mel_basis).float()
        except ImportError:
            # Fallback to manual implementation with area normalization
            # Note: Manual implementation matches librosa output exactly when htk=True
            return self._mel_basis_manual(sr, n_fft, n_mels, fmin, fmax)

    def _mel_basis_manual(self, sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> torch.Tensor:
        """Manual mel filterbank with area normalization (htk scale)."""
        # HTK mel scale
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        mel_fmin = hz_to_mel(fmin)
        mel_fmax = hz_to_mel(fmax)
        mel_points = np.linspace(mel_fmin, mel_fmax, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # FFT bin frequencies
        fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)

        mel_basis = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(n_mels):
            left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
            # Rising slope
            mask_up = (fft_freqs >= left) & (fft_freqs < center)
            mel_basis[i, mask_up] = (fft_freqs[mask_up] - left) / (center - left)
            # Falling slope
            mask_down = (fft_freqs >= center) & (fft_freqs <= right)
            mel_basis[i, mask_down] = (right - fft_freqs[mask_down]) / (right - center)

        # Area normalization (critical for correct mel values)
        enorm = 2.0 / (hz_points[2:n_mels+2] - hz_points[:n_mels])
        mel_basis *= enorm[:, np.newaxis]

        return torch.from_numpy(mel_basis).float()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        padding = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (padding, padding), mode="reflect")
        fft = torch.stft(
            audio, self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.hann_window.to(audio.device), center=False, return_complex=True,
        )
        magnitudes = fft.abs()
        mel = torch.matmul(self.mel_basis.to(audio.device), magnitudes)
        mel = torch.clamp(mel, min=self.clamp)
        return torch.log(mel)


class RMVPE(nn.Module):
    """RMVPE F0 extraction model."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        hop_length: int = 160,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.hop_length = hop_length

        self.mel_extractor = MelSpectrogram(
            is_half=dtype == torch.float16,
            n_mel_channels=128,
            sampling_rate=16000,
            win_length=1024,
            hop_length=hop_length,
            mel_fmin=30,
            mel_fmax=8000,
        )

        self.model = E2E(
            n_blocks=4,
            n_gru=1,  # Checkpoint uses 1-layer GRU
            kernel_size=(2, 2),
            en_de_layers=5,
            inter_layers=4,
            in_channels=1,
            en_out_channels=16,
        )

        if Path(model_path).exists():
            logger.info(f"Loading RMVPE from: {model_path}")
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning(f"RMVPE model not found at: {model_path}")

        self.model.to(device).to(dtype)
        self.model.eval()

        self.register_buffer(
            "cents_mapping",
            20 * torch.arange(360, device=device) + 1997.3794084376191,
        )

    @torch.no_grad()
    def mel2hidden(self, mel: torch.Tensor) -> torch.Tensor:
        # Original RVC: mel.transpose(-1, -2).unsqueeze(1)
        # Input mel: [B, mel_bins, frames] -> [B, 1, frames, mel_bins]
        mel = mel.transpose(-1, -2).unsqueeze(1)
        return self.model(mel)

    @torch.no_grad()
    def decode(self, hidden: torch.Tensor, threshold: float = 0.03) -> torch.Tensor:
        """Decode hidden representation to F0 using vectorized operations.

        Matches original RVC RMVPE implementation:
        1. Find argmax center for each frame
        2. Compute local weighted average of cents in 9-bin window
        3. Apply threshold using max salience across ALL 360 bins (not just window)
        4. Convert cents to Hz, zero out unvoiced frames
        """
        B, T, _ = hidden.shape
        device = hidden.device

        # Find center (argmax) for each frame
        center = torch.argmax(hidden, dim=2)  # [B, T]

        # CRITICAL: Get max salience across ALL 360 bins for threshold check
        # Original RVC: maxx = np.max(salience, axis=1) where salience is (frames, 360)
        max_salience_all = hidden.max(dim=2).values  # [B, T]

        # Create offset indices (-4 to +4 around center)
        offsets = torch.arange(-4, 5, device=device)  # [9]
        # Compute indices with clamping to valid range [0, 359]
        indices = torch.clamp(center.unsqueeze(-1) + offsets, 0, 359)  # [B, T, 9]

        # Gather salience values using advanced indexing
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, T, 9)
        time_idx = torch.arange(T, device=device).view(1, T, 1).expand(B, T, 9)
        salience = hidden[batch_idx, time_idx, indices]  # [B, T, 9]

        # Get cents mapping for the gathered indices
        local_cents = self.cents_mapping[indices]  # [B, T, 9]

        # Compute weighted average of cents
        salience_sum = salience.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        f0_cents = (salience * local_cents).sum(dim=-1) / salience_sum.squeeze(-1)  # [B, T]

        # Apply threshold: zero out cents where max salience (across all 360 bins) is below threshold
        # Original RVC: devided[maxx <= thred] = 0 (before Hz conversion)
        f0_cents = torch.where(max_salience_all > threshold, f0_cents, torch.zeros_like(f0_cents))

        # Convert cents to Hz: f0 = 10 * 2^(cents/1200)
        f0 = 10 * (2 ** (f0_cents / 1200))

        # Zero out unvoiced frames (where cents was 0, f0 = 10)
        # Original RVC: f0[f0 == 10] = 0
        f0 = torch.where(f0_cents > 0, f0, torch.zeros_like(f0))

        return f0.to(self.dtype)

    @torch.no_grad()
    def infer(self, audio: torch.Tensor, threshold: float = 0.03) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        # IMPORTANT: Use float32 for mel extraction (original RVC uses audio.float())
        # float16 STFT can cause numerical instability
        audio = audio.to(self.device).float()
        mel = self.mel_extractor(audio)
        # Model inference can use configured dtype
        hidden = self.mel2hidden(mel.to(self.dtype))
        return self.decode(hidden, threshold)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.infer(audio)
