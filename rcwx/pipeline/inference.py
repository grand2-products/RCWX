"""RVC inference pipeline."""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt

from rcwx.audio.denoise import denoise as denoise_audio, is_deepfilter_available
from rcwx.audio.resample import resample
from rcwx.device import get_device, get_dtype
from rcwx.downloader import get_hubert_path, get_rmvpe_path
from rcwx.models.hubert import HuBERTFeatureExtractor
from rcwx.models.rmvpe import RMVPE
from rcwx.models.synthesizer import SynthesizerLoader

logger = logging.getLogger(__name__)


def highpass_filter(audio: np.ndarray, sr: int = 16000, cutoff: int = 48) -> np.ndarray:
    """Apply high-pass filter to remove DC offset and low-frequency noise.

    Original RVC uses 5th order Butterworth filter with 48Hz cutoff at 16kHz.
    """
    if len(audio) < 100:  # Too short to filter
        return audio
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    b, a = butter(5, normalized_cutoff, btype="high")
    return filtfilt(b, a, audio).astype(np.float32)


class RVCPipeline:
    """
    Complete RVC voice conversion pipeline.

    Integrates HuBERT feature extraction, optional RMVPE F0 extraction,
    FAISS index retrieval, and synthesizer inference.
    """

    def __init__(
        self,
        model_path: str,
        index_path: Optional[str] = None,
        device: str = "auto",
        dtype: str = "float16",
        use_f0: bool = True,
        use_compile: bool = True,
        models_dir: Optional[str] = None,
    ):
        """
        Initialize the RVC pipeline.

        Args:
            model_path: Path to the RVC .pth model
            index_path: Path to FAISS .index file (optional, auto-detected if None)
            device: Device preference (auto, xpu, cuda, cpu)
            dtype: Data type (float16, float32, bfloat16)
            use_f0: Whether to use F0 extraction (if model supports it)
            use_compile: Whether to use torch.compile optimization
            models_dir: Directory containing HuBERT and RMVPE models
        """
        self.model_path = Path(model_path)

        # Auto-detect index file if not provided
        if index_path:
            self.index_path = Path(index_path)
        else:
            # Look for .index file in same directory as model
            index_candidates = list(self.model_path.parent.glob("*.index"))
            self.index_path = index_candidates[0] if index_candidates else None

        # Resolve device and dtype
        self.device = get_device(device)
        self.dtype = get_dtype(self.device, dtype)

        # torch.compile with Triton backend is not supported on Windows
        if use_compile and sys.platform == "win32":
            logger.info("torch.compile disabled on Windows (Triton not supported)")
            use_compile = False
        self.use_compile = use_compile

        # Model directory for HuBERT/RMVPE
        if models_dir:
            self.models_dir = Path(models_dir)
        else:
            self.models_dir = Path.home() / ".cache" / "rcwx" / "models"

        # Components (initialized lazily)
        self.hubert: Optional[HuBERTFeatureExtractor] = None
        self.rmvpe: Optional[RMVPE] = None
        self.synthesizer: Optional[SynthesizerLoader] = None

        # FAISS index components
        self.faiss_index = None
        self.index_features: Optional[np.ndarray] = None  # big_npy

        # Model properties
        self.has_f0: bool = use_f0
        self.sample_rate: int = 40000
        self._loaded: bool = False

    def load(self) -> None:
        """Load all models."""
        if self._loaded:
            return

        logger.info(f"Loading RVC pipeline on {self.device} with {self.dtype}")

        # Load synthesizer first to detect model type
        self.synthesizer = SynthesizerLoader(
            str(self.model_path),
            device=self.device,
            dtype=self.dtype,
            use_compile=self.use_compile,
        )
        self.synthesizer.load()

        # Update properties based on loaded model
        self.sample_rate = self.synthesizer.sample_rate
        self.has_f0 = self.synthesizer.has_f0 and self.has_f0

        # Load HuBERT
        hubert_path = get_hubert_path(self.models_dir)
        self.hubert = HuBERTFeatureExtractor(
            str(hubert_path) if hubert_path.exists() else None,
            device=self.device,
            dtype=self.dtype,
        )

        if self.use_compile:
            logger.info("Compiling HuBERT model...")
            self.hubert.model = torch.compile(self.hubert.model, mode="reduce-overhead")

        # Load RMVPE if F0 is used
        if self.has_f0:
            rmvpe_path = get_rmvpe_path(self.models_dir)
            if rmvpe_path.exists():
                self.rmvpe = RMVPE(
                    str(rmvpe_path),
                    device=self.device,
                    dtype=self.dtype,
                )
                if self.use_compile:
                    logger.info("Compiling RMVPE model...")
                    self.rmvpe.model = torch.compile(self.rmvpe.model, mode="reduce-overhead")
            else:
                logger.warning("RMVPE model not found, F0 extraction disabled")
                self.has_f0 = False

        # Load FAISS index if available
        if self.index_path and self.index_path.exists():
            self._load_faiss_index()

        self._loaded = True
        logger.info("RVC pipeline loaded successfully")

    def _load_faiss_index(self) -> None:
        """Load FAISS index for feature retrieval."""
        try:
            import faiss

            logger.info(f"Loading FAISS index from: {self.index_path}")
            self.faiss_index = faiss.read_index(str(self.index_path))

            # Reconstruct all feature vectors from the index
            # These are used for weighted averaging during retrieval
            self.index_features = self.faiss_index.reconstruct_n(
                0, self.faiss_index.ntotal
            )
            logger.info(
                f"FAISS index loaded: {self.faiss_index.ntotal} vectors, "
                f"dim={self.index_features.shape[1]}"
            )
        except ImportError:
            logger.warning("faiss-cpu not installed, index retrieval disabled")
            self.faiss_index = None
            self.index_features = None
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}")
            self.faiss_index = None
            self.index_features = None

    def _search_index(
        self,
        features: torch.Tensor,
        index_rate: float = 0.5,
    ) -> torch.Tensor:
        """
        Search FAISS index and blend retrieved features with original.

        Args:
            features: HuBERT features [B, T, C]
            index_rate: Blending ratio (0=original only, 1=index only)

        Returns:
            Blended features [B, T, C]
        """
        if self.faiss_index is None or index_rate <= 0:
            return features

        # Convert to numpy for FAISS search
        npy = features[0].cpu().numpy()
        if self.dtype == torch.float16:
            npy = npy.astype(np.float32)

        # Search for k=8 nearest neighbors (original RVC default)
        score, ix = self.faiss_index.search(npy, k=8)

        # Compute inverse squared distance weights
        # Add small epsilon to avoid division by zero
        weight = np.square(1 / (score + 1e-6))
        weight /= weight.sum(axis=1, keepdims=True)

        # Weighted average of retrieved features
        # index_features[ix] has shape [T, k, C]
        # weight has shape [T, k]
        retrieved = np.sum(
            self.index_features[ix] * np.expand_dims(weight, axis=2),
            axis=1,
        )  # [T, C]

        if self.dtype == torch.float16:
            retrieved = retrieved.astype(np.float16)

        # Blend with original features
        retrieved_tensor = torch.from_numpy(retrieved).unsqueeze(0).to(features.device)
        blended = index_rate * retrieved_tensor + (1 - index_rate) * features

        return blended

    def unload(self) -> None:
        """Unload all models to free memory."""
        self.hubert = None
        self.rmvpe = None
        self.synthesizer = None
        self.faiss_index = None
        self.index_features = None
        self._loaded = False

        # Clear CUDA/XPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

    @torch.no_grad()
    def infer(
        self,
        audio: np.ndarray | torch.Tensor,
        input_sr: int = 16000,
        pitch_shift: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.0,
        voice_gate: bool = True,
        denoise: bool = False,
        noise_reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convert voice using the RVC pipeline.

        Args:
            audio: Input audio (1D numpy array or tensor)
            input_sr: Input sample rate (default 16kHz)
            pitch_shift: Pitch shift in semitones
            f0_method: F0 extraction method ("rmvpe" or "none")
            index_rate: FAISS index blending ratio (0=off, 0.5=balanced, 1=index only)
            voice_gate: If True, mute unvoiced segments (reduces background noise)
            denoise: If True, apply spectral gate noise reduction before processing
            noise_reference: Optional noise sample for denoiser (auto-learns if None)

        Returns:
            Converted audio at model sample rate (usually 40kHz)
        """
        if not self._loaded:
            self.load()

        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        # Resample to 16kHz if needed
        if input_sr != 16000:
            audio_np = audio.numpy()
            audio_np = resample(audio_np, input_sr, 16000)
            audio = torch.from_numpy(audio_np).float()

        # Apply noise reduction if enabled
        if denoise:
            audio_np = audio.numpy()
            # Resample noise reference if provided
            if noise_reference is not None and input_sr != 16000:
                noise_reference = resample(noise_reference, input_sr, 16000)
            # Use DeepFilterNet if available, otherwise spectral gate
            audio_np = denoise_audio(
                audio_np,
                sample_rate=16000,
                method="auto",  # DeepFilterNet if available, else spectral gate
                noise_reference=noise_reference,
                device=self.device,
            )
            audio = torch.from_numpy(audio_np).float()
            logger.debug("Applied noise reduction")

        # Apply high-pass filter to remove DC offset and low-frequency noise
        # Original RVC uses 48Hz cutoff Butterworth filter
        audio_np = audio.numpy()
        audio_np = highpass_filter(audio_np, sr=16000, cutoff=48)

        # Add reflection padding (original RVC uses this for edge handling)
        # Use 50ms padding which is 800 samples at 16kHz
        x_pad = 0.05  # 50ms padding
        t_pad = int(16000 * x_pad)  # Input padding samples
        t_pad_tgt = int(self.sample_rate * x_pad)  # Output padding samples
        original_length = len(audio_np)
        audio_np = np.pad(audio_np, (t_pad, t_pad), mode="reflect")

        audio = torch.from_numpy(audio_np).float()

        # Ensure 2D for batch processing
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.device)

        # Debug: input audio stats
        logger.info(f"Input audio: shape={audio.shape} (padded from {original_length}), min={audio.min():.4f}, max={audio.max():.4f}")

        # Determine output dimension and layer based on model version
        # v1 models: layer 9, 256-dim features
        # v2 models: layer 12, 768-dim features (original RVC specification)
        if self.synthesizer.version == 1:
            output_dim = 256
            output_layer = 9
        else:
            output_dim = 768
            output_layer = 12

        # Extract HuBERT features
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            features = self.hubert.extract(audio, output_layer=output_layer, output_dim=output_dim)

        logger.info(f"HuBERT features: shape={features.shape}, min={features.min():.4f}, max={features.max():.4f}")

        # Apply FAISS index retrieval if enabled (before interpolation, like original RVC)
        if index_rate > 0 and self.faiss_index is not None:
            features = self._search_index(features, index_rate)
            logger.info(f"Index retrieval applied: index_rate={index_rate}")

        # Interpolate features to match synthesizer expectation
        # Original RVC: F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        # Uses 'nearest' mode (default) with 2x upscale
        # HuBERT hop=320 @ 16kHz (50fps) -> Synthesizer needs 100fps
        original_frames = features.shape[1]
        features = torch.nn.functional.interpolate(
            features.permute(0, 2, 1),  # [B, T, C] -> [B, C, T]
            scale_factor=2,  # Fixed 2x upscale (matches original RVC)
            mode="nearest",  # Original RVC uses nearest (default mode)
        ).permute(0, 2, 1)  # [B, C, T] -> [B, T, C]
        logger.info(f"Interpolated features: {original_frames} -> {features.shape[1]} frames (2x nearest)")

        # Feature length
        feature_lengths = torch.tensor(
            [features.shape[1]], dtype=torch.long, device=self.device
        )

        # Extract F0 if using F0 model
        pitch = None
        pitchf = None
        if self.has_f0:
            if self.rmvpe is not None and f0_method == "rmvpe":
                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    f0 = self.rmvpe.infer(audio)

                # Apply pitch shift (only to voiced regions where f0 > 0)
                if pitch_shift != 0:
                    f0 = torch.where(f0 > 0, f0 * (2 ** (pitch_shift / 12)), f0)

                # Align F0 length with features
                # RMVPE: hop=160 (100 frames/sec), HuBERT: hop=320 (50 frames/sec)
                # F0 length is approximately 2x feature length
                if f0.shape[1] != features.shape[1]:
                    f0 = torch.nn.functional.interpolate(
                        f0.unsqueeze(1),
                        size=features.shape[1],
                        mode="linear",
                        align_corners=False,
                    ).squeeze(1)

                # pitchf: continuous F0 values for NSF decoder
                pitchf = f0.to(self.dtype)

                # pitch: quantized F0 for pitch embedding (256 bins)
                # Match original RVC WebUI pitch quantization exactly:
                # 1. Convert F0 to mel scale
                # 2. Normalize to 1-255 range (for voiced frames with f0_mel > 0)
                # 3. Set unvoiced/low values to 1 (NOT 0 - original RVC convention)
                f0_mel_min = 1127 * math.log(1 + 50 / 700)    # ~69.07 (50Hz)
                f0_mel_max = 1127 * math.log(1 + 1100 / 700)  # ~942.46 (1100Hz)

                # Convert F0 to mel scale (f0=0 -> f0_mel=0)
                f0_mel = 1127 * torch.log(1 + f0 / 700)

                # Only normalize voiced frames (f0_mel > 0)
                # Original: f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
                voiced_mask = f0_mel > 0
                f0_mel_normalized = torch.where(
                    voiced_mask,
                    (f0_mel - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1,
                    f0_mel  # Keep 0 for unvoiced
                )

                # Clamp to valid range and set low values to 1
                # Original: f0_mel[f0_mel <= 1] = 1; f0_mel[f0_mel > 255] = 255
                pitch = torch.clamp(f0_mel_normalized, 1, 255).round().long()
                logger.info(f"F0: shape={f0.shape}, min={f0.min():.1f}, max={f0.max():.1f}, voiced={voiced_mask.sum().item()}/{f0.numel()}, pitch_range=[{pitch.min().item()}, {pitch.max().item()}]")

                # Store voiced mask for gating (will be used after synthesis)
                voiced_mask_for_gate = voiced_mask.float()  # [B, T]
            else:
                # F0 model but no RMVPE - use pitch=1 (unvoiced marker per RVC convention)
                pitch = torch.ones(features.shape[0], features.shape[1], dtype=torch.long, device=self.device)
                pitchf = torch.zeros(features.shape[0], features.shape[1], dtype=self.dtype, device=self.device)
                voiced_mask_for_gate = None
                logger.info("F0: using unvoiced pitch=1 (no RMVPE)")
        else:
            voiced_mask_for_gate = None

        # Run synthesizer
        logger.info(f"Synthesizer input: features={features.shape}, pitch={pitch.shape if pitch is not None else None}")
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            output = self.synthesizer.infer(
                features,
                feature_lengths,
                pitch=pitch,
                pitchf=pitchf,
            )

        logger.info(f"Synthesizer output: shape={output.shape}, min={output.min():.4f}, max={output.max():.4f}")

        # Apply voice gating to mute unvoiced segments (reduces background noise)
        if voice_gate and voiced_mask_for_gate is not None:
            output_len = output.shape[-1]

            # Upsample voiced mask to match output length
            gate_mask = torch.nn.functional.interpolate(
                voiced_mask_for_gate.unsqueeze(1),  # [B, 1, T_feat]
                size=output_len,
                mode="linear",
                align_corners=False,
            ).squeeze(1)  # [B, T_out]

            # Apply smooth attack/release to avoid clicks (5ms)
            smooth_samples = int(self.sample_rate * 0.005)
            if smooth_samples > 1:
                kernel = torch.ones(1, 1, smooth_samples, device=gate_mask.device) / smooth_samples
                gate_mask = torch.nn.functional.conv1d(
                    gate_mask.unsqueeze(1),
                    kernel,
                    padding=smooth_samples // 2,
                ).squeeze(1)
                # Ensure exact size match after convolution
                if gate_mask.shape[-1] != output_len:
                    gate_mask = gate_mask[..., :output_len]
                gate_mask = torch.clamp(gate_mask, 0, 1)

            # Apply gate
            output = output * gate_mask
            voiced_ratio = gate_mask.mean().item()
            logger.info(f"Voice gate applied: {voiced_ratio*100:.1f}% voiced")

        # Convert to numpy
        output = output.cpu().float().numpy()

        if output.ndim == 2:
            output = output[0]

        # Trim padding from output (match the input padding ratio)
        # t_pad_tgt = output_sr * x_pad where x_pad = 0.05
        t_pad_tgt = int(self.sample_rate * 0.05)
        if len(output) > 2 * t_pad_tgt:
            output = output[t_pad_tgt:-t_pad_tgt]
            logger.info(f"Trimmed {t_pad_tgt} samples from each end")

        # Verify output length
        expected_output_samples = int(original_length * self.sample_rate / 16000)
        length_diff = abs(len(output) - expected_output_samples)

        if length_diff > 400:  # Allow small rounding differences
            logger.warning(
                f"Output length mismatch: got {len(output)}, expected {expected_output_samples} "
                f"(diff={length_diff})"
            )

        logger.info(f"Final output: shape={output.shape}, min={output.min():.4f}, max={output.max():.4f}")
        return output

    def get_info(self) -> dict:
        """Get information about the loaded pipeline."""
        if not self._loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "device": self.device,
            "dtype": str(self.dtype),
            "model_path": str(self.model_path),
            "index_path": str(self.index_path) if self.index_path else None,
            "has_index": self.faiss_index is not None,
            "index_vectors": self.faiss_index.ntotal if self.faiss_index else 0,
            "version": self.synthesizer.version if self.synthesizer else None,
            "has_f0": self.has_f0,
            "sample_rate": self.sample_rate,
            "use_compile": self.use_compile,
        }
