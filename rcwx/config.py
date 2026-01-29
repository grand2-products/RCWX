"""Configuration management with JSON persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


def _default_models_dir() -> Path:
    return Path.home() / ".cache" / "rcwx" / "models"


@dataclass
class AudioConfig:
    """Audio configuration."""

    input_device_name: Optional[str] = None  # Device name (more stable than index)
    output_device_name: Optional[str] = None
    sample_rate: int = 16000
    output_sample_rate: int = 48000
    chunk_sec: float = 0.35  # RMVPE requires >= 0.32 sec
    crossfade_sec: float = 0.05
    input_gain_db: float = 0.0  # Input gain in dB


@dataclass
class DenoiseConfig:
    """Noise cancellation configuration."""

    enabled: bool = False
    method: str = "auto"  # auto, deepfilter, spectral, off
    # Spectral gate parameters (used when method=spectral)
    threshold_db: float = 6.0
    reduction_db: float = -24.0


@dataclass
class InferenceConfig:
    """Inference configuration."""

    pitch_shift: int = 0  # semitones
    use_f0: bool = True
    use_index: bool = False
    index_ratio: float = 0.5
    use_compile: bool = True
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)


@dataclass
class RCWXConfig:
    """Main configuration for RCWX."""

    models_dir: str = field(default_factory=lambda: str(_default_models_dir()))
    last_model_path: Optional[str] = None
    device: str = "auto"  # auto, xpu, cuda, cpu
    dtype: str = "float16"  # float16, float32, bfloat16

    audio: AudioConfig = field(default_factory=AudioConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> RCWXConfig:
        """Load configuration from JSON file."""
        if path is None:
            path = cls.default_path()

        if not path.exists():
            return cls()

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        audio_data = data.pop("audio", {})
        inference_data = data.pop("inference", {})

        # Migrate old config keys
        if "input_device" in audio_data:
            audio_data.pop("input_device")  # Remove old key (was int, now using name)
        if "output_device" in audio_data:
            audio_data.pop("output_device")  # Remove old key (was int, now using name)

        # Handle nested denoise config
        denoise_data = inference_data.pop("denoise", {})
        denoise_config = DenoiseConfig(**denoise_data) if denoise_data else DenoiseConfig()

        return cls(
            audio=AudioConfig(**audio_data),
            inference=InferenceConfig(denoise=denoise_config, **inference_data),
            **data,
        )

    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to JSON file."""
        if path is None:
            path = self.default_path()

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @staticmethod
    def default_path() -> Path:
        """Return the default configuration file path."""
        return Path.home() / ".config" / "rcwx" / "config.json"

    def get_models_dir(self) -> Path:
        """Return the models directory as Path."""
        return Path(self.models_dir)
