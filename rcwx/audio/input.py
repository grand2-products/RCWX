"""Audio input stream using sounddevice."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray


class AudioInput:
    """
    Audio input stream manager.

    Captures audio from microphone using sounddevice.
    """

    def __init__(
        self,
        device: Optional[int] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        blocksize: int = 1024,
        callback: Optional[Callable[[NDArray[np.float32]], None]] = None,
    ):
        """
        Initialize audio input.

        Args:
            device: Input device index (None for default)
            sample_rate: Sample rate in Hz
            channels: Number of channels (1 for mono)
            blocksize: Samples per callback
            callback: Function to call with audio data
        """
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self._callback = callback
        self._stream: Optional[sd.InputStream] = None

    def _audio_callback(
        self,
        indata: NDArray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Internal sounddevice callback."""
        if status:
            import logging
            logging.getLogger(__name__).warning(f"Input stream status: {status}")

        if self._callback is not None:
            # Convert to mono float32 if needed
            audio = indata[:, 0].astype(np.float32)
            self._callback(audio)

    def start(self) -> None:
        """Start the input stream."""
        if self._stream is not None:
            return

        import logging
        logger = logging.getLogger(__name__)

        self._stream = sd.InputStream(
            device=self.device,
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.blocksize,
            dtype=np.float32,
            callback=self._audio_callback,
        )
        self._stream.start()

        # Log actual stream parameters with device name
        device_name = "システムデフォルト"
        if self.device is not None:
            try:
                device_info = sd.query_devices(self.device)
                device_name = device_info["name"]
            except Exception:
                device_name = f"device={self.device}"
        else:
            try:
                default_device = sd.query_devices(kind="input")
                device_name = f"システムデフォルト ({default_device['name']})"
            except Exception:
                pass
        logger.info(
            f"Input stream started: {device_name}, "
            f"sr={self._stream.samplerate:.0f}Hz, blocksize={self.blocksize}"
        )

    def stop(self) -> None:
        """Stop the input stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def set_callback(self, callback: Callable[[NDArray[np.float32]], None]) -> None:
        """Set the audio callback function."""
        self._callback = callback

    @property
    def is_active(self) -> bool:
        """Check if stream is active."""
        return self._stream is not None and self._stream.active


def list_input_devices(wasapi_only: bool = True) -> list[dict]:
    """
    List available audio input devices.

    Args:
        wasapi_only: If True, only show WASAPI devices (recommended for low latency)

    Returns:
        List of device info dictionaries
    """
    devices = []

    # Find WASAPI host API index
    wasapi_hostapi = None
    if wasapi_only:
        for i, hostapi in enumerate(sd.query_hostapis()):
            if "WASAPI" in hostapi["name"]:
                wasapi_hostapi = i
                break

    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            # Filter by host API if wasapi_only
            if wasapi_only and wasapi_hostapi is not None:
                if dev["hostapi"] != wasapi_hostapi:
                    continue
            devices.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "sample_rate": dev["default_samplerate"],
            })
    return devices


def get_default_input_device() -> Optional[int]:
    """Get the default input device index."""
    try:
        return sd.query_devices(kind="input")["index"]
    except Exception:
        return None
