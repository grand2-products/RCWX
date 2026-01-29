"""Audio settings widget."""

from __future__ import annotations

import threading
from typing import Callable, Optional

import customtkinter as ctk
import numpy as np

from rcwx.audio.input import list_input_devices
from rcwx.audio.output import list_output_devices


class AudioSettingsFrame(ctk.CTkFrame):
    """
    Audio device settings widget.

    Allows users to select input/output devices and configure sample rates.
    """

    def __init__(
        self,
        master: ctk.CTk,
        on_settings_changed: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)

        self.on_settings_changed = on_settings_changed

        # Device lists
        self._input_devices: list[dict] = []
        self._output_devices: list[dict] = []

        # Selected values
        self.input_device: Optional[int] = None
        self.output_device: Optional[int] = None
        self.input_sample_rate: int = 48000  # Default, auto-detected on device change
        self.output_sample_rate: int = 48000  # Default, auto-detected on device change
        self.chunk_sec: float = 0.35
        self.input_gain_db: float = 0.0  # Input gain in dB

        self._load_device_lists()
        self._setup_ui()
        self._detect_default_sample_rates()

    def _load_device_lists(self) -> None:
        """Load device lists before UI setup."""
        self._input_devices = list_input_devices()
        self._output_devices = list_output_devices()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Input device section
        self.input_label = ctk.CTkLabel(
            self,
            text="入力デバイス",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.input_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        input_names = ["デフォルト"] + [d["name"] for d in self._input_devices]
        self.input_var = ctk.StringVar(value="デフォルト")
        self.input_var.trace_add("write", lambda *_: self._on_input_change(self.input_var.get()))
        self.input_dropdown = ctk.CTkOptionMenu(
            self,
            variable=self.input_var,
            values=input_names,
            width=300,
        )
        self.input_dropdown.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        # Output device section
        self.output_label = ctk.CTkLabel(
            self,
            text="出力デバイス",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.output_label.grid(row=2, column=0, sticky="w", padx=10, pady=(15, 5))

        output_names = ["デフォルト"] + [d["name"] for d in self._output_devices]
        self.output_var = ctk.StringVar(value="デフォルト")
        self.output_var.trace_add("write", lambda *_: self._on_output_change(self.output_var.get()))
        self.output_dropdown = ctk.CTkOptionMenu(
            self,
            variable=self.output_var,
            values=output_names,
            width=300,
        )
        self.output_dropdown.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        # Chunk size section
        self.chunk_label = ctk.CTkLabel(
            self,
            text="チャンクサイズ",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.chunk_label.grid(row=4, column=0, sticky="w", padx=10, pady=(15, 5))

        self.chunk_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.chunk_frame.grid(row=5, column=0, padx=10, pady=5, sticky="ew")

        self.chunk_var = ctk.StringVar(value="350ms (バランス)")
        self.chunk_options = [
            ("200ms (低遅延/F0なし)", 0.2),
            ("350ms (バランス)", 0.35),
            ("500ms (高品質)", 0.5),
        ]

        for i, (label, value) in enumerate(self.chunk_options):
            rb = ctk.CTkRadioButton(
                self.chunk_frame,
                text=label,
                variable=self.chunk_var,
                value=label,
                command=self._on_chunk_change,
            )
            rb.grid(row=0, column=i, padx=5, pady=5)

        # Input level meter section
        self.level_label = ctk.CTkLabel(
            self,
            text="入力レベル",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.level_label.grid(row=6, column=0, sticky="w", padx=10, pady=(15, 5))

        self.level_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.level_frame.grid(row=7, column=0, padx=10, pady=5, sticky="ew")
        self.level_frame.grid_columnconfigure(0, weight=1)

        self.level_bar = ctk.CTkProgressBar(self.level_frame, width=280, height=20)
        self.level_bar.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.level_bar.set(0)

        self.level_value = ctk.CTkLabel(self.level_frame, text="-∞ dB", width=60)
        self.level_value.grid(row=0, column=1)

        self.monitor_btn = ctk.CTkButton(
            self,
            text="モニター開始",
            width=120,
            command=self._toggle_monitor,
        )
        self.monitor_btn.grid(row=8, column=0, padx=10, pady=(5, 10), sticky="w")

        # Input gain section
        self.gain_label = ctk.CTkLabel(
            self,
            text="入力ゲイン補正",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.gain_label.grid(row=9, column=0, sticky="w", padx=10, pady=(15, 5))

        self.gain_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.gain_frame.grid(row=10, column=0, padx=10, pady=5, sticky="ew")

        self.gain_slider = ctk.CTkSlider(
            self.gain_frame,
            from_=-12,
            to=24,
            number_of_steps=36,
            width=200,
            command=self._on_gain_change,
        )
        self.gain_slider.set(0)
        self.gain_slider.grid(row=0, column=0, padx=(0, 10))

        self.gain_value_label = ctk.CTkLabel(self.gain_frame, text="0 dB", width=50)
        self.gain_value_label.grid(row=0, column=1)

        # Recommended gain display
        self.recommended_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.recommended_frame.grid(row=11, column=0, padx=10, pady=5, sticky="ew")

        self.recommended_label = ctk.CTkLabel(
            self.recommended_frame,
            text="推奨: -- dB",
            text_color="gray",
        )
        self.recommended_label.grid(row=0, column=0, sticky="w")

        self.apply_recommended_btn = ctk.CTkButton(
            self.recommended_frame,
            text="推奨値をセット",
            width=100,
            command=self._apply_recommended_gain,
        )
        self.apply_recommended_btn.grid(row=0, column=1, padx=(10, 0))

        # Configure grid
        self.grid_columnconfigure(0, weight=1)

        # Monitor state
        self._monitoring = False
        self._monitor_stream = None
        self._peak_db = -60.0
        self._recommended_gain = 0.0

    def _refresh_devices(self) -> None:
        """Refresh all device lists."""
        self._refresh_input_devices()
        self._refresh_output_devices()

    def _refresh_input_devices(self) -> None:
        """Refresh input device list."""
        self._input_devices = list_input_devices()
        input_names = ["デフォルト"] + [d["name"] for d in self._input_devices]
        self.input_dropdown.configure(values=input_names)

    def _refresh_output_devices(self) -> None:
        """Refresh output device list."""
        self._output_devices = list_output_devices()
        output_names = ["デフォルト"] + [d["name"] for d in self._output_devices]
        self.output_dropdown.configure(values=output_names)

    def _detect_default_sample_rates(self) -> None:
        """Detect sample rates for default devices."""
        import sounddevice as sd
        try:
            # Get default input device info
            default_input = sd.query_devices(kind="input")
            self.input_sample_rate = int(default_input["default_samplerate"])
        except Exception:
            self.input_sample_rate = 48000

        try:
            # Get default output device info
            default_output = sd.query_devices(kind="output")
            self.output_sample_rate = int(default_output["default_samplerate"])
        except Exception:
            self.output_sample_rate = 48000

    def _on_input_change(self, value: str) -> None:
        """Handle input device change."""
        if value == "デフォルト":
            self.input_device = None
            self._detect_default_sample_rates()
        else:
            # デバイスを検索、見つからなければNone
            self.input_device = None
            for device in self._input_devices:
                if device["name"] == value:
                    self.input_device = device["index"]
                    self.input_sample_rate = int(device["sample_rate"])
                    break

        if self.on_settings_changed:
            self.on_settings_changed()

    def _on_output_change(self, value: str) -> None:
        """Handle output device change."""
        if value == "デフォルト":
            self.output_device = None
            self._detect_default_sample_rates()
        else:
            # デバイスを検索、見つからなければNone
            self.output_device = None
            for device in self._output_devices:
                if device["name"] == value:
                    self.output_device = device["index"]
                    self.output_sample_rate = int(device["sample_rate"])
                    break

        if self.on_settings_changed:
            self.on_settings_changed()

    def _on_chunk_change(self) -> None:
        """Handle chunk size change."""
        value = self.chunk_var.get()
        for label, sec in self.chunk_options:
            if label == value:
                self.chunk_sec = sec
                break

        if self.on_settings_changed:
            self.on_settings_changed()

    def _toggle_monitor(self) -> None:
        """Toggle input level monitoring."""
        if self._monitoring:
            self._stop_monitor()
        else:
            self._start_monitor()

    def _start_monitor(self) -> None:
        """Start input level monitoring."""
        import sounddevice as sd

        self._monitoring = True
        self.monitor_btn.configure(text="モニター停止", fg_color="#cc3333")

        def audio_callback(indata, frames, time, status):
            if not self._monitoring:
                return
            # Calculate RMS level
            rms = np.sqrt(np.mean(indata ** 2))
            # Calculate peak level
            peak = np.max(np.abs(indata))
            # Convert to dB (with floor at -60 dB)
            rms_db = 20 * np.log10(max(rms, 1e-6))
            peak_db = 20 * np.log10(max(peak, 1e-6))
            rms_db = max(rms_db, -60)
            peak_db = max(peak_db, -60)
            # Normalize to 0-1 range (-60 to 0 dB)
            level = (rms_db + 60) / 60
            # Update UI from main thread
            self.after(0, lambda l=level, r=rms_db, p=peak_db: self._update_level(l, r, p))

        try:
            # Use device's native sample rate
            monitor_sr = self.input_sample_rate
            blocksize = int(monitor_sr * 0.1)  # 100ms blocks
            self._monitor_stream = sd.InputStream(
                device=self.input_device,
                channels=1,
                samplerate=monitor_sr,
                blocksize=blocksize,
                callback=audio_callback,
            )
            self._monitor_stream.start()
        except Exception as e:
            self._monitoring = False
            self.monitor_btn.configure(text="モニター開始", fg_color=["#3B8ED0", "#1F6AA5"])
            self.level_value.configure(text=f"エラー")

    def _stop_monitor(self) -> None:
        """Stop input level monitoring."""
        self._monitoring = False
        if self._monitor_stream:
            self._monitor_stream.stop()
            self._monitor_stream.close()
            self._monitor_stream = None
        self.monitor_btn.configure(text="モニター開始", fg_color=["#3B8ED0", "#1F6AA5"])
        self.level_bar.set(0)
        self.level_value.configure(text="-∞ dB")

    def _update_level(self, level: float, rms_db: float, peak_db: float) -> None:
        """Update level meter display and recommended gain."""
        self.level_bar.set(min(level, 1.0))
        if rms_db <= -60:
            self.level_value.configure(text="-∞ dB")
        else:
            self.level_value.configure(text=f"{rms_db:.0f} dB")

        # Track peak and calculate recommended gain
        # Target peak: -6 dB (leaving headroom)
        self._peak_db = peak_db
        target_peak = -6.0
        self._recommended_gain = target_peak - peak_db

        # Clamp to reasonable range
        self._recommended_gain = max(-12, min(24, self._recommended_gain))

        if peak_db <= -60:
            self.recommended_label.configure(text="推奨: -- dB (信号なし)")
        else:
            self.recommended_label.configure(text=f"推奨: {self._recommended_gain:+.0f} dB (ピーク: {peak_db:.0f} dB)")

    def _on_gain_change(self, value: float) -> None:
        """Handle input gain slider change."""
        self.input_gain_db = round(value)
        self.gain_value_label.configure(text=f"{self.input_gain_db:+.0f} dB")
        if self.on_settings_changed:
            self.on_settings_changed()

    def _apply_recommended_gain(self) -> None:
        """Apply recommended gain value."""
        self.gain_slider.set(self._recommended_gain)
        self._on_gain_change(self._recommended_gain)

    def stop_monitor(self) -> None:
        """Public method to stop monitoring (called when closing app)."""
        if self._monitoring:
            self._stop_monitor()

    def get_input_device_name(self) -> str:
        """Get the currently selected input device name."""
        return self.input_var.get()

    def get_output_device_name(self) -> str:
        """Get the currently selected output device name."""
        return self.output_var.get()

    def set_input_device(self, name: str) -> None:
        """Set input device by name (for restoring saved settings)."""
        # Check if device exists in current list
        available_names = ["デフォルト"] + [d["name"] for d in self._input_devices]
        if name in available_names:
            self.input_var.set(name)
            # _on_input_change will be called via trace

    def set_output_device(self, name: str) -> None:
        """Set output device by name (for restoring saved settings)."""
        # Check if device exists in current list
        available_names = ["デフォルト"] + [d["name"] for d in self._output_devices]
        if name in available_names:
            self.output_var.set(name)
            # _on_output_change will be called via trace
