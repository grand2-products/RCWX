"""Main RCWX GUI application."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Optional

import customtkinter as ctk
import numpy as np
import sounddevice as sd

from rcwx.audio.resample import resample
from rcwx.config import RCWXConfig
from rcwx.device import get_device, get_device_name
from rcwx.gui.widgets.audio_settings import AudioSettingsFrame
from rcwx.gui.widgets.latency_monitor import LatencyMonitor
from rcwx.gui.widgets.model_selector import ModelSelector
from rcwx.gui.widgets.pitch_control import PitchControl
from rcwx.audio.denoise import is_ml_denoiser_available
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeConfig, RealtimeStats, RealtimeVoiceChanger

logger = logging.getLogger(__name__)


class RCWXApp(ctk.CTk):
    """
    Main RCWX application window.
    """

    def __init__(self):
        super().__init__()

        # Configure window
        self.title("RCWX - RVC Voice Changer")
        self.geometry("650x550")
        self.minsize(600, 500)

        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Load configuration
        self.config = RCWXConfig.load()

        # Pipeline components
        self.pipeline: Optional[RVCPipeline] = None
        self.voice_changer: Optional[RealtimeVoiceChanger] = None

        # State
        self._is_running = False
        self._loading = False

        # Compute device info once
        self._device = get_device(self.config.device)
        self._device_name = get_device_name(self._device)

        # Setup UI
        self._setup_ui()

        # Initialize status bar device display
        self.status_bar.set_device(self._device_name)

        # Load last model if available
        if self.config.last_model_path and Path(self.config.last_model_path).exists():
            self.model_selector.set_model(self.config.last_model_path)

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_ui(self) -> None:
        """Setup the main UI layout."""
        # Create tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=(10, 5))

        # Add tabs
        self.tab_main = self.tabview.add("メイン")
        self.tab_audio = self.tabview.add("オーディオ")
        self.tab_settings = self.tabview.add("詳細設定")

        # Setup main tab
        self._setup_main_tab()

        # Setup audio tab
        self._setup_audio_tab()

        # Update audio device display in main panel
        self._update_audio_device_display()

        # Setup settings tab
        self._setup_settings_tab()

        # Status bar
        self.status_bar = LatencyMonitor(self, height=40)
        self.status_bar.pack(fill="x", padx=10, pady=(5, 10))

    def _setup_main_tab(self) -> None:
        """Setup the main tab content with 2-column layout."""
        # Scrollable container
        self.main_scroll = ctk.CTkScrollableFrame(self.tab_main, fg_color="transparent")
        self.main_scroll.pack(fill="both", expand=True)

        # 2-column container
        self.main_columns = ctk.CTkFrame(self.main_scroll, fg_color="transparent")
        self.main_columns.pack(fill="both", expand=True, padx=5, pady=5)
        self.main_columns.grid_columnconfigure(0, weight=1)
        self.main_columns.grid_columnconfigure(1, weight=1)

        # === Left column ===
        self.left_column = ctk.CTkFrame(self.main_columns, fg_color="transparent")
        self.left_column.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Model selector
        self.model_selector = ModelSelector(
            self.left_column,
            on_model_selected=self._on_model_selected,
        )
        self.model_selector.pack(fill="x", pady=(0, 10))

        # Pitch control
        self.pitch_control = PitchControl(
            self.left_column,
            on_pitch_changed=self._on_pitch_changed,
            on_f0_mode_changed=self._on_f0_mode_changed,
        )
        self.pitch_control.pack(fill="x", pady=(0, 10))

        # Index control
        self.index_frame = ctk.CTkFrame(self.left_column)
        self.index_frame.pack(fill="x", pady=(0, 10))

        self.index_label = ctk.CTkLabel(
            self.index_frame,
            text="■ Index検索",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.index_label.pack(anchor="w", padx=10, pady=(10, 5))

        self.use_index_var = ctk.BooleanVar(value=self.config.inference.use_index)
        self.use_index_cb = ctk.CTkCheckBox(
            self.index_frame,
            text="Indexを使用",
            variable=self.use_index_var,
            command=self._on_index_changed,
        )
        self.use_index_cb.pack(anchor="w", padx=10, pady=5)

        self.index_ratio_frame = ctk.CTkFrame(self.index_frame, fg_color="transparent")
        self.index_ratio_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.index_ratio_label = ctk.CTkLabel(
            self.index_ratio_frame,
            text="Index率:",
            font=ctk.CTkFont(size=11),
        )
        self.index_ratio_label.grid(row=0, column=0, padx=(0, 5))

        self.index_ratio_slider = ctk.CTkSlider(
            self.index_ratio_frame,
            from_=0,
            to=1,
            number_of_steps=20,
            width=120,
            command=self._on_index_ratio_changed,
        )
        self.index_ratio_slider.set(self.config.inference.index_ratio)
        self.index_ratio_slider.grid(row=0, column=1, padx=5)

        self.index_ratio_value = ctk.CTkLabel(
            self.index_ratio_frame,
            text=f"{self.config.inference.index_ratio:.2f}",
            width=40,
        )
        self.index_ratio_value.grid(row=0, column=2)

        self.index_status = ctk.CTkLabel(
            self.index_frame,
            text="",
            font=ctk.CTkFont(size=10),
            text_color="gray",
        )
        self.index_status.pack(anchor="w", padx=10, pady=(0, 5))

        # Noise cancellation control
        self.denoise_frame = ctk.CTkFrame(self.left_column)
        self.denoise_frame.pack(fill="x", pady=(0, 10))

        self.denoise_label = ctk.CTkLabel(
            self.denoise_frame,
            text="■ ノイズキャンセリング",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.denoise_label.pack(anchor="w", padx=10, pady=(10, 5))

        self.use_denoise_var = ctk.BooleanVar(value=self.config.inference.denoise.enabled)
        self.use_denoise_cb = ctk.CTkCheckBox(
            self.denoise_frame,
            text="ノイズキャンセリングを有効化",
            variable=self.use_denoise_var,
            command=self._on_denoise_changed,
        )
        self.use_denoise_cb.pack(anchor="w", padx=10, pady=5)

        # Method selection
        self.denoise_method_frame = ctk.CTkFrame(self.denoise_frame, fg_color="transparent")
        self.denoise_method_frame.pack(fill="x", padx=10, pady=(0, 5))

        self.denoise_method_label = ctk.CTkLabel(
            self.denoise_method_frame,
            text="方式:",
            font=ctk.CTkFont(size=11),
        )
        self.denoise_method_label.grid(row=0, column=0, padx=(0, 5))

        self.denoise_method_var = ctk.StringVar(value=self.config.inference.denoise.method)
        self.denoise_method_menu = ctk.CTkOptionMenu(
            self.denoise_method_frame,
            variable=self.denoise_method_var,
            values=["auto", "ml", "spectral"],
            width=120,
            command=lambda _: self._on_denoise_changed(),
        )
        self.denoise_method_menu.grid(row=0, column=1, padx=5)

        # Status label
        ml_status = "✓ 利用可能" if is_ml_denoiser_available() else "✗ 未インストール"
        self.denoise_status = ctk.CTkLabel(
            self.denoise_frame,
            text=f"ML Denoiser: {ml_status}",
            font=ctk.CTkFont(size=10),
            text_color="green" if is_ml_denoiser_available() else "gray",
        )
        self.denoise_status.pack(anchor="w", padx=10, pady=(0, 10))

        # === Right column ===
        self.right_column = ctk.CTkFrame(self.main_columns, fg_color="transparent")
        self.right_column.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Audio device info section
        self.device_frame = ctk.CTkFrame(self.right_column)
        self.device_frame.pack(fill="x", pady=(0, 10))

        self.device_section_label = ctk.CTkLabel(
            self.device_frame,
            text="■ オーディオデバイス",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.device_section_label.pack(anchor="w", padx=10, pady=(10, 5))

        # Input device (microphone)
        self.mic_label = ctk.CTkLabel(
            self.device_frame,
            text="🎤 デフォルト",
            font=ctk.CTkFont(size=11),
        )
        self.mic_label.pack(anchor="w", padx=15, pady=(0, 2))

        # Output device (speaker)
        self.speaker_label = ctk.CTkLabel(
            self.device_frame,
            text="🔊 デフォルト",
            font=ctk.CTkFont(size=11),
        )
        self.speaker_label.pack(anchor="w", padx=15, pady=(0, 5))

        # Inference device (GPU/CPU)
        self.inference_device_label = ctk.CTkLabel(
            self.device_frame,
            text=f"⚡ {self._device_name}",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self.inference_device_label.pack(anchor="w", padx=15, pady=(0, 10))

        # Test section
        self.test_frame = ctk.CTkFrame(self.right_column)
        self.test_frame.pack(fill="x", pady=(0, 10))

        self.test_label = ctk.CTkLabel(
            self.test_frame,
            text="■ オーディオテスト",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.test_label.pack(anchor="w", padx=10, pady=(10, 5))

        self.test_btn = ctk.CTkButton(
            self.test_frame,
            text="🎤 テスト (3秒録音→再生)",
            command=self._run_audio_test,
        )
        self.test_btn.pack(fill="x", padx=10, pady=(0, 5))

        self.test_status = ctk.CTkLabel(
            self.test_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self.test_status.pack(anchor="w", padx=10, pady=(0, 10))

        # Start/Stop button
        self.control_frame = ctk.CTkFrame(self.right_column)
        self.control_frame.pack(fill="x", pady=(0, 10))

        self.start_btn = ctk.CTkButton(
            self.control_frame,
            text="▶ 開始",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=60,
            command=self._toggle_running,
        )
        self.start_btn.pack(fill="x", padx=10, pady=10)

    def _setup_audio_tab(self) -> None:
        """Setup the audio settings tab."""
        # Scrollable container
        self.audio_scroll = ctk.CTkScrollableFrame(self.tab_audio, fg_color="transparent")
        self.audio_scroll.pack(fill="both", expand=True)

        self.audio_settings = AudioSettingsFrame(
            self.audio_scroll,
            on_settings_changed=self._on_audio_settings_changed,
        )
        self.audio_settings.pack(fill="both", expand=True, padx=10, pady=10)

        # Restore saved audio settings
        saved_gain = self.config.audio.input_gain_db
        if saved_gain != 0.0:
            self.audio_settings.gain_slider.set(saved_gain)
            self.audio_settings.input_gain_db = saved_gain
            self.audio_settings.gain_value_label.configure(text=f"{saved_gain:+.0f} dB")

        # Restore chunk size
        saved_chunk = self.config.audio.chunk_sec
        for label, value in self.audio_settings.chunk_options:
            if value == saved_chunk:
                self.audio_settings.chunk_var.set(label)
                self.audio_settings.chunk_sec = saved_chunk
                break

        # Restore saved device selections
        if self.config.audio.input_device_name:
            self.audio_settings.set_input_device(self.config.audio.input_device_name)
        if self.config.audio.output_device_name:
            self.audio_settings.set_output_device(self.config.audio.output_device_name)

    def _setup_settings_tab(self) -> None:
        """Setup the advanced settings tab."""
        # Scrollable container
        self.settings_scroll = ctk.CTkScrollableFrame(self.tab_settings, fg_color="transparent")
        self.settings_scroll.pack(fill="both", expand=True)

        # Compile option (not available on Windows - Triton not supported)
        compile_default = False if sys.platform == "win32" else self.config.inference.use_compile
        self.compile_var = ctk.BooleanVar(value=compile_default)
        if sys.platform != "win32":
            self.compile_cb = ctk.CTkCheckBox(
                self.settings_scroll,
                text="torch.compileを使用 (初回起動時にコンパイル)",
                variable=self.compile_var,
                command=self._save_config,
            )
            self.compile_cb.pack(anchor="w", padx=20, pady=10)

        # Device selection
        self.device_label = ctk.CTkLabel(
            self.settings_scroll,
            text="デバイス選択",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.device_label.pack(anchor="w", padx=20, pady=(20, 5))

        self.device_var = ctk.StringVar(value=self.config.device)
        self.device_menu = ctk.CTkOptionMenu(
            self.settings_scroll,
            variable=self.device_var,
            values=["auto", "xpu", "cuda", "cpu"],
            command=lambda _: self._save_config(),
        )
        self.device_menu.pack(anchor="w", padx=20, pady=5)

        # Data type selection
        self.dtype_label = ctk.CTkLabel(
            self.settings_scroll,
            text="データ型",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.dtype_label.pack(anchor="w", padx=20, pady=(20, 5))

        self.dtype_var = ctk.StringVar(value=self.config.dtype)
        self.dtype_menu = ctk.CTkOptionMenu(
            self.settings_scroll,
            variable=self.dtype_var,
            values=["float16", "float32", "bfloat16"],
            command=lambda _: self._save_config(),
        )
        self.dtype_menu.pack(anchor="w", padx=20, pady=5)

        # Models directory
        self.models_dir_label = ctk.CTkLabel(
            self.settings_scroll,
            text="モデルディレクトリ",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.models_dir_label.pack(anchor="w", padx=20, pady=(20, 5))

        self.models_dir_entry = ctk.CTkEntry(
            self.settings_scroll,
            width=400,
        )
        self.models_dir_entry.pack(anchor="w", padx=20, pady=5)
        self.models_dir_entry.insert(0, self.config.models_dir)
        self.models_dir_entry.bind("<FocusOut>", lambda _: self._save_config())

        # Apply button
        self.apply_btn = ctk.CTkButton(
            self.settings_scroll,
            text="設定を適用 (モデル再読込)",
            command=self._apply_settings,
        )
        self.apply_btn.pack(anchor="w", padx=20, pady=(20, 10))

        # Settings info label
        self.settings_info = ctk.CTkLabel(
            self.settings_scroll,
            text="※ デバイス/データ型の変更はモデル再読込後に反映されます",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self.settings_info.pack(anchor="w", padx=20, pady=5)

    def _apply_settings(self) -> None:
        """Apply settings and reload model."""
        if self.model_selector.model_path:
            # Stop if running
            if self._is_running:
                self._stop_voice_changer()

            # Unload current pipeline
            if self.pipeline:
                self.pipeline.unload()
                self.pipeline = None

            # Reload with new settings
            self._load_model_async(self.model_selector.model_path)

    def _on_model_selected(self, path: str) -> None:
        """Handle model selection."""
        logger.info(f"Model selected: {path}")

        # Save to config
        self.config.last_model_path = path
        self.config.save()

        # Unload current pipeline if running
        if self._is_running:
            self._stop_voice_changer()

        # Load new model in background
        self._load_model_async(path)

    def _load_model_async(self, path: str) -> None:
        """Load model in background thread."""
        if self._loading:
            return

        self._loading = True
        self.status_bar.set_loading()
        self.start_btn.configure(state="disabled")

        def load_thread():
            try:
                self.pipeline = RVCPipeline(
                    path,
                    device=self.device_var.get(),
                    dtype=self.dtype_var.get(),
                    use_f0=self.pitch_control.use_f0,
                    use_compile=self.compile_var.get(),
                    models_dir=self.models_dir_entry.get(),
                )
                self.pipeline.load()

                # Update UI from main thread
                self.after(0, self._on_model_loaded)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.after(0, lambda: self._on_model_load_error(str(e)))

        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()

    def _on_model_loaded(self) -> None:
        """Called when model is loaded successfully."""
        self._loading = False
        self.start_btn.configure(state="normal")
        self.status_bar.set_running(False)

        # Update model info and device display
        if self.pipeline:
            self.model_selector.set_model_info(
                has_f0=self.pipeline.has_f0,
                version=self.pipeline.synthesizer.version if self.pipeline.synthesizer else 2,
            )
            self.pitch_control.set_f0_enabled(self.pipeline.has_f0)

            # Update device name in status bar
            device_name = get_device_name(self.pipeline.device)
            self.status_bar.set_device(device_name)

            # Update index status
            if self.pipeline.faiss_index is not None:
                n_vectors = self.pipeline.faiss_index.ntotal
                self.index_status.configure(
                    text=f"Index読込済 ({n_vectors}ベクトル)",
                    text_color="green",
                )
            else:
                self.index_status.configure(
                    text="Indexなし",
                    text_color="gray",
                )

    def _on_model_load_error(self, error: str) -> None:
        """Called when model loading fails."""
        self._loading = False
        self.start_btn.configure(state="normal")
        self.status_bar.set_running(False)

        # Show error dialog
        dialog = ctk.CTkToplevel(self)
        dialog.title("エラー")
        dialog.geometry("400x150")
        dialog.transient(self)
        dialog.grab_set()

        label = ctk.CTkLabel(
            dialog,
            text=f"モデルの読み込みに失敗しました:\n{error}",
            wraplength=350,
        )
        label.pack(pady=20)

        btn = ctk.CTkButton(dialog, text="OK", command=dialog.destroy)
        btn.pack(pady=10)

    def _on_pitch_changed(self, value: int) -> None:
        """Handle pitch change."""
        if self.voice_changer:
            self.voice_changer.set_pitch_shift(value)

    def _on_f0_mode_changed(self, use_f0: bool) -> None:
        """Handle F0 mode change."""
        if self.voice_changer:
            self.voice_changer.set_f0_mode(use_f0)

    def _on_index_changed(self) -> None:
        """Handle index checkbox change."""
        self._save_config()
        # Update voice changer if running
        if self.voice_changer:
            self.voice_changer.set_index_rate(self._get_index_rate())

    def _on_denoise_changed(self) -> None:
        """Handle denoise settings change."""
        self._save_config()
        # Update voice changer if running
        if self.voice_changer:
            self.voice_changer.set_denoise(
                self.use_denoise_var.get(),
                self.denoise_method_var.get(),
            )

    def _on_index_ratio_changed(self, value: float) -> None:
        """Handle index ratio slider change."""
        self.index_ratio_value.configure(text=f"{value:.2f}")
        self._save_config()
        # Update voice changer if running
        if self.voice_changer:
            self.voice_changer.set_index_rate(self._get_index_rate())

    def _get_index_rate(self) -> float:
        """Get current index rate (0 if disabled)."""
        if self.use_index_var.get():
            return self.index_ratio_slider.get()
        return 0.0

    def _on_audio_settings_changed(self) -> None:
        """Handle audio settings change."""
        # Update device display in main panel
        self._update_audio_device_display()
        # Save immediately
        self._save_config()

    def _save_config(self) -> None:
        """Save all config settings immediately."""
        try:
            self.config.device = self.device_var.get()
            self.config.dtype = self.dtype_var.get()
            self.config.models_dir = self.models_dir_entry.get()
            self.config.inference.use_compile = self.compile_var.get()
            self.config.inference.use_index = self.use_index_var.get()
            self.config.inference.index_ratio = self.index_ratio_slider.get()
            self.config.inference.denoise.enabled = self.use_denoise_var.get()
            self.config.inference.denoise.method = self.denoise_method_var.get()
            self.config.audio.chunk_sec = self.audio_settings.chunk_sec
            self.config.audio.input_gain_db = self.audio_settings.input_gain_db
            self.config.audio.input_device_name = self.audio_settings.get_input_device_name()
            self.config.audio.output_device_name = self.audio_settings.get_output_device_name()
            self.config.save()
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def _update_audio_device_display(self) -> None:
        """Update audio device labels in main panel."""
        if hasattr(self, "audio_settings"):
            input_name = self.audio_settings.get_input_device_name()
            output_name = self.audio_settings.get_output_device_name()
            # Truncate long names
            if len(input_name) > 35:
                input_name = input_name[:32] + "..."
            if len(output_name) > 35:
                output_name = output_name[:32] + "..."
            self.mic_label.configure(text=f"🎤 {input_name}")
            self.speaker_label.configure(text=f"🔊 {output_name}")

    def _toggle_running(self) -> None:
        """Toggle voice changer on/off."""
        if self._is_running:
            self._stop_voice_changer()
        else:
            self._start_voice_changer()

    def _check_same_audio_interface(self) -> bool:
        """Check if input and output use the same audio interface (potential feedback)."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()

            input_idx = self.audio_settings.input_device
            output_idx = self.audio_settings.output_device

            # Use defaults if None
            if input_idx is None:
                input_idx = sd.default.device[0]
            if output_idx is None:
                output_idx = sd.default.device[1]

            if input_idx is None or output_idx is None:
                return False

            input_name = devices[input_idx]["name"].lower()
            output_name = devices[output_idx]["name"].lower()

            # Check for common interface indicators
            # "High Definition Audio" is the typical onboard audio
            hda_keywords = ["high definition audio", "realtek", "hd audio"]
            input_is_hda = any(kw in input_name for kw in hda_keywords)
            output_is_hda = any(kw in output_name for kw in hda_keywords)

            return input_is_hda and output_is_hda
        except Exception:
            return False

    def _start_voice_changer(self) -> None:
        """Start the voice changer."""
        if self.pipeline is None:
            logger.warning("No model loaded")
            return

        if self._loading:
            return

        # Check for same audio interface (potential feedback)
        if self._check_same_audio_interface():
            logger.warning("Input and output use same audio interface - feedback may occur")
            self._show_warning(
                "フィードバック警告",
                "入力と出力が同じオーディオインターフェースを使用しています。\n"
                "フィードバックループが発生する可能性があります。\n\n"
                "推奨: USBマイクなど、別のインターフェースを入力に使用してください。"
            )

        # Stop audio monitor to avoid device conflict
        self.audio_settings.stop_monitor()

        # Create realtime config with auto-detected sample rates
        rt_config = RealtimeConfig(
            input_device=self.audio_settings.input_device,
            output_device=self.audio_settings.output_device,
            mic_sample_rate=self.audio_settings.input_sample_rate,
            output_sample_rate=self.audio_settings.output_sample_rate,
            chunk_sec=self.audio_settings.chunk_sec,
            pitch_shift=self.pitch_control.pitch,
            use_f0=self.pitch_control.use_f0,
            input_gain_db=self.audio_settings.input_gain_db,
            index_rate=self._get_index_rate(),
            denoise_enabled=self.use_denoise_var.get(),
            denoise_method=self.denoise_method_var.get(),
        )

        # Create voice changer
        self.voice_changer = RealtimeVoiceChanger(
            self.pipeline,
            config=rt_config,
        )
        self.voice_changer.on_stats_update = self._on_stats_update
        self.voice_changer.on_error = self._on_inference_error

        # Start
        try:
            self.voice_changer.start()
            self._is_running = True
            self.start_btn.configure(text="■ 停止", fg_color="#cc3333")
            self.status_bar.set_running(True)
        except Exception as e:
            logger.error(f"Failed to start voice changer: {e}")

    def _stop_voice_changer(self) -> None:
        """Stop the voice changer."""
        if self.voice_changer:
            self.voice_changer.stop()
            self.voice_changer = None

        self._is_running = False
        self.start_btn.configure(text="▶ 開始", fg_color=["#3B8ED0", "#1F6AA5"])
        self.status_bar.set_running(False)

    def _on_stats_update(self, stats: RealtimeStats) -> None:
        """Handle stats update from voice changer."""
        # Update UI from main thread
        self.after(0, lambda: self.status_bar.update_stats(stats))

    def _on_inference_error(self, error_msg: str) -> None:
        """Handle inference error from voice changer."""
        # Update UI from main thread
        self.after(0, lambda: self._show_error(error_msg))

    def _show_warning(self, title: str, message: str) -> None:
        """Show warning dialog."""
        import customtkinter as ctk

        dialog = ctk.CTkToplevel(self)
        dialog.title(title)
        dialog.geometry("400x200")
        dialog.transient(self)
        dialog.grab_set()

        # Center on parent
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 400) // 2
        y = self.winfo_y() + (self.winfo_height() - 200) // 2
        dialog.geometry(f"+{x}+{y}")

        label = ctk.CTkLabel(
            dialog,
            text=message,
            justify="left",
            wraplength=360,
        )
        label.pack(pady=20, padx=20)

        btn = ctk.CTkButton(dialog, text="OK", command=dialog.destroy)
        btn.pack(pady=10)

    def _show_error(self, error_msg: str) -> None:
        """Show error message in UI."""
        # Show in model selector's status label (truncate long messages)
        short_msg = error_msg[:60] + "..." if len(error_msg) > 60 else error_msg
        self.model_selector.status_label.configure(text=short_msg, text_color="#ff6666")

    def _run_audio_test(self) -> None:
        """Run audio test: record -> convert -> playback."""
        if self._is_running:
            self.test_status.configure(text="変換中は使用できません", text_color="orange")
            return

        # Disable button during test
        self.test_btn.configure(state="disabled")
        self.test_status.configure(text="録音中...", text_color="white")

        def test_thread():
            try:
                from scipy.io import wavfile
                duration = 3.0  # seconds

                # Debug output directory
                debug_dir = Path("debug_audio")
                debug_dir.mkdir(exist_ok=True)

                # Get device settings (auto-detected sample rates)
                input_device = self.audio_settings.input_device
                output_device = self.audio_settings.output_device
                mic_sr = self.audio_settings.input_sample_rate
                out_sr = self.audio_settings.output_sample_rate
                process_sr = 16000  # HuBERT/RMVPE expect 16kHz

                # Record at device's native rate
                logger.info(f"Recording: device={input_device}, sr={mic_sr}, duration={duration}s")
                self.after(0, lambda: self.test_status.configure(text="🔴 録音中...", text_color="#ff6666"))
                audio_raw = sd.rec(
                    int(duration * mic_sr),
                    samplerate=mic_sr,
                    channels=1,
                    dtype=np.float32,
                    device=input_device,
                )
                sd.wait()
                audio_raw = audio_raw.flatten()
                logger.info(f"Recorded: shape={audio_raw.shape}, min={audio_raw.min():.4f}, max={audio_raw.max():.4f}")

                # Apply input gain
                input_gain_db = self.audio_settings.input_gain_db
                if input_gain_db != 0.0:
                    gain_linear = 10 ** (input_gain_db / 20)
                    audio_raw = audio_raw * gain_linear
                    logger.info(f"Applied gain: {input_gain_db:+.0f} dB, max={audio_raw.max():.4f}")

                # Save raw input (with gain applied)
                wavfile.write(debug_dir / "01_input_raw.wav", mic_sr, audio_raw)
                logger.info(f"Saved: debug_audio/01_input_raw.wav ({mic_sr}Hz)")

                # Resample to 16kHz for processing
                audio = audio_raw
                if mic_sr != process_sr:
                    audio = resample(audio, mic_sr, process_sr)
                    logger.info(f"Resampled to 16kHz: shape={audio.shape}, min={audio.min():.4f}, max={audio.max():.4f}")

                # Save resampled input
                wavfile.write(debug_dir / "02_input_16k.wav", process_sr, audio)
                logger.info(f"Saved: debug_audio/02_input_16k.wav ({process_sr}Hz)")

                # Convert if pipeline is loaded
                output_sr = out_sr  # Default to output device rate
                if self.pipeline is not None:
                    self.after(0, lambda: self.test_status.configure(text="🔄 変換中...", text_color="#66b3ff"))
                    import torch
                    audio_tensor = torch.from_numpy(audio).float()
                    f0_method = "rmvpe" if self.pitch_control.use_f0 else "none"
                    audio_converted = self.pipeline.infer(
                        audio_tensor,
                        pitch_shift=self.pitch_control.pitch,
                        f0_method=f0_method,
                        index_rate=self._get_index_rate(),
                    )

                    # Save converted output at model rate
                    model_sr = self.pipeline.sample_rate
                    wavfile.write(debug_dir / "03_output_model.wav", model_sr, audio_converted)
                    logger.info(f"Saved: debug_audio/03_output_model.wav ({model_sr}Hz)")

                    # Resample from model rate to output device rate
                    if model_sr != out_sr:
                        audio = resample(audio_converted, model_sr, out_sr)
                    else:
                        audio = audio_converted
                else:
                    # No conversion - resample back to output rate for playback
                    if process_sr != out_sr:
                        audio = resample(audio, process_sr, out_sr)

                # Save final output
                wavfile.write(debug_dir / "04_output_final.wav", out_sr, audio)
                logger.info(f"Saved: debug_audio/04_output_final.wav ({out_sr}Hz)")

                # Playback at output device's native rate
                self.after(0, lambda: self.test_status.configure(text="🔊 再生中...", text_color="#66ff66"))
                sd.play(audio, samplerate=output_sr, device=output_device)
                sd.wait()

                # Done
                if self.pipeline is not None:
                    self.after(0, lambda: self.test_status.configure(text="✓ 完了 (debug_audio/に保存)", text_color="green"))
                else:
                    self.after(0, lambda: self.test_status.configure(text="✓ 完了 (変換なし)", text_color="gray"))

            except Exception as e:
                logger.error(f"Audio test failed: {e}")
                error_msg = str(e)[:40]
                self.after(0, lambda msg=error_msg: self.test_status.configure(text=f"エラー: {msg}", text_color="red"))

            finally:
                self.after(0, lambda: self.test_btn.configure(state="normal"))

        thread = threading.Thread(target=test_thread, daemon=True)
        thread.start()

    def _on_close(self) -> None:
        """Handle window close."""
        # Stop voice changer
        if self._is_running:
            self._stop_voice_changer()

        # Stop audio monitor
        self.audio_settings.stop_monitor()

        # Save config
        self._save_config()

        # Destroy window
        self.destroy()

    def run(self) -> None:
        """Run the application."""
        self.mainloop()


def run_gui() -> None:
    """Entry point for GUI application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = RCWXApp()
    app.run()


if __name__ == "__main__":
    run_gui()
