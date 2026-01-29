"""Real-time voice conversion pipeline."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Callable, Optional

import numpy as np

from rcwx.audio.buffer import ChunkBuffer, OutputBuffer
from rcwx.audio.denoise import denoise as denoise_audio
from rcwx.audio.input import AudioInput
from rcwx.audio.output import AudioOutput
from rcwx.audio.resample import resample
from rcwx.pipeline.inference import RVCPipeline

logger = logging.getLogger(__name__)


@dataclass
class RealtimeStats:
    """Statistics for real-time processing."""

    latency_ms: float = 0.0
    inference_ms: float = 0.0
    buffer_underruns: int = 0
    buffer_overruns: int = 0
    frames_processed: int = 0
    feedback_detected: bool = False
    feedback_correlation: float = 0.0

    def reset(self) -> None:
        self.latency_ms = 0.0
        self.inference_ms = 0.0
        self.buffer_underruns = 0
        self.buffer_overruns = 0
        self.frames_processed = 0
        self.feedback_detected = False
        self.feedback_correlation = 0.0


@dataclass
class RealtimeConfig:
    """Configuration for real-time processing."""

    input_device: Optional[int] = None
    output_device: Optional[int] = None
    # Microphone capture rate (native device rate for better compatibility)
    mic_sample_rate: int = 48000
    # Internal processing rate (HuBERT/RMVPE expect 16kHz)
    input_sample_rate: int = 16000
    output_sample_rate: int = 48000
    # Note: RMVPE requires at least 32 mel frames (0.32 sec at 160 hop)
    # Use 0.35 sec for margin
    chunk_sec: float = 0.35
    crossfade_sec: float = 0.05  # 50ms crossfade to smooth chunk boundaries
    pitch_shift: int = 0
    use_f0: bool = True
    max_queue_size: int = 2  # Reduced from 4 to minimize latency
    # Number of chunks to pre-buffer before starting output
    prebuffer_chunks: int = 1  # Reduced from 2 to minimize latency
    # Input gain in dB
    input_gain_db: float = 0.0
    # FAISS index rate (0=disabled, 0.5=balanced, 1=index only)
    index_rate: float = 0.0
    # Noise cancellation
    denoise_enabled: bool = False
    denoise_method: str = "auto"  # auto, deepfilter, spectral


class RealtimeVoiceChanger:
    """
    Real-time voice conversion using RVC pipeline.

    Manages audio input/output streams and runs inference in a separate thread.
    """

    def __init__(
        self,
        pipeline: RVCPipeline,
        config: Optional[RealtimeConfig] = None,
    ):
        """
        Initialize the real-time voice changer.

        Args:
            pipeline: RVC pipeline for inference
            config: Real-time configuration
        """
        self.pipeline = pipeline
        self.config = config or RealtimeConfig()

        # Calculate buffer sizes (at mic sample rate for input buffering)
        self.mic_chunk_samples = int(self.config.mic_sample_rate * self.config.chunk_sec)

        # Audio streams
        self.audio_input: Optional[AudioInput] = None
        self.audio_output: Optional[AudioOutput] = None

        # Buffers - simple non-overlapping chunks with output crossfade
        self.input_buffer = ChunkBuffer(
            self.mic_chunk_samples,
            crossfade_samples=0,
            context_samples=0,  # No overlap - simple sequential chunks
        )

        # Crossfade samples at output sample rate (to smooth boundaries)
        self.output_crossfade_samples = int(
            self.config.output_sample_rate * self.config.crossfade_sec
        )
        # Max buffer = 1.5 chunks (tight latency, drops old samples to catch up)
        self.output_buffer = OutputBuffer(
            max_latency_samples=int(
                self.config.output_sample_rate * self.config.chunk_sec * 1.5
            ),
            fade_samples=256,
        )
        # Previous output for crossfade
        self._prev_output: Optional[np.ndarray] = None

        # Processing state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._input_queue: Queue = Queue(maxsize=self.config.max_queue_size)
        self._output_queue: Queue = Queue(maxsize=self.config.max_queue_size)

        # Pre-buffering state (wait for N chunks before starting output)
        self._prebuffer_chunks = self.config.prebuffer_chunks
        self._chunks_ready = 0
        self._output_started = False

        # Statistics
        self.stats = RealtimeStats()

        # Callbacks
        self.on_stats_update: Optional[Callable[[RealtimeStats], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

        # Error tracking
        self._last_error: Optional[str] = None
        self._error_count: int = 0

        # Feedback detection: store recent output for correlation check
        # Store ~1 second of output for comparison with input
        self._output_history_size = self.config.output_sample_rate  # 1 second
        self._output_history: np.ndarray = np.zeros(self._output_history_size, dtype=np.float32)
        self._output_history_pos = 0
        self._feedback_check_interval = 10  # Check every N chunks
        self._feedback_warning_shown = False

        # Crossfade windows (at output sample rate)
        if self.output_crossfade_samples > 0:
            self._fade_in = np.linspace(0, 1, self.output_crossfade_samples, dtype=np.float32)
            self._fade_out = np.linspace(1, 0, self.output_crossfade_samples, dtype=np.float32)
        else:
            self._fade_in = np.array([], dtype=np.float32)
            self._fade_out = np.array([], dtype=np.float32)

    def _apply_crossfade(self, output: np.ndarray) -> np.ndarray:
        """
        Apply crossfade between consecutive output chunks.

        Blends the end of the previous chunk with the start of the current chunk
        to reduce audible discontinuities at chunk boundaries.
        """
        # TEMPORARILY DISABLED for debugging echo issue
        # TODO: Re-enable after fixing the echo problem
        return output

        # cf = self.output_crossfade_samples
        #
        # if cf == 0 or self._prev_output is None:
        #     self._prev_output = output.copy()
        #     return output
        #
        # result = output.copy()
        #
        # # Crossfade: blend end of previous chunk with start of current
        # if len(self._prev_output) >= cf and len(result) >= cf:
        #     prev_tail = self._prev_output[-cf:]
        #     curr_head = result[:cf]
        #     blended = prev_tail * self._fade_out + curr_head * self._fade_in
        #     result[:cf] = blended
        #
        # self._prev_output = output.copy()
        # return result

    def _store_output_history(self, output: np.ndarray) -> None:
        """Store output samples for feedback detection."""
        # Resample to mic rate if different for comparison
        if self.config.output_sample_rate != self.config.mic_sample_rate:
            output = resample(output, self.config.output_sample_rate, self.config.mic_sample_rate)

        # Store in circular buffer
        out_len = len(output)
        if out_len >= self._output_history_size:
            # Output larger than buffer - just store the last part
            self._output_history[:] = output[-self._output_history_size:]
            self._output_history_pos = 0
        else:
            # Fit into circular buffer
            end_pos = self._output_history_pos + out_len
            if end_pos <= self._output_history_size:
                self._output_history[self._output_history_pos:end_pos] = output
            else:
                # Wrap around
                first_part = self._output_history_size - self._output_history_pos
                self._output_history[self._output_history_pos:] = output[:first_part]
                self._output_history[:out_len - first_part] = output[first_part:]
            self._output_history_pos = end_pos % self._output_history_size

    def _check_feedback(self, input_audio: np.ndarray) -> float:
        """
        Check for feedback by computing cross-correlation between input and output history.

        Returns correlation coefficient (0-1). High values (>0.5) indicate feedback.
        """
        if len(input_audio) < 1000:
            return 0.0

        # Only check if there's significant signal in input
        input_rms = np.sqrt(np.mean(input_audio**2))
        if input_rms < 0.01:  # Too quiet to detect
            return 0.0

        output_rms = np.sqrt(np.mean(self._output_history**2))
        if output_rms < 0.01:  # No output yet
            return 0.0

        # Normalize both signals
        input_norm = input_audio - np.mean(input_audio)
        output_norm = self._output_history - np.mean(self._output_history)

        input_std = np.std(input_norm)
        output_std = np.std(output_norm)

        if input_std < 1e-6 or output_std < 1e-6:
            return 0.0

        input_norm = input_norm / input_std
        output_norm = output_norm / output_std

        # Compute cross-correlation using FFT for efficiency
        # Look for correlation at various delays (100ms to 1s)
        try:
            # Use a subset of output history for speed
            check_len = min(len(input_audio), self.config.mic_sample_rate // 2)
            corr = np.correlate(input_norm[:check_len], output_norm[:check_len], mode='valid')
            max_corr = np.max(np.abs(corr)) / check_len
            return float(max_corr)
        except Exception:
            return 0.0

    def _on_audio_input(self, audio: np.ndarray) -> None:
        """Callback for audio input."""
        # Debug: log input audio signature to detect feedback
        if self.stats.frames_processed < 20:
            audio_hash = hash(audio[:100].tobytes()) % 10000
            rms = np.sqrt(np.mean(audio**2))
            logger.info(
                f"[INPUT-RAW] len={len(audio)}, hash={audio_hash}, "
                f"rms={rms:.6f}, first5={audio[:5]}"
            )

        # Add raw audio to buffer (resampling done in inference thread)
        self.input_buffer.add_input(audio)

        # Log input buffer state periodically
        if self.stats.frames_processed < 5 or self.stats.frames_processed % 50 == 0:
            logger.debug(
                f"[INPUT] received={len(audio)}, "
                f"input_buffer={self.input_buffer.buffered_samples}, "
                f"input_queue={self._input_queue.qsize()}"
            )

        # Process ALL available chunks (not just one) to prevent input buffer buildup
        chunks_queued = 0
        while self.input_buffer.has_chunk():
            chunk = self.input_buffer.get_chunk()
            if chunk is not None:
                try:
                    self._input_queue.put_nowait(chunk)
                    chunks_queued += 1
                except Exception:
                    self.stats.buffer_overruns += 1
                    logger.warning(f"[INPUT] Queue full, dropping chunk")
                    break  # Queue full, stop trying

        # Debug: log if multiple chunks were queued (indicates we're falling behind)
        if chunks_queued > 1:
            logger.warning(f"[INPUT] Falling behind: queued {chunks_queued} chunks at once")

    def _on_audio_output(self, frames: int) -> np.ndarray:
        """Callback for audio output."""
        # Check for new processed audio
        chunks_added = 0
        total_dropped = 0
        try:
            while True:
                audio = self._output_queue.get_nowait()
                dropped = self.output_buffer.add(audio)
                self._chunks_ready += 1
                chunks_added += 1
                if dropped > 0:
                    # Old samples were dropped to catch up to real-time
                    self.stats.buffer_overruns += 1
                    total_dropped += dropped
        except Empty:
            pass

        # Log output state periodically
        if self.stats.frames_processed < 10 or self.stats.frames_processed % 50 == 0:
            logger.debug(
                f"[OUTPUT] frames={frames}, chunks_added={chunks_added}, "
                f"output_buffer={self.output_buffer.available}, "
                f"output_queue={self._output_queue.qsize()}, "
                f"dropped={total_dropped}"
            )

        # Wait for pre-buffering before starting output
        if not self._output_started:
            if self._chunks_ready >= self._prebuffer_chunks:
                self._output_started = True
                logger.info(f"Pre-buffering complete, starting output ({self.output_buffer.available} samples)")
            else:
                # Return silence while pre-buffering
                return np.zeros(frames, dtype=np.float32)

        # Get output samples
        output = self.output_buffer.get(frames)

        if self.output_buffer.available == 0:
            self.stats.buffer_underruns += 1
            if self.stats.buffer_underruns <= 5 or self.stats.buffer_underruns % 20 == 0:
                logger.warning(f"[OUTPUT] Buffer underrun #{self.stats.buffer_underruns}")

        return output

    def _inference_thread(self) -> None:
        """Background thread for inference processing."""
        logger.info("Inference thread started")

        # Log audio flow for debugging sample rate issues
        logger.info(
            f"Audio flow: mic({self.config.mic_sample_rate}Hz) -> "
            f"process({self.config.input_sample_rate}Hz) -> "
            f"model({self.pipeline.sample_rate}Hz) -> "
            f"output({self.config.output_sample_rate}Hz)"
        )

        if self.config.input_sample_rate != 16000:
            logger.warning(
                f"Input sample rate {self.config.input_sample_rate}Hz differs from "
                "expected 16kHz for HuBERT/RMVPE"
            )

        # Ratio for output resampling
        output_ratio = self.pipeline.sample_rate / self.config.output_sample_rate

        while self._running:
            try:
                # Get input chunk (at mic sample rate)
                chunk = self._input_queue.get(timeout=0.5)

                # Process timing
                start_time = time.perf_counter()

                # Apply input gain
                if self.config.input_gain_db != 0.0:
                    gain_linear = 10 ** (self.config.input_gain_db / 20)
                    chunk = chunk * gain_linear

                # Store raw input chunk at mic rate for feedback detection
                chunk_at_mic_rate = chunk.copy()

                # Debug: log input chunk stats
                if self.stats.frames_processed < 3:
                    rms = np.sqrt(np.mean(chunk**2))
                    logger.info(
                        f"Raw input chunk: len={len(chunk)}, "
                        f"min={chunk.min():.4f}, max={chunk.max():.4f}, "
                        f"rms={rms:.4f}"
                    )

                # Resample from mic rate to processing rate
                if self.config.mic_sample_rate != self.config.input_sample_rate:
                    chunk = resample(
                        chunk,
                        self.config.mic_sample_rate,
                        self.config.input_sample_rate,
                    )
                    if self.stats.frames_processed < 3:
                        logger.info(
                            f"After resample: len={len(chunk)}, "
                            f"min={chunk.min():.4f}, max={chunk.max():.4f}"
                        )

                # Apply noise cancellation if enabled
                if self.config.denoise_enabled:
                    if self.stats.frames_processed < 3:
                        logger.info(
                            f"Applying denoise (method={self.config.denoise_method})"
                        )
                    chunk = denoise_audio(
                        chunk,
                        sample_rate=self.config.input_sample_rate,
                        method=self.config.denoise_method,
                        device="cpu",  # ML denoiser runs on CPU for stability
                    )
                    if self.stats.frames_processed < 3:
                        logger.info(
                            f"After denoise: len={len(chunk)}, "
                            f"min={chunk.min():.4f}, max={chunk.max():.4f}"
                        )

                # Run inference
                output = self.pipeline.infer(
                    chunk,
                    input_sr=self.config.input_sample_rate,
                    pitch_shift=self.config.pitch_shift,
                    f0_method="rmvpe" if self.config.use_f0 else "none",
                    index_rate=self.config.index_rate,
                )

                # Resample to output sample rate first
                if self.pipeline.sample_rate != self.config.output_sample_rate:
                    output = resample(
                        output,
                        self.pipeline.sample_rate,
                        self.config.output_sample_rate,
                    )

                # Remove DC offset (can cause clicks between chunks)
                output = output - np.mean(output)

                # Soft clipping to prevent harsh distortion
                max_val = np.max(np.abs(output))
                if max_val > 1.0:
                    output = np.tanh(output)  # Soft clip
                    if self.stats.frames_processed <= 5:
                        logger.warning(f"Audio clipping detected: max={max_val:.2f}")

                # Apply crossfade for smooth chunk transitions
                output = self._apply_crossfade(output)

                # Debug: log output audio signature to detect feedback
                if self.stats.frames_processed < 20:
                    output_hash = hash(output[:100].tobytes()) % 10000
                    output_rms = np.sqrt(np.mean(output**2))
                    logger.info(
                        f"[OUTPUT-PROC] len={len(output)}, hash={output_hash}, "
                        f"rms={output_rms:.6f}, first5={output[:5]}"
                    )

                # Store output for feedback detection
                self._store_output_history(output)

                # Check for feedback periodically
                if self.stats.frames_processed > 0 and self.stats.frames_processed % self._feedback_check_interval == 0:
                    # Get raw input chunk for comparison (at mic rate)
                    raw_input = chunk_at_mic_rate if 'chunk_at_mic_rate' in locals() else chunk
                    correlation = self._check_feedback(raw_input)
                    self.stats.feedback_correlation = correlation

                    if correlation > 0.3 and not self._feedback_warning_shown:
                        self.stats.feedback_detected = True
                        self._feedback_warning_shown = True
                        logger.warning(
                            f"[FEEDBACK] 音声フィードバックを検出しました (相関係数={correlation:.2f}). "
                            "Windowsの「このデバイスを聴く」が有効になっていないか確認してください。"
                        )
                        if self.on_error:
                            self.on_error(
                                "フィードバック検出: 入力と出力が接続されている可能性があります。\n"
                                "「サウンド設定」→「録音デバイス」→「プロパティ」→「聴く」タブで\n"
                                "「このデバイスを聴く」が無効になっているか確認してください。"
                            )

                # Update stats
                inference_time = time.perf_counter() - start_time
                self.stats.inference_ms = inference_time * 1000
                self.stats.frames_processed += 1

                # Log chunks for debugging
                if self.stats.frames_processed <= 10 or self.stats.frames_processed % 20 == 0:
                    logger.info(
                        f"[INFER] Chunk #{self.stats.frames_processed}: "
                        f"in={len(chunk)}, out={len(output)}, "
                        f"infer={self.stats.inference_ms:.0f}ms, "
                        f"latency={self.stats.latency_ms:.0f}ms, "
                        f"buf={self.output_buffer.available}, "
                        f"under={self.stats.buffer_underruns}, over={self.stats.buffer_overruns}"
                    )

                # Calculate latency
                self.stats.latency_ms = (
                    self.config.chunk_sec * 1000
                    + self.stats.inference_ms
                    + (self.output_buffer.available / self.config.output_sample_rate)
                    * 1000
                )

                # Send to output (block briefly if queue is full)
                try:
                    self._output_queue.put(output, timeout=0.1)
                except Exception:
                    self.stats.buffer_overruns += 1
                    logger.warning("Output queue full, dropping chunk")

                # Call stats callback
                if self.on_stats_update:
                    self.on_stats_update(self.stats)

            except Empty:
                continue
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Inference error: {error_msg}")

                # Notify UI of error (throttle to avoid spam)
                self._error_count += 1
                if self._last_error != error_msg or self._error_count % 10 == 1:
                    self._last_error = error_msg
                    if self.on_error:
                        self.on_error(f"推論エラー: {error_msg}")
                continue

        logger.info("Inference thread stopped")

    def start(self) -> None:
        """Start real-time voice conversion."""
        if self._running:
            return

        logger.info("Starting real-time voice changer...")

        # Validate chunk size for RMVPE (needs >= 32 mel frames)
        min_chunk_sec = 0.32  # 32 frames * 160 hop / 16000 Hz
        if self.config.use_f0 and self.config.chunk_sec < min_chunk_sec:
            old_chunk = self.config.chunk_sec
            self.config.chunk_sec = 0.35
            logger.warning(
                f"Chunk size {old_chunk}s too small for RMVPE, increased to {self.config.chunk_sec}s"
            )
            # Recalculate buffer sizes
            self.mic_chunk_samples = int(self.config.mic_sample_rate * self.config.chunk_sec)
            self.input_buffer = ChunkBuffer(
                self.mic_chunk_samples,
                crossfade_samples=0,
                context_samples=0,
            )

        # Ensure pipeline is loaded
        if not self.pipeline._loaded:
            self.pipeline.load()

        # Reset stats and buffers
        self.stats.reset()
        self.input_buffer.clear()
        self.output_buffer.clear()

        # Reset pre-buffering state
        self._chunks_ready = 0
        self._output_started = False
        self._prev_output = None

        # Reset feedback detection state
        self._output_history.fill(0)
        self._output_history_pos = 0
        self._feedback_warning_shown = False

        # Clear queues
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except Empty:
                break
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except Empty:
                break

        # Start inference thread
        self._running = True
        self._thread = threading.Thread(
            target=self._inference_thread,
            daemon=True,
            name="RCWX-Inference",
        )
        self._thread.start()

        # Calculate output blocksize
        output_chunk_sec = self.config.chunk_sec / 4
        output_blocksize = int(self.config.output_sample_rate * output_chunk_sec)

        # Start audio input at mic's native rate (resample to 16kHz in callback)
        logger.info(
            f"Audio config: mic_sr={self.config.mic_sample_rate}, "
            f"out_sr={self.config.output_sample_rate}, "
            f"chunk={self.config.chunk_sec}s, "
            f"mic_chunk_samples={self.mic_chunk_samples}"
        )

        self.audio_input = AudioInput(
            device=self.config.input_device,
            sample_rate=self.config.mic_sample_rate,
            blocksize=int(self.config.mic_sample_rate * output_chunk_sec),
            callback=self._on_audio_input,
        )
        self.audio_input.start()

        # Start audio output
        self.audio_output = AudioOutput(
            device=self.config.output_device,
            sample_rate=self.config.output_sample_rate,
            blocksize=output_blocksize,
            callback=self._on_audio_output,
        )
        self.audio_output.start()

        logger.info("Real-time voice changer started")

    def stop(self) -> None:
        """Stop real-time voice conversion."""
        if not self._running:
            return

        logger.info("Stopping real-time voice changer...")

        self._running = False

        # Stop audio streams
        if self.audio_input:
            self.audio_input.stop()
            self.audio_input = None

        if self.audio_output:
            self.audio_output.stop()
            self.audio_output = None

        # Wait for inference thread
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        logger.info("Real-time voice changer stopped")

    def set_pitch_shift(self, semitones: int) -> None:
        """Set pitch shift in semitones."""
        self.config.pitch_shift = semitones

    def set_f0_mode(self, use_f0: bool) -> None:
        """Set F0 mode (True for RMVPE, False for no-F0)."""
        self.config.use_f0 = use_f0

    def set_denoise(self, enabled: bool, method: str = "auto") -> None:
        """Set noise cancellation settings.

        Args:
            enabled: Enable/disable noise cancellation
            method: "auto", "deepfilter", or "spectral"
        """
        self.config.denoise_enabled = enabled
        self.config.denoise_method = method

    def set_index_rate(self, index_rate: float) -> None:
        """Set FAISS index blending rate.

        Args:
            index_rate: 0=disabled, 0.5=balanced, 1=index only
        """
        self.config.index_rate = index_rate

    @property
    def is_running(self) -> bool:
        """Check if voice changer is running."""
        return self._running
