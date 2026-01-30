"""Integration test for realtime voice conversion pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time

from rcwx.audio.buffer import ChunkBuffer, OutputBuffer
from rcwx.audio.resample import resample


class TestChunkBuffer:
    """Test ChunkBuffer functionality."""

    def test_basic_chunking(self):
        """Test basic chunk accumulation and retrieval."""
        chunk_samples = 1000
        buffer = ChunkBuffer(chunk_samples, crossfade_samples=0)

        # Add samples in small batches
        for _ in range(5):
            buffer.add_input(np.random.randn(250).astype(np.float32))

        assert buffer.buffered_samples == 1250
        assert buffer.has_chunk()

        chunk = buffer.get_chunk()
        assert chunk is not None
        assert len(chunk) == chunk_samples
        assert buffer.buffered_samples == 250  # Remaining

    def test_chunk_continuity(self):
        """Test that chunks are continuous (no gaps or overlaps)."""
        chunk_samples = 1000
        buffer = ChunkBuffer(chunk_samples, crossfade_samples=0)

        # Create known signal
        total_samples = 5000
        signal = np.arange(total_samples, dtype=np.float32)

        # Add all at once
        buffer.add_input(signal)

        # Extract chunks
        chunks = []
        while buffer.has_chunk():
            chunks.append(buffer.get_chunk())

        # Verify chunks are continuous
        reconstructed = np.concatenate(chunks)
        expected = signal[:len(reconstructed)]
        np.testing.assert_array_equal(reconstructed, expected)


class TestOutputBuffer:
    """Test OutputBuffer functionality."""

    def test_basic_output(self):
        """Test basic add and get."""
        buffer = OutputBuffer(max_latency_samples=10000, fade_samples=0)

        audio = np.random.randn(1000).astype(np.float32)
        buffer.add(audio)

        assert buffer.available == 1000

        output = buffer.get(500)
        assert len(output) == 500
        assert buffer.available == 500

    def test_underrun_zeros(self):
        """Test that underrun returns zeros."""
        buffer = OutputBuffer(max_latency_samples=10000, fade_samples=0)

        output = buffer.get(1000)
        assert len(output) == 1000
        assert np.all(output == 0)


class TestResample:
    """Test resampling functionality."""

    def test_downsample_48k_to_16k(self):
        """Test downsampling from 48kHz to 16kHz."""
        # Create 1 second of 48kHz audio
        duration = 1.0
        orig_sr = 48000
        target_sr = 16000

        t = np.linspace(0, duration, int(orig_sr * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave

        resampled = resample(audio, orig_sr, target_sr)

        expected_len = int(len(audio) * target_sr / orig_sr)
        assert abs(len(resampled) - expected_len) <= 1

    def test_upsample_40k_to_48k(self):
        """Test upsampling from 40kHz to 48kHz."""
        duration = 0.35
        orig_sr = 40000
        target_sr = 48000

        audio = np.random.randn(int(orig_sr * duration)).astype(np.float32)
        resampled = resample(audio, orig_sr, target_sr)

        expected_len = int(len(audio) * target_sr / orig_sr)
        assert abs(len(resampled) - expected_len) <= 1


class TestRealtimeDataFlow:
    """Test the complete realtime data flow."""

    def test_chunk_size_calculation(self):
        """Test that chunk sizes are calculated correctly."""
        mic_sr = 44100
        chunk_sec = 0.35

        mic_chunk_samples = int(mic_sr * chunk_sec)
        assert mic_chunk_samples == 15435

        # After resample to 16kHz
        processing_sr = 16000
        processing_samples = int(mic_chunk_samples * processing_sr / mic_sr)
        # Using resample function
        test_audio = np.zeros(mic_chunk_samples, dtype=np.float32)
        resampled = resample(test_audio, mic_sr, processing_sr)
        assert abs(len(resampled) - int(chunk_sec * processing_sr)) <= 10

    def test_output_length_calculation(self):
        """Test that output length matches input duration."""
        input_sr = 16000
        output_sr = 40000
        chunk_sec = 0.35

        input_samples = int(input_sr * chunk_sec)  # 5600
        expected_output = int(input_samples * output_sr / input_sr)  # 14000

        assert input_samples == 5600
        assert expected_output == 14000

    def test_feature_interpolation_ratio(self):
        """Test HuBERT to Synthesizer feature ratio."""
        input_samples = 5600  # 350ms @ 16kHz

        # HuBERT: hop=320, so frames = input_samples / 320
        hubert_frames = input_samples // 320  # 17

        # Expected output @ 40kHz
        output_sr = 40000
        input_sr = 16000
        expected_output = int(input_samples * output_sr / input_sr)  # 14000

        # Synthesizer upsample rate
        upsample_rate = 400
        expected_frames = expected_output // upsample_rate  # 35

        assert hubert_frames == 17
        assert expected_frames == 35
        assert expected_frames == hubert_frames * 2 + 1  # Roughly 2x

    def test_full_chunk_pipeline_simulation(self):
        """Simulate full chunk processing pipeline."""
        # Config
        mic_sr = 44100
        processing_sr = 16000
        output_sr = 40000
        final_sr = 48000
        chunk_sec = 0.35

        # 1. Simulate mic input
        mic_samples = int(mic_sr * chunk_sec)
        mic_audio = np.sin(
            2 * np.pi * 440 * np.linspace(0, chunk_sec, mic_samples)
        ).astype(np.float32)

        # 2. Resample to processing rate
        processed = resample(mic_audio, mic_sr, processing_sr)
        assert abs(len(processed) - int(processing_sr * chunk_sec)) <= 10

        # 3. Simulate inference output (should be chunk_sec duration at output_sr)
        inference_output_samples = int(chunk_sec * output_sr)
        inference_output = np.random.randn(inference_output_samples).astype(np.float32)

        # 4. Resample to final output rate
        final_output = resample(inference_output, output_sr, final_sr)
        expected_final = int(chunk_sec * final_sr)
        assert abs(len(final_output) - expected_final) <= 10

        print(f"Mic: {len(mic_audio)} @ {mic_sr}Hz")
        print(f"Processed: {len(processed)} @ {processing_sr}Hz")
        print(f"Inference out: {len(inference_output)} @ {output_sr}Hz")
        print(f"Final: {len(final_output)} @ {final_sr}Hz")


class TestBufferContinuity:
    """Test that audio chunks connect properly."""

    def test_continuous_sine_wave(self):
        """Test that a continuous sine wave remains continuous through chunking."""
        # Create 2 seconds of continuous sine wave
        sr = 16000
        duration = 2.0
        freq = 440

        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        continuous_audio = np.sin(2 * np.pi * freq * t)

        # Process through ChunkBuffer
        chunk_samples = int(sr * 0.35)
        buffer = ChunkBuffer(chunk_samples, crossfade_samples=0)
        buffer.add_input(continuous_audio)

        chunks = []
        while buffer.has_chunk():
            chunks.append(buffer.get_chunk().copy())

        # Reconstruct
        reconstructed = np.concatenate(chunks)

        # Check continuity at boundaries
        for i in range(len(chunks) - 1):
            end_of_chunk = chunks[i][-10:]
            start_of_next = chunks[i + 1][:10]

            # The difference should be small for a continuous signal
            # (accounting for the sine wave progression)
            boundary = np.concatenate([end_of_chunk, start_of_next])
            diff = np.diff(boundary)
            max_diff = np.max(np.abs(diff))

            # For a 440Hz sine at 16kHz, max slope is 2*pi*440/16000 ≈ 0.17
            assert max_diff < 0.2, f"Discontinuity at chunk boundary {i}: max_diff={max_diff}"


def test_actual_gui_code_path():
    """
    Test using ACTUAL RealtimeVoiceChanger code path.
    This is exactly what happens when GUI processes audio.
    """
    import sys
    from pathlib import Path
    from scipy.io import wavfile

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from rcwx.config import RCWXConfig
    from rcwx.pipeline.inference import RVCPipeline
    from rcwx.pipeline.realtime import RealtimeConfig, RealtimeVoiceChanger
    from rcwx.audio.crossfade import CrossfadeState, apply_crossfade, trim_edges

    print("=" * 70)
    print("Actual GUI Code Path Test")
    print("=" * 70)

    # Load config
    config = RCWXConfig.load()
    if not config.last_model_path:
        print("SKIP: No model configured")
        return

    print(f"\nModel: {config.last_model_path}")

    # Create pipeline
    pipeline = RVCPipeline(
        config.last_model_path,
        device=config.device,
        use_compile=False,
    )
    pipeline.load()

    # Create RealtimeVoiceChanger exactly like GUI does
    rt_config = RealtimeConfig(
        mic_sample_rate=48000,
        output_sample_rate=48000,
        chunk_sec=config.audio.chunk_sec or 0.5,
        pitch_shift=config.inference.pitch_shift,
        use_f0=config.inference.use_f0,
        f0_method=config.inference.f0_method,
        index_rate=config.inference.index_ratio if config.inference.use_index else 0.0,
        voice_gate_mode=config.inference.voice_gate_mode,
        energy_threshold=config.inference.energy_threshold,
        use_feature_cache=config.inference.use_feature_cache,
        context_sec=config.inference.context_sec,
        extra_sec=config.inference.extra_sec,
        crossfade_sec=config.inference.crossfade_sec,
        lookahead_sec=config.inference.lookahead_sec,
        use_sola=config.inference.use_sola,
        prebuffer_chunks=1,
    )

    print(f"\nConfig:")
    print(f"  chunk_sec={rt_config.chunk_sec}")
    print(f"  context_sec={rt_config.context_sec}")
    print(f"  crossfade_sec={rt_config.crossfade_sec}")

    changer = RealtimeVoiceChanger(pipeline, config=rt_config)

    # Load test audio
    voice_path = Path("sample_data/kakita.wav")
    if not voice_path.exists():
        voice_path = Path("sample_data/pure_sine.wav")
    if not voice_path.exists():
        print("SKIP: No test audio")
        return

    sr, voice_data = wavfile.read(voice_path)
    if voice_data.dtype == np.int16:
        voice_data = voice_data.astype(np.float32) / 32768.0
    if len(voice_data.shape) > 1:
        voice_data = voice_data[:, 0]
    voice_data = voice_data[:int(5 * sr)]  # First 5 seconds
    print(f"Input: {voice_path}, {len(voice_data)/sr:.2f}s @ {sr}Hz")

    # Initialize like start() does
    pipeline.clear_cache()
    changer.mic_chunk_samples = int(changer.config.mic_sample_rate * changer.config.chunk_sec)
    changer.mic_context_samples = int(changer.config.mic_sample_rate * changer.config.context_sec)
    changer.mic_lookahead_samples = int(changer.config.mic_sample_rate * changer.config.lookahead_sec)

    changer.input_buffer = ChunkBuffer(
        changer.mic_chunk_samples,
        crossfade_samples=0,
        context_samples=changer.mic_context_samples,
        lookahead_samples=changer.mic_lookahead_samples,
    )

    changer.output_crossfade_samples = int(
        changer.config.output_sample_rate * changer.config.crossfade_sec
    )
    changer.output_extra_samples = int(
        changer.config.output_sample_rate * changer.config.extra_sec
    )
    changer.output_context_samples = int(
        changer.config.output_sample_rate * changer.config.context_sec
    )

    changer._crossfade_state = CrossfadeState(cf_samples=changer.output_crossfade_samples)

    # Warmup
    print("Warming up...")
    mic_total = (changer.mic_chunk_samples + changer.mic_context_samples +
                 changer.mic_lookahead_samples)
    warmup_samples = int(mic_total * changer.config.input_sample_rate /
                        changer.config.mic_sample_rate)

    warmup_audio = np.zeros(warmup_samples, dtype=np.float32)
    _ = pipeline.infer(warmup_audio, input_sr=changer.config.input_sample_rate,
                       pitch_shift=0, f0_method=changer.config.f0_method if changer.config.use_f0 else "none",
                       index_rate=0.0, voice_gate_mode="off", use_feature_cache=False)

    t = np.arange(warmup_samples) / changer.config.input_sample_rate
    warmup_audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    for _ in range(3):
        _ = pipeline.infer(warmup_audio, input_sr=changer.config.input_sample_rate,
                           pitch_shift=changer.config.pitch_shift,
                           f0_method=changer.config.f0_method if changer.config.use_f0 else "none",
                           index_rate=changer.config.index_rate,
                           voice_gate_mode=changer.config.voice_gate_mode,
                           use_feature_cache=True)

    # Reset
    changer.stats.reset()
    changer.input_buffer.clear()
    changer.output_buffer.clear()
    changer._chunks_ready = 0
    changer._crossfade_state = CrossfadeState(cf_samples=changer.output_crossfade_samples)

    # Process using ACTUAL _inference_thread logic
    print("Processing...")
    outputs = []
    mic_block_size = 4096
    input_pos = 0
    chunk_count = 0

    while input_pos < len(voice_data):
        block = voice_data[input_pos:input_pos + mic_block_size]
        if len(block) < mic_block_size:
            block = np.pad(block, (0, mic_block_size - len(block)))
        changer.input_buffer.add_input(block)
        input_pos += mic_block_size

        while changer.input_buffer.has_chunk():
            chunk = changer.input_buffer.get_chunk()
            if chunk is None:
                break

            # EXACT CODE FROM _inference_thread
            if changer.config.input_gain_db != 0.0:
                gain_linear = 10 ** (changer.config.input_gain_db / 20)
                chunk = chunk * gain_linear

            if changer.config.mic_sample_rate != changer.config.input_sample_rate:
                chunk = resample(chunk, changer.config.mic_sample_rate, changer.config.input_sample_rate)

            output = pipeline.infer(
                chunk, input_sr=changer.config.input_sample_rate,
                pitch_shift=changer.config.pitch_shift,
                f0_method=changer.config.f0_method if changer.config.use_f0 else "none",
                index_rate=changer.config.index_rate,
                voice_gate_mode=changer.config.voice_gate_mode,
                energy_threshold=changer.config.energy_threshold,
                use_feature_cache=changer.config.use_feature_cache,
            )

            if pipeline.sample_rate != changer.config.output_sample_rate:
                output = resample(output, pipeline.sample_rate, changer.config.output_sample_rate)

            max_val = np.max(np.abs(output))
            if max_val > 1.0:
                output = np.tanh(output)

            trimmed_output = trim_edges(
                output,
                context_samples=changer.output_context_samples,
                extra_samples=changer.output_extra_samples,
            )

            cf_result = apply_crossfade(
                trimmed_output,
                changer._crossfade_state,
                use_sola=changer.config.use_sola,
                sola_search_ratio=changer.config.sola_search_ratio,
            )

            outputs.append(cf_result.audio)
            chunk_count += 1
            if chunk_count <= 3:
                print(f"  Chunk {chunk_count}: len={len(cf_result.audio)}")

    print(f"Total chunks: {chunk_count}")

    if len(outputs) < 2:
        print("ERROR: Not enough output")
        return

    full_output = np.concatenate(outputs)
    print(f"Total output: {len(full_output)/48000:.2f}s")

    # Save
    out_path = Path("tests/test_gui_path_output.wav")
    max_val = np.abs(full_output).max()
    if max_val > 0:
        full_output = full_output / max_val * 0.9
    wavfile.write(out_path, 48000, (full_output * 32767).astype(np.int16))
    print(f"Saved: {out_path}")

    # Analyze
    print("\n--- Boundary Jumps ---")
    pos = 0
    jumps = []
    for i, o in enumerate(outputs):
        if i > 0 and i < 10:
            jump = abs(full_output[pos] - full_output[pos-1])
            jumps.append(jump)
            print(f"  {i}: {jump:.6f}")
        pos += len(o)

    max_jump = max(jumps) if jumps else 0
    print(f"\nMax jump: {max_jump:.6f} (target < 0.1)")
    print(f"Result: {'PASS' if max_jump < 0.1 else 'FAIL'}")


if __name__ == "__main__":
    # Run the actual GUI code path test
    test_actual_gui_code_path()
