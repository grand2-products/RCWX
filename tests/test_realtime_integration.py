"""Integration test for realtime voice conversion pipeline."""

import numpy as np
import pytest
import time
from unittest.mock import MagicMock, patch

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
