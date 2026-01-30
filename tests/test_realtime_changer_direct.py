"""
Direct test of RealtimeVoiceChanger with simulated audio input.

This test simulates the ACTUAL GUI behavior by:
1. Creating RealtimeVoiceChanger with real config
2. Feeding audio chunks like the mic callback does
3. Checking for buffer underruns
"""

import sys
import time
import numpy as np
from pathlib import Path
from scipy.io import wavfile

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeConfig, RealtimeVoiceChanger


def test_realtime_direct():
    """Test RealtimeVoiceChanger directly with simulated mic input."""
    print("=" * 70)
    print("Direct RealtimeVoiceChanger Test")
    print("=" * 70)

    # Load config (same as GUI)
    config = RCWXConfig.load()
    if not config.last_model_path:
        print("ERROR: No model configured")
        return False

    # Create pipeline (same as GUI)
    print(f"\nLoading model: {config.last_model_path}")
    pipeline = RVCPipeline(
        config.last_model_path,
        device=config.device,
        use_compile=config.inference.use_compile,
    )
    pipeline.load()

    # Create RealtimeConfig (same as GUI)
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
    )

    print(f"\nRealtimeConfig:")
    print(f"  chunk_sec={rt_config.chunk_sec}")
    print(f"  context_sec={rt_config.context_sec}")
    print(f"  lookahead_sec={rt_config.lookahead_sec}")
    print(f"  crossfade_sec={rt_config.crossfade_sec}")
    print(f"  prebuffer_chunks={rt_config.prebuffer_chunks}")
    print(f"  max_queue_size={rt_config.max_queue_size}")

    # Create voice changer (same as GUI)
    changer = RealtimeVoiceChanger(pipeline, config=rt_config)

    # Load test audio
    voice_path = Path("sample_data/pure_sine.wav")
    if not voice_path.exists():
        print(f"ERROR: Test file not found: {voice_path}")
        return False

    sr, voice_data = wavfile.read(voice_path)
    if voice_data.dtype == np.int16:
        voice_data = voice_data.astype(np.float32) / 32768.0
    print(f"\nInput: {voice_path}, {len(voice_data)/sr:.2f}s @ {sr}Hz")

    # Warmup inference (like start() does - 4 passes)
    print("\nWarming up inference...")
    warmup_start = time.perf_counter()
    # Calculate actual input size (same as ChunkBuffer output after resample)
    # ChunkBuffer returns: chunk_samples + context_samples + lookahead_samples
    # where chunk_samples = mic_chunk (after fix)
    # So total = mic_chunk + mic_context + mic_lookahead
    mic_chunk = int(rt_config.mic_sample_rate * rt_config.chunk_sec)
    mic_context = int(rt_config.mic_sample_rate * rt_config.context_sec)
    mic_lookahead = int(rt_config.mic_sample_rate * rt_config.lookahead_sec)
    mic_total = mic_chunk + mic_context + mic_lookahead
    warmup_samples = int(mic_total * rt_config.input_sample_rate / rt_config.mic_sample_rate)
    print(f"  Warmup size: {warmup_samples} samples @ {rt_config.input_sample_rate}Hz")

    # Warmup 1: silence
    warmup_audio = np.zeros(warmup_samples, dtype=np.float32)
    _ = pipeline.infer(
        warmup_audio,
        input_sr=rt_config.input_sample_rate,
        pitch_shift=0,
        f0_method=rt_config.f0_method if rt_config.use_f0 else "none",
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=False,
    )
    print(f"  Warmup 1 (silence): {(time.perf_counter()-warmup_start)*1000:.0f}ms")

    # Warmup 2-4: tone with index (same as realtime.py)
    t = np.arange(warmup_samples) / rt_config.input_sample_rate
    warmup_audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    for i in range(3):
        warmup_start2 = time.perf_counter()
        _ = pipeline.infer(
            warmup_audio,
            input_sr=rt_config.input_sample_rate,
            pitch_shift=rt_config.pitch_shift,
            f0_method=rt_config.f0_method if rt_config.use_f0 else "none",
            index_rate=rt_config.index_rate,
            voice_gate_mode=rt_config.voice_gate_mode,
            use_feature_cache=True,
        )
        print(f"  Warmup {i+2} (tone+index): {(time.perf_counter()-warmup_start2)*1000:.0f}ms")
    print(f"Total warmup: {(time.perf_counter()-warmup_start)*1000:.0f}ms")

    # Simulate mic input/output callbacks
    mic_block_size = 4096  # Typical mic callback size
    output_block_size = 1024  # Typical output callback size

    outputs = []
    input_pos = 0
    total_input_time = 0.0
    total_output_time = 0.0

    print("\nSimulating real-time processing...")
    print("(Feeding audio and collecting output)")

    # Initialize buffers like start() does
    changer._running = True
    changer.input_buffer.clear()
    changer.output_buffer.clear()
    changer._chunks_ready = 0
    changer._output_started = False
    changer._crossfade_state.prev_tail = None
    changer._crossfade_state.frames_processed = 0

    # Process in a loop simulating real-time
    start_time = time.perf_counter()
    chunk_times = []

    while input_pos < len(voice_data):
        # Simulate mic callback (input arrives)
        block = voice_data[input_pos:input_pos + mic_block_size]
        if len(block) < mic_block_size:
            block = np.pad(block, (0, mic_block_size - len(block)))

        # Feed to input buffer (like _on_audio_input)
        changer.input_buffer.add_input(block)

        # Process available chunks (like _inference_thread, but synchronous)
        while changer.input_buffer.has_chunk():
            chunk = changer.input_buffer.get_chunk()
            if chunk is None:
                break

            chunk_start = time.perf_counter()

            # This is what _inference_thread does
            from rcwx.audio.resample import resample
            from rcwx.audio.crossfade import apply_crossfade, trim_edges

            # Resample to processing rate
            if rt_config.mic_sample_rate != rt_config.input_sample_rate:
                chunk = resample(chunk, rt_config.mic_sample_rate, rt_config.input_sample_rate)

            # Inference
            output = pipeline.infer(
                chunk,
                input_sr=rt_config.input_sample_rate,
                pitch_shift=rt_config.pitch_shift,
                f0_method=rt_config.f0_method if rt_config.use_f0 else "none",
                index_rate=rt_config.index_rate,
                voice_gate_mode=rt_config.voice_gate_mode,
                energy_threshold=rt_config.energy_threshold,
                use_feature_cache=rt_config.use_feature_cache,
            )

            # Resample to output rate
            if pipeline.sample_rate != rt_config.output_sample_rate:
                output = resample(output, pipeline.sample_rate, rt_config.output_sample_rate)

            # Trim edges
            out_context = int(rt_config.output_sample_rate * rt_config.context_sec)
            out_extra = int(rt_config.output_sample_rate * rt_config.extra_sec)
            trimmed = trim_edges(output, out_context, out_extra)

            # Crossfade
            cf_result = apply_crossfade(
                trimmed,
                changer._crossfade_state,
                use_sola=rt_config.use_sola,
            )

            # Add to output buffer
            changer.output_buffer.add(cf_result.audio)
            changer._chunks_ready += 1

            chunk_time = time.perf_counter() - chunk_start
            chunk_times.append(chunk_time)

            if len(chunk_times) <= 3:
                print(f"  Chunk {len(chunk_times)}: infer={chunk_time*1000:.0f}ms, buf={changer.output_buffer.available}")

        # Simulate output callback (output consumed)
        if changer._chunks_ready >= rt_config.prebuffer_chunks:
            while changer.output_buffer.available >= output_block_size:
                out_block = changer.output_buffer.get(output_block_size)
                outputs.append(out_block)

        input_pos += mic_block_size

    # Drain remaining output
    while changer.output_buffer.available > 0:
        remaining = min(output_block_size, changer.output_buffer.available)
        out_block = changer.output_buffer.get(remaining)
        outputs.append(out_block)

    elapsed = time.perf_counter() - start_time

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.2f}s (input duration: {len(voice_data)/sr:.2f}s)")
    print(f"  Chunks processed: {len(chunk_times)}")
    if chunk_times:
        print(f"  Avg inference: {np.mean(chunk_times)*1000:.0f}ms")
        print(f"  Max inference: {np.max(chunk_times)*1000:.0f}ms")
    print(f"  Output samples: {sum(len(o) for o in outputs)}")
    print(f"  Buffer drops: {changer.output_buffer.samples_dropped}")

    # Analyze output quality
    if outputs:
        full_output = np.concatenate(outputs)

        # Save output
        out_path = Path("tests/test_realtime_direct_output.wav")
        max_val = np.abs(full_output).max()
        if max_val > 0:
            out_norm = full_output / max_val * 0.9
        else:
            out_norm = full_output
        wavfile.write(out_path, rt_config.output_sample_rate, (out_norm * 32767).astype(np.int16))
        print(f"  Output saved: {out_path}")

        # Energy analysis
        window = int(rt_config.output_sample_rate * 0.01)
        energies = []
        for i in range(0, len(full_output) - window, window):
            e = np.sqrt(np.mean(full_output[i:i+window]**2))
            energies.append(e)

        energies = np.array(energies)
        voiced = energies[energies > 0.01]

        if len(voiced) > 0:
            cv = np.std(voiced) / np.mean(voiced)
            min_ratio = np.min(voiced) / np.mean(voiced)
            drops = np.sum(voiced < np.mean(voiced) * 0.5)

            print(f"\n  Quality metrics:")
            print(f"    CV: {cv:.3f} (target < 0.3)")
            print(f"    min/mean: {min_ratio:.3f} (target > 0.3)")
            print(f"    drops: {drops} (target < 5)")

            passed = cv < 0.3 and min_ratio > 0.3 and drops < 5
            print(f"\n{'[PASS]' if passed else '[FAIL]'}")
            return passed

    print("\n[FAIL] No output produced")
    return False


if __name__ == "__main__":
    success = test_realtime_direct()
    sys.exit(0 if success else 1)
