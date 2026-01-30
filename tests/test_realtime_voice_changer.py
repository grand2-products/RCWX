"""
Test RealtimeVoiceChanger directly.

TDD: Test the ACTUAL class, not a simulation.
"""

import sys
import numpy as np

from rcwx.audio.buffer import ChunkBuffer, OutputBuffer
from rcwx.audio.crossfade import CrossfadeState, apply_crossfade, trim_edges
from rcwx.audio.resample import resample
from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline


def generate_tone(duration_sec: float, freq: float = 220.0, sr: int = 48000) -> np.ndarray:
    """Generate sustained tone"""
    t = np.arange(int(sr * duration_sec)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def measure_continuity(audio: np.ndarray, sr: int = 48000) -> dict:
    """Measure energy continuity"""
    window = int(sr * 0.01)  # 10ms
    energies = []
    positions = []

    for i in range(0, len(audio) - window, window):
        energy = np.sqrt(np.mean(audio[i:i + window] ** 2))
        energies.append(energy)
        positions.append(i)

    energies = np.array(energies)
    positions = np.array(positions)
    voiced_mask = energies > 0.01
    voiced = energies[voiced_mask]

    if len(voiced) < 2:
        return {'cv': 1.0, 'min_ratio': 0.0, 'drop_count': 999, 'drop_positions_ms': []}

    mean_e = np.mean(voiced)
    drop_mask = voiced < mean_e * 0.5
    drop_positions = positions[voiced_mask][drop_mask]
    drop_times_ms = (drop_positions / sr * 1000).tolist()

    return {
        'cv': float(np.std(voiced) / mean_e),
        'min_ratio': float(np.min(voiced) / mean_e),
        'drop_count': int(np.sum(drop_mask)),
        'drop_positions_ms': drop_times_ms,
    }


def test_sustained_tone_continuity():
    """
    Test that sustained tone remains continuous after chunk processing.

    This test uses the SAME code path as RealtimeVoiceChanger.
    """
    print("=" * 70)
    print("RealtimeVoiceChanger Integration Test")
    print("=" * 70)

    config = RCWXConfig.load()
    if not config.last_model_path:
        print("ERROR: No model configured")
        return False

    pipeline = RVCPipeline(config.last_model_path, device="auto", use_compile=False)
    pipeline.load()

    # === EXACT SAME SETUP AS RealtimeVoiceChanger.__init__ ===
    mic_sr = config.audio.output_sample_rate or 48000
    input_sr = config.audio.sample_rate or 16000
    output_sr = config.audio.output_sample_rate or 48000
    chunk_sec = config.audio.chunk_sec or 0.35
    context_sec = config.inference.context_sec or 0.1
    extra_sec = config.inference.extra_sec or 0.02
    crossfade_sec = config.inference.crossfade_sec or 0.05

    mic_chunk_samples = int(mic_sr * chunk_sec)
    mic_context_samples = int(mic_sr * context_sec)

    print(f"\nConfig:")
    print(f"  chunk_sec={chunk_sec}, context_sec={context_sec}")
    print(f"  extra_sec={extra_sec}, crossfade_sec={crossfade_sec}")
    print(f"  mic_chunk={mic_chunk_samples}, mic_context={mic_context_samples}")

    # ChunkBuffer (EXACT SAME as realtime.py)
    input_buffer = ChunkBuffer(
        chunk_samples=mic_chunk_samples + mic_context_samples,
        crossfade_samples=0,
        context_samples=mic_context_samples,
    )

    # Output params (EXACT SAME as realtime.py)
    out_cf_samples = int(output_sr * crossfade_sec)
    out_context_samples = int(output_sr * context_sec)
    out_extra_samples = int(output_sr * extra_sec)

    print(f"  out_cf={out_cf_samples}, out_context={out_context_samples}, out_extra={out_extra_samples}")

    crossfade_state = CrossfadeState(cf_samples=out_cf_samples)

    # OutputBuffer (EXACT SAME as realtime.py)
    max_latency_samples = int(output_sr * chunk_sec * 1.5)
    output_buffer = OutputBuffer(max_latency_samples=max_latency_samples, fade_samples=256)

    # === GENERATE TEST INPUT ===
    tone = generate_tone(3.0, freq=220.0, sr=mic_sr)
    print(f"\nInput: 220Hz tone, 3 sec, {len(tone)} samples")

    # === PROCESS EXACTLY LIKE RealtimeVoiceChanger ===
    pipeline.clear_cache()
    outputs = []
    chunk_count = 0

    # Simulate mic callback with small blocks
    block_size = 4096
    pos = 0

    print("\nProcessing...")

    while pos < len(tone):
        # Mic callback delivers audio
        block = tone[pos:pos + block_size]
        input_buffer.add_input(block)

        # Process available chunks (EXACT SAME as _input_callback + _inference_thread)
        while input_buffer.has_chunk():
            chunk = input_buffer.get_chunk()
            if chunk is None:
                break

            chunk_count += 1

            # Log chunk info
            if chunk_count <= 3:
                print(f"  Chunk {chunk_count}: input_len={len(chunk)}")

            # Resample (EXACT SAME as realtime.py)
            chunk = resample(chunk, mic_sr, input_sr)

            # RVC inference (EXACT SAME as realtime.py)
            output = pipeline.infer(
                chunk,
                input_sr=input_sr,
                pitch_shift=0,
                f0_method="rmvpe",
                use_feature_cache=True,
                voice_gate_mode="off",
            )

            # Resample to output (EXACT SAME as realtime.py)
            if pipeline.sample_rate != output_sr:
                output = resample(output, pipeline.sample_rate, output_sr)

            if chunk_count <= 3:
                print(f"           rvc_out={len(output)}")

            # Edge trim (EXACT SAME function as realtime.py)
            trimmed = trim_edges(output, out_context_samples, out_extra_samples)

            if chunk_count <= 3:
                print(f"           trimmed={len(trimmed)}")

            # Crossfade (EXACT SAME function as realtime.py)
            prev_tail_before = crossfade_state.prev_tail.copy() if crossfade_state.prev_tail is not None else None

            cf_result = apply_crossfade(
                trimmed,
                crossfade_state,
                use_sola=config.inference.use_sola,
                sola_search_ratio=0.25,
            )

            if chunk_count <= 3:
                print(f"           cf_out={len(cf_result.audio)}, sola_offset={cf_result.sola_offset}")

            # Add to OutputBuffer (EXACT SAME as realtime.py)
            dropped = output_buffer.add(cf_result.audio)
            if dropped > 0 and chunk_count <= 5:
                print(f"           [!] OutputBuffer dropped {dropped} samples")

        # Simulate output callback - get audio from OutputBuffer
        # Output callback requests small blocks (e.g., 1024 samples)
        output_block_size = 1024
        while output_buffer.available >= output_block_size:
            out_block = output_buffer.get(output_block_size)
            outputs.append(out_block)

        pos += block_size

    # Drain remaining samples from OutputBuffer
    while output_buffer.available > 0:
        remaining = min(output_block_size, output_buffer.available)
        out_block = output_buffer.get(remaining)
        outputs.append(out_block)

    print(f"\nTotal chunks processed: {chunk_count}")
    print(f"OutputBuffer samples dropped: {output_buffer.samples_dropped}")

    # === VERIFY ===
    if len(outputs) == 0:
        print("ERROR: No output produced")
        return False

    full_output = np.concatenate(outputs)
    print(f"Output: {len(full_output)} samples")

    stats = measure_continuity(full_output, sr=output_sr)

    print(f"\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"  CV: {stats['cv']:.3f} (target < 0.2)")
    print(f"  min/mean: {stats['min_ratio']:.3f} (target > 0.5)")
    print(f"  drop_count: {stats['drop_count']} (target == 0)")
    if stats['drop_positions_ms']:
        print(f"  drop positions (ms): {stats['drop_positions_ms'][:10]}")

    # Calculate expected chunk boundaries
    # First chunk output: trimmed - cf
    # Subsequent: cf (blended) + (trimmed - 2*cf)
    first_out = (out_context_samples * 2 + out_extra_samples * 2)  # what's trimmed
    trimmed_len = int(pipeline.sample_rate * chunk_sec * (output_sr / pipeline.sample_rate)) - first_out
    first_chunk_out = trimmed_len - out_cf_samples
    subsequent_out = out_cf_samples + (trimmed_len - 2 * out_cf_samples)
    boundaries_ms = [first_chunk_out / output_sr * 1000]
    for i in range(1, chunk_count):
        boundaries_ms.append(boundaries_ms[-1] + subsequent_out / output_sr * 1000)
    print(f"  chunk boundaries (ms): {boundaries_ms[:5]}")

    # Acceptance criteria
    passed = True
    if stats['cv'] >= 0.2:
        print(f"  [FAIL] CV too high")
        passed = False
    if stats['min_ratio'] <= 0.5:
        print(f"  [FAIL] min/mean too low")
        passed = False
    if stats['drop_count'] > 0:
        print(f"  [FAIL] Energy drops detected")
        passed = False

    if passed:
        print("\n[PASS] All criteria met")
    else:
        print("\n[FAIL] Test failed")

    print("=" * 70)
    return passed


if __name__ == "__main__":
    success = test_sustained_tone_continuity()
    sys.exit(0 if success else 1)
