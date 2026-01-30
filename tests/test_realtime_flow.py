"""
Realtime flow test

Tests the EXACT same flow as realtime.py:
1. ChunkBuffer for input buffering
2. trim_edges for edge removal
3. apply_crossfade for smooth transitions
"""

import numpy as np
from scipy.io import wavfile

from rcwx.audio.buffer import ChunkBuffer
from rcwx.audio.crossfade import CrossfadeState, apply_crossfade, trim_edges
from rcwx.audio.resample import resample
from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline


def generate_sustained_tone(duration_sec: float, freq: float = 220.0, sr: int = 48000) -> np.ndarray:
    """Generate sustained tone at mic sample rate"""
    t = np.arange(int(sr * duration_sec)) / sr
    amp = 0.5 + 0.1 * np.sin(2 * np.pi * 5 * t)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def measure_energy_continuity(audio: np.ndarray, window_ms: float = 10, sr: int = 48000) -> dict:
    """Measure energy continuity"""
    window = int(sr * window_ms / 1000)
    energies = []

    for i in range(0, len(audio) - window, window):
        energy = np.sqrt(np.mean(audio[i:i + window] ** 2))
        energies.append(energy)

    energies = np.array(energies)
    voiced = energies[energies > 0.01]

    if len(voiced) < 2:
        return {'cv': 1.0, 'min_ratio': 0.0, 'drop_count': 999}

    mean_energy = np.mean(voiced)
    min_energy = np.min(voiced)
    threshold = mean_energy * 0.5
    drop_count = np.sum(voiced < threshold)

    return {
        'cv': float(np.std(voiced) / mean_energy),
        'min_ratio': float(min_energy / mean_energy),
        'drop_count': int(drop_count),
    }


def simulate_realtime_flow(
    audio_at_mic_rate: np.ndarray,
    pipeline: RVCPipeline,
    mic_sample_rate: int = 48000,
    input_sample_rate: int = 16000,
    output_sample_rate: int = 48000,
    chunk_sec: float = 0.35,
    context_sec: float = 0.1,
    extra_sec: float = 0.02,
    crossfade_sec: float = 0.05,
    use_sola: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Simulate EXACT realtime.py flow:

    1. Audio arrives in small blocks (simulating mic callback)
    2. ChunkBuffer accumulates and returns [left_ctx | main | right_ctx]
    3. Resample mic_rate -> input_rate
    4. RVC inference
    5. Resample model_rate -> output_rate
    6. trim_edges removes context+extra
    7. apply_crossfade blends chunks
    """
    # === SAME PARAMETERS AS realtime.py ===
    mic_chunk_samples = int(mic_sample_rate * chunk_sec)
    mic_context_samples = int(mic_sample_rate * context_sec)

    # ChunkBuffer setup (SAME as realtime.py __init__)
    input_buffer = ChunkBuffer(
        chunk_samples=mic_chunk_samples + mic_context_samples,  # main + right_context
        crossfade_samples=0,  # Crossfade handled separately
        context_samples=mic_context_samples,  # Keep as left context for next
    )

    # Output parameters
    out_crossfade_samples = int(output_sample_rate * crossfade_sec)
    out_extra_samples = int(output_sample_rate * extra_sec)
    out_context_samples = int(output_sample_rate * context_sec)

    # Crossfade state (SAME as realtime.py)
    crossfade_state = CrossfadeState(cf_samples=out_crossfade_samples)

    pipeline.clear_cache()

    outputs = []
    stats = {
        'chunks_processed': 0,
        'mic_blocks_received': 0,
        'sola_offsets': [],
        'sola_correlations': [],
    }

    # === SIMULATE MIC CALLBACK ===
    # Audio arrives in small blocks (e.g., 4096 samples at a time)
    mic_block_size = 4096
    pos = 0

    while pos < len(audio_at_mic_rate):
        # Simulate mic callback delivering audio
        block_end = min(pos + mic_block_size, len(audio_at_mic_rate))
        mic_block = audio_at_mic_rate[pos:block_end]
        stats['mic_blocks_received'] += 1

        # Add to input buffer (SAME as realtime.py _input_callback)
        input_buffer.add_input(mic_block)

        # Process all available chunks
        while input_buffer.has_chunk():
            # Get chunk from buffer (SAME as realtime.py _inference_thread)
            chunk_at_mic_rate = input_buffer.get_chunk()
            if chunk_at_mic_rate is None:
                break

            # Resample mic -> processing rate
            chunk = resample(chunk_at_mic_rate, mic_sample_rate, input_sample_rate)

            # RVC inference
            output = pipeline.infer(
                chunk,
                input_sr=input_sample_rate,
                pitch_shift=0,
                f0_method="rmvpe",
                use_feature_cache=True,
                voice_gate_mode="off",
            )

            # Resample model -> output rate
            if pipeline.sample_rate != output_sample_rate:
                output = resample(output, pipeline.sample_rate, output_sample_rate)

            # Edge trim (SAME function as realtime.py)
            trimmed = trim_edges(output, out_context_samples, out_extra_samples)

            # Crossfade (SAME function as realtime.py)
            cf_result = apply_crossfade(
                trimmed,
                crossfade_state,
                use_sola=use_sola,
                sola_search_ratio=0.25,
            )

            outputs.append(cf_result.audio)
            stats['sola_offsets'].append(cf_result.sola_offset)
            stats['sola_correlations'].append(cf_result.sola_correlation)
            stats['chunks_processed'] += 1

        pos = block_end

    if not outputs:
        return np.array([]), stats

    return np.concatenate(outputs), stats


def test_realtime_flow():
    """Test using exact realtime.py flow"""
    print("=" * 70)
    print("Realtime Flow Test (using ChunkBuffer + shared modules)")
    print("=" * 70)

    config = RCWXConfig.load()
    model_path = config.last_model_path
    if not model_path:
        print("ERROR: No model configured")
        return False

    pipeline = RVCPipeline(model_path, device="auto", use_compile=False)
    pipeline.load()

    # Generate test audio at mic rate (48kHz)
    mic_sr = 48000
    tone = generate_sustained_tone(3.0, freq=220.0, sr=mic_sr)
    print(f"\nInput: 220Hz sustained tone, 3 sec @ {mic_sr}Hz")

    # Reference: single pass (resample -> infer -> resample)
    print("\n" + "-" * 70)
    print("[Reference] Single pass")
    print("-" * 70)

    pipeline.clear_cache()
    tone_16k = resample(tone, mic_sr, 16000)
    out_single = pipeline.infer(
        tone_16k, input_sr=16000, pitch_shift=0, f0_method="rmvpe",
        use_feature_cache=False, voice_gate_mode="off"
    )
    out_single = resample(out_single, pipeline.sample_rate, mic_sr)

    stats_single = measure_energy_continuity(out_single, sr=mic_sr)
    print(f"  CV: {stats_single['cv']:.3f}")
    print(f"  min/mean: {stats_single['min_ratio']:.3f}")
    print(f"  drop_count: {stats_single['drop_count']}")

    # Test: realtime flow simulation
    print("\n" + "-" * 70)
    print("[Test] Realtime flow (ChunkBuffer + trim + crossfade)")
    print("-" * 70)

    out_realtime, proc_stats = simulate_realtime_flow(
        tone, pipeline,
        mic_sample_rate=48000,
        input_sample_rate=16000,
        output_sample_rate=48000,
        chunk_sec=0.35,
        context_sec=0.1,
        extra_sec=0.02,
        crossfade_sec=0.05,
        use_sola=True,
    )

    print(f"  Mic blocks received: {proc_stats['mic_blocks_received']}")
    print(f"  Chunks processed: {proc_stats['chunks_processed']}")

    valid_corrs = [c for c in proc_stats['sola_correlations'] if c > 0]
    if valid_corrs:
        print(f"  SOLA correlation: mean={np.mean(valid_corrs):.3f}")
        print(f"  SOLA offsets: {proc_stats['sola_offsets'][:5]}...")

    stats_realtime = measure_energy_continuity(out_realtime, sr=mic_sr)
    print(f"\n  CV: {stats_realtime['cv']:.3f}")
    print(f"  min/mean: {stats_realtime['min_ratio']:.3f}")
    print(f"  drop_count: {stats_realtime['drop_count']}")

    # Save output
    if len(out_realtime) > 0:
        max_val = np.abs(out_realtime).max()
        if max_val > 0:
            out_norm = out_realtime / max_val * 0.9
        else:
            out_norm = out_realtime
        wavfile.write("test_realtime_flow.wav", mic_sr, (out_norm * 32767).astype(np.int16))
        print("  -> test_realtime_flow.wav saved")

    # Judgment
    print("\n" + "=" * 70)
    print("Result")
    print("=" * 70)

    EXPECTED_CV = 0.2
    EXPECTED_MIN_RATIO = 0.5
    EXPECTED_DROP_COUNT = 0

    print(f"\nTarget: CV < {EXPECTED_CV}, min/mean > {EXPECTED_MIN_RATIO}, drop_count == {EXPECTED_DROP_COUNT}")
    print()

    single_ok = stats_single['cv'] < EXPECTED_CV and stats_single['min_ratio'] > EXPECTED_MIN_RATIO
    print(f"Single pass: CV={stats_single['cv']:.3f}, min/mean={stats_single['min_ratio']:.3f} "
          f"-> {'PASS' if single_ok else 'FAIL'} (reference)")

    realtime_ok = (
        stats_realtime['cv'] < EXPECTED_CV and
        stats_realtime['min_ratio'] > EXPECTED_MIN_RATIO and
        stats_realtime['drop_count'] == EXPECTED_DROP_COUNT
    )
    print(f"Realtime:    CV={stats_realtime['cv']:.3f}, min/mean={stats_realtime['min_ratio']:.3f}, "
          f"drops={stats_realtime['drop_count']} -> {'PASS' if realtime_ok else 'FAIL'}")

    if not realtime_ok:
        print("\n*** Issues remain ***")
        if stats_realtime['cv'] >= EXPECTED_CV:
            print(f"  - CV too high: {stats_realtime['cv']:.3f} >= {EXPECTED_CV}")
        if stats_realtime['min_ratio'] <= EXPECTED_MIN_RATIO:
            print(f"  - min/mean too low: {stats_realtime['min_ratio']:.3f} <= {EXPECTED_MIN_RATIO}")
        if stats_realtime['drop_count'] > EXPECTED_DROP_COUNT:
            print(f"  - Energy drops: {stats_realtime['drop_count']} > {EXPECTED_DROP_COUNT}")

    print("=" * 70)

    return realtime_ok


if __name__ == "__main__":
    success = test_realtime_flow()
    exit(0 if success else 1)
