"""
w-okada style continuity test

Tests the shared crossfade module used by realtime.py
"""

import numpy as np
from scipy.io import wavfile

from rcwx.audio.crossfade import CrossfadeState, apply_crossfade, trim_edges
from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline


def generate_sustained_tone(duration_sec: float, freq: float = 220.0, sr: int = 16000) -> np.ndarray:
    """Generate sustained tone"""
    t = np.arange(int(sr * duration_sec)) / sr
    amp = 0.5 + 0.1 * np.sin(2 * np.pi * 5 * t)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def measure_energy_continuity(audio: np.ndarray, window_ms: float = 10, sr: int = 48000) -> dict:
    """Measure energy continuity"""
    window = int(sr * window_ms / 1000)
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
    voiced_positions = positions[voiced_mask]

    if len(voiced) < 2:
        return {'cv': 1.0, 'min_ratio': 0.0, 'drop_count': 999, 'drop_positions': []}

    mean_energy = np.mean(voiced)
    min_energy = np.min(voiced)
    threshold = mean_energy * 0.5
    drop_mask = voiced < threshold
    drop_count = np.sum(drop_mask)
    drop_positions = voiced_positions[drop_mask].tolist()

    return {
        'cv': float(np.std(voiced) / mean_energy),
        'min_ratio': float(min_energy / mean_energy),
        'drop_count': int(drop_count),
        'drop_positions': drop_positions,
        'min_position': int(positions[np.argmin(energies[voiced_mask])]) if len(voiced) > 0 else 0,
    }


def process_with_shared_module(
    audio: np.ndarray,
    pipeline: RVCPipeline,
    chunk_sec: float = 0.35,
    context_sec: float = 0.1,
    extra_sec: float = 0.02,
    crossfade_sec: float = 0.05,
    use_sola: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Process audio using the SAME shared module as realtime.py
    """
    sr_in = 16000
    chunk_samples = int(sr_in * chunk_sec)
    context_samples = int(sr_in * context_sec)

    sr_out = pipeline.sample_rate
    ratio = sr_out / sr_in

    # Output parameters (same as realtime.py)
    cf_samples = int(sr_out * crossfade_sec)
    out_extra_samples = int(sr_out * extra_sec)
    out_context_samples = int(context_samples * ratio)

    pipeline.clear_cache()

    # Use the SAME CrossfadeState as realtime.py
    crossfade_state = CrossfadeState(cf_samples=cf_samples)

    outputs = []
    stats = {
        'chunks_processed': 0,
        'sola_offsets': [],
        'sola_correlations': [],
        'trim_amounts': [],
        'input_ranges': [],
    }

    pos = 0
    chunk_num = 0

    while pos + chunk_samples <= len(audio):
        # w-okada style input: [left_context | main | right_context]
        left_ctx_start = max(0, pos - context_samples)
        right_ctx_end = min(len(audio), pos + chunk_samples + context_samples)

        chunk = audio[left_ctx_start:right_ctx_end]

        # Padding for edges
        left_pad = max(0, context_samples - pos)
        right_pad = max(0, (pos + chunk_samples + context_samples) - len(audio))

        if left_pad > 0:
            chunk = np.pad(chunk, (left_pad, 0), mode='constant')
        if right_pad > 0:
            chunk = np.pad(chunk, (0, right_pad), mode='constant')

        stats['input_ranges'].append({
            'main_start': pos,
            'main_end': pos + chunk_samples,
            'chunk_len': len(chunk),
        })

        if len(chunk) < chunk_samples:
            break

        # RVC inference
        out = pipeline.infer(
            chunk,
            input_sr=sr_in,
            pitch_shift=0,
            f0_method="rmvpe",
            use_feature_cache=True,
            voice_gate_mode="off",
        )

        stats['trim_amounts'].append({
            'out_len': len(out),
            'context': out_context_samples,
            'extra': out_extra_samples,
        })

        # Use SAME trim_edges function as realtime.py
        trimmed = trim_edges(out, out_context_samples, out_extra_samples)

        # Use SAME apply_crossfade function as realtime.py
        cf_result = apply_crossfade(
            trimmed,
            crossfade_state,
            use_sola=use_sola,
            sola_search_ratio=0.25,
        )

        outputs.append(cf_result.audio)
        stats['sola_offsets'].append(cf_result.sola_offset)
        stats['sola_correlations'].append(cf_result.sola_correlation)

        pos += chunk_samples
        chunk_num += 1
        stats['chunks_processed'] += 1

    if not outputs:
        return np.array([]), stats

    return np.concatenate(outputs), stats


def test_wokada_continuity():
    """w-okada style continuity test using shared module"""
    print("=" * 70)
    print("w-okada style continuity test (shared module)")
    print("=" * 70)

    config = RCWXConfig.load()
    model_path = config.last_model_path
    if not model_path:
        print("ERROR: No model configured")
        return False

    pipeline = RVCPipeline(model_path, device="auto", use_compile=False)
    pipeline.load()
    sr_out = pipeline.sample_rate

    tone = generate_sustained_tone(3.0, freq=220.0, sr=16000)
    print(f"\nInput: 220Hz sustained tone, 3 sec")

    # Reference: single pass
    print("\n" + "-" * 70)
    print("[Reference] Single pass")
    print("-" * 70)

    pipeline.clear_cache()
    out_single = pipeline.infer(
        tone, input_sr=16000, pitch_shift=0, f0_method="rmvpe",
        use_feature_cache=False, voice_gate_mode="off"
    )
    stats_single = measure_energy_continuity(out_single, sr=sr_out)
    print(f"  CV: {stats_single['cv']:.3f}")
    print(f"  min/mean: {stats_single['min_ratio']:.3f}")
    print(f"  drop_count: {stats_single['drop_count']}")

    # Test: w-okada style with shared module
    print("\n" + "-" * 70)
    print("[Test] w-okada style (shared module, context=100ms, extra=20ms)")
    print("-" * 70)

    out_wokada, proc_stats = process_with_shared_module(
        tone, pipeline,
        chunk_sec=0.35,
        context_sec=0.1,
        extra_sec=0.02,
        crossfade_sec=0.05,
        use_sola=True,
    )

    print(f"  Chunks processed: {proc_stats['chunks_processed']}")

    print(f"\n  Input ranges:")
    for i, r in enumerate(proc_stats['input_ranges'][:3]):
        print(f"    Chunk {i}: main=[{r['main_start']} - {r['main_end']}], len={r['chunk_len']}")

    print(f"\n  Trim details:")
    for i, trim in enumerate(proc_stats['trim_amounts'][:3]):
        trimmed_len = trim['out_len'] - 2 * (trim['context'] + trim['extra'])
        print(f"    Chunk {i}: out={trim['out_len']}, context={trim['context']}, extra={trim['extra']} -> {trimmed_len}")

    valid_corrs = [c for c in proc_stats['sola_correlations'] if c > 0]
    if valid_corrs:
        print(f"\n  SOLA correlation: mean={np.mean(valid_corrs):.3f}")
        print(f"  SOLA offsets: {proc_stats['sola_offsets'][:5]}...")

    stats_wokada = measure_energy_continuity(out_wokada, sr=sr_out)
    print(f"\n  CV: {stats_wokada['cv']:.3f}")
    print(f"  min/mean: {stats_wokada['min_ratio']:.3f}")
    print(f"  drop_count: {stats_wokada['drop_count']}")
    if stats_wokada['drop_positions']:
        drop_times = [p / sr_out * 1000 for p in stats_wokada['drop_positions'][:5]]
        print(f"  drop positions (ms): {drop_times}")
    print(f"  min position: {stats_wokada['min_position'] / sr_out * 1000:.1f}ms")

    # Save output
    if len(out_wokada) > 0:
        max_val = np.abs(out_wokada).max()
        if max_val > 0:
            out_norm = out_wokada / max_val * 0.9
        else:
            out_norm = out_wokada
        wavfile.write("test_wokada_continuity.wav", sr_out, (out_norm * 32767).astype(np.int16))
        print("  -> test_wokada_continuity.wav saved")

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

    wokada_ok = (
        stats_wokada['cv'] < EXPECTED_CV and
        stats_wokada['min_ratio'] > EXPECTED_MIN_RATIO and
        stats_wokada['drop_count'] == EXPECTED_DROP_COUNT
    )
    print(f"w-okada:     CV={stats_wokada['cv']:.3f}, min/mean={stats_wokada['min_ratio']:.3f}, "
          f"drops={stats_wokada['drop_count']} -> {'PASS' if wokada_ok else 'FAIL'}")

    if not wokada_ok:
        print("\n*** Issues remain ***")
        if stats_wokada['cv'] >= EXPECTED_CV:
            print(f"  - CV too high: {stats_wokada['cv']:.3f} >= {EXPECTED_CV}")
        if stats_wokada['min_ratio'] <= EXPECTED_MIN_RATIO:
            print(f"  - min/mean too low: {stats_wokada['min_ratio']:.3f} <= {EXPECTED_MIN_RATIO}")
        if stats_wokada['drop_count'] > EXPECTED_DROP_COUNT:
            print(f"  - Energy drops: {stats_wokada['drop_count']} > {EXPECTED_DROP_COUNT}")

    print("=" * 70)

    return wokada_ok


if __name__ == "__main__":
    success = test_wokada_continuity()
    exit(0 if success else 1)
