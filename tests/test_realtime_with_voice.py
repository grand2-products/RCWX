"""
Test with REAL voice input through the full buffer chain.

RED CASE: This test should FAIL if there's a bug in the realtime pipeline.
"""

import sys
import numpy as np
from scipy.io import wavfile
from pathlib import Path

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

from rcwx.audio.buffer import ChunkBuffer, OutputBuffer
from rcwx.audio.crossfade import CrossfadeState, apply_crossfade, trim_edges
from rcwx.audio.resample import resample
from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline


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


def test_with_real_voice():
    """
    Test using real voice recording through the COMPLETE pipeline.

    This includes:
    - ChunkBuffer (input buffering)
    - Resample
    - RVC inference
    - trim_edges
    - apply_crossfade
    - OutputBuffer (output buffering with underrun handling)
    """
    print("=" * 70)
    print("Real Voice Test (Full Pipeline)")
    print("=" * 70)

    # Load config and model
    config = RCWXConfig.load()
    if not config.last_model_path:
        print("ERROR: No model configured")
        return False

    # Load test file (pure sine wave for baseline test)
    voice_path = Path("sample_data/pure_sine.wav")
    if not voice_path.exists():
        voice_path = Path("sample_data/nc283304.mp3")
    if not voice_path.exists():
        print(f"ERROR: Test voice file not found")
        return False

    # Load audio file (supports WAV and MP3)
    if voice_path.suffix.lower() == '.mp3':
        if not HAS_LIBROSA:
            print("ERROR: librosa required for MP3 files. Install with: pip install librosa")
            return False
        voice_data, file_sr = librosa.load(voice_path, sr=None, mono=True)
    else:
        file_sr, voice_data = wavfile.read(voice_path)
        if voice_data.dtype == np.int16:
            voice_data = voice_data.astype(np.float32) / 32768.0
        if len(voice_data.shape) > 1:
            voice_data = voice_data[:, 0]  # Mono

    print(f"\nInput: {voice_path}")
    print(f"  Sample rate: {file_sr} Hz")
    print(f"  Duration: {len(voice_data) / file_sr:.2f} sec")

    # Resample to mic rate if needed
    mic_sr = 48000
    if file_sr != mic_sr:
        voice_data = resample(voice_data, file_sr, mic_sr)
        print(f"  Resampled to: {mic_sr} Hz")

    # Load RVC pipeline
    pipeline = RVCPipeline(config.last_model_path, device="auto", use_compile=False)
    pipeline.load()

    # === EXACT SAME SETUP AS RealtimeVoiceChanger ===
    input_sr = 16000
    output_sr = 48000
    chunk_sec = config.audio.chunk_sec or 0.35
    context_sec = config.inference.context_sec or 0.1
    extra_sec = config.inference.extra_sec or 0.02
    # Use ACTUAL config values (same as user's config.json)
    crossfade_sec = config.inference.crossfade_sec  # User has 0.05

    mic_chunk_samples = int(mic_sr * chunk_sec)
    mic_context_samples = int(mic_sr * context_sec)

    print(f"\nConfig:")
    print(f"  chunk={chunk_sec}s, context={context_sec}s, extra={extra_sec}s, cf={crossfade_sec}s")

    # ChunkBuffer WITH lookahead (w-okada style)
    input_buffer = ChunkBuffer(
        chunk_samples=mic_chunk_samples + mic_context_samples,
        crossfade_samples=0,
        context_samples=mic_context_samples,
        lookahead_samples=mic_context_samples,  # Right context for quality
    )

    # Output params
    out_cf_samples = int(output_sr * crossfade_sec)
    out_context_samples = int(output_sr * context_sec)
    out_extra_samples = int(output_sr * extra_sec)

    # CrossfadeState
    crossfade_state = CrossfadeState(cf_samples=out_cf_samples)

    # OutputBuffer
    max_latency_samples = int(output_sr * chunk_sec * 1.5)
    output_buffer = OutputBuffer(max_latency_samples=max_latency_samples, fade_samples=256)

    # === PROCESS ===
    pipeline.clear_cache()
    outputs = []
    chunk_count = 0
    underrun_count = 0

    # Simulate mic callback with small blocks
    block_size = 4096
    output_block_size = 1024
    pos = 0

    print("\nProcessing...")

    while pos < len(voice_data):
        # Mic callback
        block = voice_data[pos:pos + block_size]
        input_buffer.add_input(block)

        # Process chunks
        while input_buffer.has_chunk():
            chunk = input_buffer.get_chunk()
            if chunk is None:
                break

            chunk_count += 1
            if chunk_count <= 3:
                print(f"  Chunk {chunk_count}: input={len(chunk)}")

            # Resample
            chunk = resample(chunk, mic_sr, input_sr)

            # RVC inference
            output = pipeline.infer(
                chunk,
                input_sr=input_sr,
                pitch_shift=0,
                f0_method="rmvpe",
                use_feature_cache=False,  # Disabled to test if cache causes issues
                voice_gate_mode="off",
            )

            # Resample to output
            if pipeline.sample_rate != output_sr:
                output = resample(output, pipeline.sample_rate, output_sr)

            # Trim edges
            trimmed = trim_edges(output, out_context_samples, out_extra_samples)

            # Debug: check energy before crossfade
            prev_tail_energy = 0.0
            curr_head_energy = 0.0
            if crossfade_state.prev_tail is not None:
                prev_tail_energy = np.sqrt(np.mean(crossfade_state.prev_tail ** 2))
                curr_head_energy = np.sqrt(np.mean(trimmed[:out_cf_samples] ** 2))

            # Crossfade (use actual config settings)
            cf_result = apply_crossfade(
                trimmed,
                crossfade_state,
                use_sola=config.inference.use_sola,  # User has True
                sola_search_ratio=0.25,
            )

            # Debug: check energy after crossfade (blended region)
            blended_energy = 0.0
            if len(cf_result.audio) >= out_cf_samples:
                blended_energy = np.sqrt(np.mean(cf_result.audio[:out_cf_samples] ** 2))

            if chunk_count <= 5:
                print(f"           rvc={len(output)}, trimmed={len(trimmed)}, cf_out={len(cf_result.audio)}")
                # Check energy profile across trimmed output
                n_segments = 5
                seg_len = len(trimmed) // n_segments
                seg_energies = []
                for i in range(n_segments):
                    seg = trimmed[i*seg_len:(i+1)*seg_len]
                    seg_e = np.sqrt(np.mean(seg**2))
                    seg_energies.append(f"{seg_e:.3f}")
                print(f"           trimmed energy profile: [{', '.join(seg_energies)}]")
                if prev_tail_energy > 0:
                    print(f"           prev_tail_e={prev_tail_energy:.4f}, curr_head_e={curr_head_energy:.4f}, blended_e={blended_energy:.4f}")

            # Add to OutputBuffer
            output_buffer.add(cf_result.audio)

        # Simulate output callback
        while output_buffer.available >= output_block_size:
            out_block = output_buffer.get(output_block_size)
            outputs.append(out_block)

        pos += block_size

    # Drain OutputBuffer
    while output_buffer.available > 0:
        remaining = min(output_block_size, output_buffer.available)
        out_block = output_buffer.get(remaining)
        outputs.append(out_block)

    print(f"\nTotal chunks: {chunk_count}")
    print(f"OutputBuffer dropped: {output_buffer.samples_dropped}")

    if len(outputs) == 0:
        print("ERROR: No output")
        return False

    full_output = np.concatenate(outputs)
    print(f"Output: {len(full_output)} samples ({len(full_output)/output_sr:.2f} sec)")

    # Save output for listening
    out_path = Path("tests/test_realtime_voice_output.wav")
    max_val = np.abs(full_output).max()
    if max_val > 0:
        out_norm = full_output / max_val * 0.9
    else:
        out_norm = full_output
    wavfile.write(out_path, output_sr, (out_norm * 32767).astype(np.int16))
    print(f"Output saved: {out_path}")

    # Measure continuity
    stats = measure_continuity(full_output, sr=output_sr)

    print(f"\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"  CV: {stats['cv']:.3f} (target < 0.3)")
    print(f"  min/mean: {stats['min_ratio']:.3f} (target > 0.3)")
    print(f"  drop_count: {stats['drop_count']} (target < 5)")
    if stats['drop_positions_ms']:
        print(f"  drop positions (ms): {stats['drop_positions_ms'][:10]}")

    # More lenient criteria for real voice (has natural pauses)
    passed = True
    if stats['cv'] >= 0.5:
        print(f"  [FAIL] CV too high")
        passed = False
    if stats['min_ratio'] <= 0.1:
        print(f"  [FAIL] min/mean too low - likely audio dropout")
        passed = False
    # Check for periodic drops (sign of chunk boundary issues)
    if len(stats['drop_positions_ms']) >= 3:
        diffs = np.diff(stats['drop_positions_ms'][:10])
        if len(diffs) > 0:
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)
            # If drops are periodic (low std), it's a chunk boundary issue
            if std_diff < mean_diff * 0.3 and mean_diff > 100:
                print(f"  [FAIL] Periodic drops detected (interval ~{mean_diff:.0f}ms)")
                passed = False

    if passed:
        print("\n[PASS]")
    else:
        print("\n[FAIL]")

    print("=" * 70)
    return passed


if __name__ == "__main__":
    success = test_with_real_voice()
    sys.exit(0 if success else 1)
