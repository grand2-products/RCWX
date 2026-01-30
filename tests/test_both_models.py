"""Test crossfade fix with both models."""

import sys
import numpy as np
from pathlib import Path
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeConfig
from rcwx.audio.crossfade import CrossfadeState, apply_crossfade, trim_edges
from rcwx.audio.resample import resample


def test_model(model_path: str, model_name: str):
    """Test a single model."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {model_name}")
    print(f"Path: {model_path}")
    print("=" * 70)

    # Create pipeline
    pipeline = RVCPipeline(
        model_path,
        device="xpu",
        use_compile=False,
    )
    pipeline.load()
    print(f"Model sample rate: {pipeline.sample_rate}Hz")

    # Config
    rt_config = RealtimeConfig(
        mic_sample_rate=48000,
        output_sample_rate=48000,
        chunk_sec=0.5,
        context_sec=0.1,
        extra_sec=0.02,
        crossfade_sec=0.08,
        lookahead_sec=0.1,
        use_sola=False,
    )

    # Calculate sizes
    mic_chunk = int(rt_config.mic_sample_rate * rt_config.chunk_sec)
    mic_context = int(rt_config.mic_sample_rate * rt_config.context_sec)
    mic_lookahead = int(rt_config.mic_sample_rate * rt_config.lookahead_sec)
    total_input_mic = mic_chunk + mic_context + mic_lookahead

    out_context = int(rt_config.output_sample_rate * rt_config.context_sec)
    out_extra = int(rt_config.output_sample_rate * rt_config.extra_sec)
    out_crossfade = int(rt_config.output_sample_rate * rt_config.crossfade_sec)

    # Create test audio: 3 chunks
    total_input_samples = 3 * mic_chunk + mic_context + mic_lookahead
    t = np.arange(total_input_samples) / rt_config.mic_sample_rate
    test_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Warmup
    print("Warming up...")
    warmup_samples = int(total_input_mic * rt_config.input_sample_rate / rt_config.mic_sample_rate)
    warmup_audio = np.zeros(warmup_samples, dtype=np.float32)
    _ = pipeline.infer(warmup_audio, input_sr=rt_config.input_sample_rate, pitch_shift=0,
                       f0_method="rmvpe", index_rate=0.0, voice_gate_mode="off", use_feature_cache=False)

    # Process 3 chunks
    cf_state = CrossfadeState(cf_samples=out_crossfade)
    outputs = []
    boundary_jumps = []

    for i in range(3):
        if i == 0:
            chunk = test_audio[:total_input_mic]
        else:
            chunk_start = i * mic_chunk
            left_start = max(0, chunk_start - mic_context)
            right_end = min(len(test_audio), chunk_start + mic_chunk + mic_lookahead)
            chunk = test_audio[left_start:right_end]
            if left_start == 0 and i > 0:
                pad = mic_context - chunk_start
                if pad > 0:
                    chunk = np.concatenate([np.zeros(pad, dtype=np.float32), chunk])

        # Resample to 16kHz
        chunk_16k = resample(chunk, rt_config.mic_sample_rate, rt_config.input_sample_rate)

        # Inference
        output = pipeline.infer(
            chunk_16k, input_sr=rt_config.input_sample_rate, pitch_shift=0,
            f0_method="rmvpe", index_rate=0.0, voice_gate_mode="off", use_feature_cache=False,
        )

        # Resample to output rate
        if pipeline.sample_rate != rt_config.output_sample_rate:
            output = resample(output, pipeline.sample_rate, rt_config.output_sample_rate)

        # Trim and crossfade
        trimmed = trim_edges(output, out_context, out_extra)
        cf_result = apply_crossfade(trimmed, cf_state, use_sola=False)
        outputs.append(cf_result.audio)

    # Analyze boundaries
    full_output = np.concatenate(outputs)
    pos = 0
    for i, o in enumerate(outputs):
        if i > 0:
            prev_end = pos - 1
            curr_start = pos
            if prev_end >= 0 and curr_start < len(full_output):
                jump = abs(full_output[curr_start] - full_output[prev_end])
                boundary_jumps.append(jump)
        pos += len(o)

    # Calculate energy at boundaries
    chunk_samples = len(outputs[0])
    energy_discs = []
    window = int(rt_config.output_sample_rate * 0.005)
    for i in range(1, len(outputs)):
        pos = sum(len(outputs[j]) for j in range(i))
        if pos + window <= len(full_output) and pos >= window:
            before_rms = np.sqrt(np.mean(full_output[pos-window:pos]**2))
            after_rms = np.sqrt(np.mean(full_output[pos:pos+window]**2))
            if before_rms > 1e-6:
                disc = abs(1 - after_rms/before_rms) * 100
                energy_discs.append(disc)

    # Results
    print(f"\nResults:")
    print(f"  Boundary jumps: {[f'{j:.4f}' for j in boundary_jumps]}")
    print(f"  Energy discontinuities: {[f'{d:.1f}%' for d in energy_discs]}")

    max_jump = max(boundary_jumps) if boundary_jumps else 0
    max_disc = max(energy_discs) if energy_discs else 0

    passed = max_jump < 0.1 and max_disc < 10
    print(f"\n  Max jump: {max_jump:.4f} (target < 0.1)")
    print(f"  Max energy disc: {max_disc:.1f}% (target < 10%)")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def main():
    models = [
        ("C:/lib/github/grand2-products/RCWX/model/kurumi/kurumi.pth", "kurumi"),
        ("C:/lib/github/grand2-products/RCWX/model/kana/kana/voice.pth", "kana"),
    ]

    results = {}
    for model_path, model_name in models:
        if Path(model_path).exists():
            results[model_name] = test_model(model_path, model_name)
        else:
            print(f"\nModel not found: {model_path}")
            results[model_name] = None

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    for name, result in results.items():
        if result is None:
            status = "NOT FOUND"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r for r in results.values() if r is not None)
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
