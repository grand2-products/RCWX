"""Debug kana model energy discontinuity."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeConfig
from rcwx.audio.crossfade import CrossfadeState, apply_crossfade, trim_edges
from rcwx.audio.resample import resample


def main():
    model_path = "C:/lib/github/grand2-products/RCWX/model/kana/kana/voice.pth"

    print("Loading kana model...")
    pipeline = RVCPipeline(model_path, device="xpu", use_compile=False)
    pipeline.load()
    print(f"Model sample rate: {pipeline.sample_rate}Hz")

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
    mic_chunk = int(rt_config.mic_sample_rate * rt_config.chunk_sec)  # 24000
    mic_context = int(rt_config.mic_sample_rate * rt_config.context_sec)  # 4800
    mic_lookahead = int(rt_config.mic_sample_rate * rt_config.lookahead_sec)  # 4800
    total_input_mic = mic_chunk + mic_context + mic_lookahead  # 33600

    out_context = int(rt_config.output_sample_rate * rt_config.context_sec)  # 4800
    out_extra = int(rt_config.output_sample_rate * rt_config.extra_sec)  # 960
    out_crossfade = int(rt_config.output_sample_rate * rt_config.crossfade_sec)  # 3840

    # For 40kHz model, calculate expected output sizes
    model_context = int(pipeline.sample_rate * rt_config.context_sec)  # 4000
    model_extra = int(pipeline.sample_rate * rt_config.extra_sec)  # 800

    print(f"\nSizes:")
    print(f"  Input (48kHz): total={total_input_mic}")
    print(f"  Processing (16kHz): {int(total_input_mic * 16000 / 48000)}")
    print(f"  Model output ({pipeline.sample_rate}Hz): ?")
    print(f"  Output trim (48kHz): context={out_context}, extra={out_extra}")
    print(f"  Crossfade: {out_crossfade}")

    # Create test audio
    total_input_samples = 3 * mic_chunk + mic_context + mic_lookahead
    t = np.arange(total_input_samples) / rt_config.mic_sample_rate
    test_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Warmup
    print("\nWarming up...")
    warmup_samples = int(total_input_mic * rt_config.input_sample_rate / rt_config.mic_sample_rate)
    warmup_audio = np.zeros(warmup_samples, dtype=np.float32)
    _ = pipeline.infer(warmup_audio, input_sr=rt_config.input_sample_rate, pitch_shift=0,
                       f0_method="rmvpe", index_rate=0.0, voice_gate_mode="off", use_feature_cache=False)

    # Process chunks and trace each step
    cf_state = CrossfadeState(cf_samples=out_crossfade)
    outputs = []

    for i in range(3):
        print(f"\n--- Chunk {i+1} ---")

        if i == 0:
            chunk = test_audio[:total_input_mic]
        else:
            chunk_start = i * mic_chunk
            left_start = max(0, chunk_start - mic_context)
            right_end = min(len(test_audio), chunk_start + mic_chunk + mic_lookahead)
            chunk = test_audio[left_start:right_end]

        print(f"  Input (48kHz): {len(chunk)} samples, RMS={np.sqrt(np.mean(chunk**2)):.4f}")

        # Resample to 16kHz
        chunk_16k = resample(chunk, rt_config.mic_sample_rate, rt_config.input_sample_rate)
        print(f"  After 48k->16k: {len(chunk_16k)} samples, RMS={np.sqrt(np.mean(chunk_16k**2)):.4f}")

        # Inference
        output = pipeline.infer(
            chunk_16k, input_sr=rt_config.input_sample_rate, pitch_shift=0,
            f0_method="rmvpe", index_rate=0.0, voice_gate_mode="off", use_feature_cache=False,
        )
        print(f"  Model output ({pipeline.sample_rate}Hz): {len(output)} samples, RMS={np.sqrt(np.mean(output**2)):.4f}")

        # Resample to 48kHz
        if pipeline.sample_rate != rt_config.output_sample_rate:
            output_48k = resample(output, pipeline.sample_rate, rt_config.output_sample_rate)
            print(f"  After {pipeline.sample_rate}->48k: {len(output_48k)} samples, RMS={np.sqrt(np.mean(output_48k**2)):.4f}")
        else:
            output_48k = output

        # Trim edges
        trimmed = trim_edges(output_48k, out_context, out_extra)
        print(f"  After trim: {len(trimmed)} samples, RMS={np.sqrt(np.mean(trimmed**2)):.4f}")

        # Crossfade
        cf_result = apply_crossfade(trimmed, cf_state, use_sola=False)
        print(f"  After crossfade: {len(cf_result.audio)} samples, RMS={np.sqrt(np.mean(cf_result.audio**2)):.4f}")

        outputs.append(cf_result.audio)

    # Analyze
    print("\n--- Analysis ---")
    for i, o in enumerate(outputs):
        rms = np.sqrt(np.mean(o**2))
        print(f"  Chunk {i+1} output RMS: {rms:.4f}")

    # Check boundary energies
    full_output = np.concatenate(outputs)
    window = int(rt_config.output_sample_rate * 0.005)  # 5ms

    print("\n--- Boundary Energy ---")
    pos = 0
    for i, o in enumerate(outputs):
        if i > 0:
            before = full_output[pos-window:pos]
            after = full_output[pos:pos+window]
            before_rms = np.sqrt(np.mean(before**2))
            after_rms = np.sqrt(np.mean(after**2))
            if before_rms > 1e-6:
                disc = abs(1 - after_rms/before_rms) * 100
                print(f"  Boundary {i}: before={before_rms:.4f}, after={after_rms:.4f}, disc={disc:.1f}%")
        pos += len(o)


if __name__ == "__main__":
    main()
