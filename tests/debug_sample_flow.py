"""Debug sample flow through the system."""

import sys
import numpy as np
from pathlib import Path
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeConfig, RealtimeVoiceChanger
from rcwx.audio.crossfade import CrossfadeState, apply_crossfade, trim_edges
from rcwx.audio.resample import resample


def main():
    print("=" * 70)
    print("Sample Flow Debug")
    print("=" * 70)

    # Load config
    config = RCWXConfig.load()
    if not config.last_model_path:
        print("ERROR: No model configured")
        return

    # Create pipeline
    print(f"Loading model: {config.last_model_path}")
    pipeline = RVCPipeline(
        config.last_model_path,
        device=config.device,
        use_compile=False,  # Disable compile for cleaner debugging
    )
    pipeline.load()

    # Create RealtimeConfig with explicit values
    rt_config = RealtimeConfig(
        mic_sample_rate=48000,
        output_sample_rate=48000,
        chunk_sec=0.5,  # 500ms chunks
        pitch_shift=0,
        use_f0=True,
        f0_method="rmvpe",
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=False,
        context_sec=0.1,  # 100ms context
        extra_sec=0.02,   # 20ms extra
        crossfade_sec=0.08,  # 80ms crossfade
        lookahead_sec=0.1,
        use_sola=False,  # Disable SOLA for predictable behavior
        prebuffer_chunks=1,
    )

    print(f"\nConfig:")
    print(f"  chunk_sec={rt_config.chunk_sec}")
    print(f"  context_sec={rt_config.context_sec}")
    print(f"  extra_sec={rt_config.extra_sec}")
    print(f"  crossfade_sec={rt_config.crossfade_sec}")

    # Calculate expected sizes
    mic_chunk = int(rt_config.mic_sample_rate * rt_config.chunk_sec)
    mic_context = int(rt_config.mic_sample_rate * rt_config.context_sec)
    mic_lookahead = int(rt_config.mic_sample_rate * rt_config.lookahead_sec)
    total_input_mic = mic_chunk + mic_context + mic_lookahead

    out_context = int(rt_config.output_sample_rate * rt_config.context_sec)
    out_extra = int(rt_config.output_sample_rate * rt_config.extra_sec)
    out_crossfade = int(rt_config.output_sample_rate * rt_config.crossfade_sec)

    print(f"\nSizes at mic rate (48kHz):")
    print(f"  chunk={mic_chunk}, context={mic_context}, lookahead={mic_lookahead}")
    print(f"  total input per chunk={total_input_mic}")

    print(f"\nSizes at output rate (48kHz):")
    print(f"  context={out_context}, extra={out_extra}, crossfade={out_crossfade}")

    # Create test audio: exactly 3 chunks worth of input
    # We need 3 * mic_chunk + mic_context + mic_lookahead total
    # Because ChunkBuffer keeps context from previous for next chunk
    total_input_samples = 3 * mic_chunk + mic_context + mic_lookahead
    print(f"\nTest input: {total_input_samples} samples = {total_input_samples/48000:.3f}s")

    # Generate sine wave
    t = np.arange(total_input_samples) / rt_config.mic_sample_rate
    test_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Create crossfade state with correct cf_samples
    cf_state = CrossfadeState(cf_samples=out_crossfade)
    print(f"\nCrossfadeState: cf_samples={cf_state.cf_samples}")

    # Process 3 chunks manually
    outputs = []
    chunk_start = 0

    for i in range(3):
        # Extract chunk with context and lookahead
        # Structure: [context | main | lookahead]
        if i == 0:
            # First chunk: no left context yet, use zeros
            chunk = test_audio[:total_input_mic]
            chunk_end = mic_chunk
        else:
            # Subsequent chunks: include left context from previous
            chunk_start = i * mic_chunk
            chunk_end = chunk_start + mic_chunk
            # Grab context from before, main, and lookahead after
            left_start = max(0, chunk_start - mic_context)
            right_end = min(len(test_audio), chunk_end + mic_lookahead)
            chunk = test_audio[left_start:right_end]

            # If we don't have enough left context, pad
            if left_start == 0 and i > 0:
                pad = mic_context - chunk_start
                if pad > 0:
                    chunk = np.concatenate([np.zeros(pad, dtype=np.float32), chunk])

        print(f"\nChunk {i+1}:")
        print(f"  Input: {len(chunk)} samples")

        # Resample to processing rate (16kHz)
        chunk_16k = resample(chunk, rt_config.mic_sample_rate, rt_config.input_sample_rate)
        print(f"  After resample to 16kHz: {len(chunk_16k)} samples")

        # Run inference
        output = pipeline.infer(
            chunk_16k,
            input_sr=rt_config.input_sample_rate,
            pitch_shift=0,
            f0_method="rmvpe",
            index_rate=0.0,
            voice_gate_mode="off",
            use_feature_cache=False,
        )
        print(f"  RVC output ({pipeline.sample_rate}Hz): {len(output)} samples")

        # Resample to output rate
        if pipeline.sample_rate != rt_config.output_sample_rate:
            output = resample(output, pipeline.sample_rate, rt_config.output_sample_rate)
        print(f"  After resample to 48kHz: {len(output)} samples")

        # Trim edges
        trimmed = trim_edges(output, out_context, out_extra)
        print(f"  After trim (ctx={out_context}, extra={out_extra}): {len(trimmed)} samples")

        # Apply crossfade
        cf_result = apply_crossfade(trimmed, cf_state, use_sola=False)
        final = cf_result.audio
        print(f"  After crossfade: {len(final)} samples")
        print(f"    prev_tail now: {len(cf_state.prev_tail) if cf_state.prev_tail is not None else 'None'}")

        outputs.append(final)

    # Analyze total
    total_output = sum(len(o) for o in outputs)
    print(f"\n" + "=" * 70)
    print(f"Total output: {total_output} samples = {total_output/48000:.3f}s")
    print(f"Expected (3 chunks): ~{3 * (mic_chunk * rt_config.output_sample_rate / rt_config.mic_sample_rate - (out_context + out_extra) * 2):.0f} samples")

    # Concatenate and save
    full_output = np.concatenate(outputs)
    out_path = Path("tests/debug_sample_flow_output.wav")
    max_val = np.abs(full_output).max()
    if max_val > 0:
        full_output = full_output / max_val * 0.9
    wavfile.write(out_path, rt_config.output_sample_rate, (full_output * 32767).astype(np.int16))
    print(f"\nOutput saved: {out_path}")

    # Analyze chunk boundaries
    print(f"\nChunk boundary analysis:")
    pos = 0
    for i, o in enumerate(outputs):
        if i > 0:
            # Check transition from previous chunk
            prev_end = pos - 1
            curr_start = pos
            if prev_end >= 0 and curr_start < len(full_output):
                jump = abs(full_output[curr_start] - full_output[prev_end])
                print(f"  Boundary {i}: pos={pos}, jump={jump:.6f}")
        pos += len(o)


if __name__ == "__main__":
    main()
