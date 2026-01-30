"""Exact integration test using RealtimeVoiceChanger's actual code path."""

import sys
import time
import numpy as np
from pathlib import Path
from queue import Queue, Empty
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeVoiceChanger, RealtimeConfig
from rcwx.audio.resample import resample


def test_realtime_exact(model_path: str, model_name: str):
    """
    Test using RealtimeVoiceChanger's exact internal logic.

    Simulates: mic callback -> input buffer -> inference thread -> output
    """
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")

    # Load test audio
    voice_path = Path("sample_data/kakita.wav")
    if not voice_path.exists():
        print("Test audio not found")
        return None

    sr, voice_data = wavfile.read(voice_path)
    if voice_data.dtype == np.int16:
        voice_data = voice_data.astype(np.float32) / 32768.0
    if len(voice_data.shape) > 1:
        voice_data = voice_data[:, 0]

    # Use full audio (or limit to 10 seconds)
    max_duration = 10.0
    if len(voice_data) / sr > max_duration:
        voice_data = voice_data[:int(max_duration * sr)]
    print(f"Input: {len(voice_data)/sr:.2f}s @ {sr}Hz")

    # Create pipeline
    pipeline = RVCPipeline(model_path, device="xpu", use_compile=False)
    pipeline.load()
    print(f"Model SR: {pipeline.sample_rate}Hz")

    # Create RealtimeVoiceChanger with default config
    changer = RealtimeVoiceChanger(pipeline)
    config = changer.config

    print(f"\nConfig:")
    print(f"  chunk_sec: {config.chunk_sec}s")
    print(f"  context_sec: {config.context_sec}s")
    print(f"  lookahead_sec: {config.lookahead_sec}s")
    print(f"  crossfade_sec: {config.crossfade_sec}s")
    print(f"  prebuffer_chunks: {config.prebuffer_chunks}")

    # Manually initialize (same as start() but without audio devices)
    pipeline.clear_cache()
    changer._recalculate_buffers()

    # Reset state
    changer.stats.reset()
    changer.input_buffer.clear()
    changer.output_buffer.clear()
    changer._chunks_ready = 0
    changer._output_started = False
    from rcwx.audio.crossfade import SOLAState
    changer._sola_state = SOLAState.create(
        changer.output_crossfade_samples,
        changer.config.output_sample_rate,
    )

    # Warmup
    print("\nWarmup...")
    mic_total = (changer.mic_chunk_samples + changer.mic_context_samples +
                 changer.mic_lookahead_samples)
    warmup_samples = int(mic_total * config.input_sample_rate / config.mic_sample_rate)

    warmup_audio = np.zeros(warmup_samples, dtype=np.float32)
    try:
        pipeline.infer(warmup_audio, input_sr=config.input_sample_rate,
                      pitch_shift=0, f0_method="rmvpe", index_rate=0.0,
                      voice_gate_mode="off", use_feature_cache=False)
    except:
        pass

    t = np.arange(warmup_samples) / config.input_sample_rate
    warmup_audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    for _ in range(3):
        try:
            pipeline.infer(warmup_audio, input_sr=config.input_sample_rate,
                          pitch_shift=0, f0_method="rmvpe", index_rate=0.0,
                          voice_gate_mode="off", use_feature_cache=True)
        except:
            pass

    # Reset after warmup
    changer.input_buffer.clear()
    changer._sola_state = SOLAState.create(
        changer.output_crossfade_samples,
        changer.config.output_sample_rate,
    )

    # Resample input to mic rate if needed
    if sr != config.mic_sample_rate:
        voice_data = resample(voice_data, sr, config.mic_sample_rate)

    # Simulate mic input blocks
    mic_block_size = 4096  # Typical audio callback size

    # Collect outputs
    outputs = []
    inference_times = []
    chunk_count = 0
    input_pos = 0

    print("\nProcessing...")
    start_time = time.perf_counter()

    while input_pos < len(voice_data):
        # Simulate mic callback: add block to input buffer
        block = voice_data[input_pos:input_pos + mic_block_size]
        if len(block) < mic_block_size:
            block = np.pad(block, (0, mic_block_size - len(block)))

        changer.input_buffer.add_input(block)
        input_pos += mic_block_size

        # Process available chunks (same as _inference_thread)
        while changer.input_buffer.has_chunk():
            chunk = changer.input_buffer.get_chunk()
            if chunk is None:
                break

            infer_start = time.perf_counter()

            # Resample to processing rate
            if config.mic_sample_rate != config.input_sample_rate:
                chunk = resample(chunk, config.mic_sample_rate, config.input_sample_rate)

            # Inference
            try:
                output = pipeline.infer(
                    chunk,
                    input_sr=config.input_sample_rate,
                    pitch_shift=config.pitch_shift,
                    f0_method=config.f0_method if config.use_f0 else "none",
                    index_rate=config.index_rate,
                    voice_gate_mode=config.voice_gate_mode,
                    energy_threshold=config.energy_threshold,
                    use_feature_cache=config.use_feature_cache,
                )
            except Exception as e:
                print(f"  Inference error: {e}")
                continue

            # Resample to output rate
            if pipeline.sample_rate != config.output_sample_rate:
                output = resample(output, pipeline.sample_rate, config.output_sample_rate)

            # Soft clip
            max_val = np.max(np.abs(output))
            if max_val > 1.0:
                output = np.tanh(output)

            # SOLA crossfade (exact same as realtime.py)
            from rcwx.audio.crossfade import apply_sola_crossfade
            cf_result = apply_sola_crossfade(output, changer._sola_state)
            output = cf_result.audio

            infer_time = time.perf_counter() - infer_start
            inference_times.append(infer_time)

            # Skip silent
            rms = np.sqrt(np.mean(output**2))
            if rms < 0.001:
                continue

            outputs.append(output.copy())
            chunk_count += 1

    total_time = time.perf_counter() - start_time
    print(f"Processed {chunk_count} chunks in {total_time:.2f}s")

    if len(outputs) < 2:
        print("Not enough output chunks")
        return None

    # Analysis
    print(f"\n=== Results ===")

    # Inference time stats
    if inference_times:
        avg_infer = np.mean(inference_times) * 1000
        max_infer = np.max(inference_times) * 1000
        print(f"Inference time: avg={avg_infer:.1f}ms, max={max_infer:.1f}ms")

    # Theoretical latency
    input_latency = config.chunk_sec + config.context_sec + config.lookahead_sec
    sola_latency = changer._sola_state.sola_buffer_frame / config.output_sample_rate
    total_latency = input_latency + avg_infer/1000 + sola_latency
    print(f"Theoretical latency: {total_latency*1000:.0f}ms")
    print(f"  - Input buffering: {input_latency*1000:.0f}ms")
    print(f"  - Inference: {avg_infer:.0f}ms")
    print(f"  - SOLA buffer: {sola_latency*1000:.0f}ms")

    # Boundary analysis
    full_output = np.concatenate(outputs)
    print(f"\nOutput: {len(full_output)/config.output_sample_rate:.2f}s")

    pos = 0
    jumps = []
    for i, out in enumerate(outputs):
        if i > 0 and pos > 0:
            prev_last = full_output[pos - 1]
            curr_first = full_output[pos]
            jump = abs(curr_first - prev_last)
            jumps.append({
                "index": i,
                "pos": pos,
                "jump": jump,
                "prev_last": prev_last,
                "curr_first": curr_first,
            })
        pos += len(out)

    max_jump = max(j["jump"] for j in jumps) if jumps else 0
    avg_jump = np.mean([j["jump"] for j in jumps]) if jumps else 0

    print(f"\nBoundary analysis:")
    print(f"  Boundaries: {len(jumps)}")
    print(f"  Max jump: {max_jump:.6f} (target < 0.1)")
    print(f"  Avg jump: {avg_jump:.6f}")

    # Show large jumps
    large_jumps = [j for j in jumps if j["jump"] > 0.05]
    if large_jumps:
        print(f"  Large jumps (>0.05):")
        for j in large_jumps[:5]:
            print(f"    Boundary {j['index']}: jump={j['jump']:.4f}")

    # Save output
    output_path = f"test_realtime_exact_{model_name}.wav"
    wavfile.write(output_path, config.output_sample_rate, full_output)
    print(f"\nSaved: {output_path}")

    # Pass/fail
    passed = max_jump < 0.1
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")

    return {
        "passed": passed,
        "max_jump": max_jump,
        "avg_jump": avg_jump,
        "avg_infer_ms": avg_infer,
        "latency_ms": total_latency * 1000,
    }


def main():
    models = [
        ("C:/lib/github/grand2-products/RCWX/model/kurumi/kurumi.pth", "kurumi"),
        ("C:/lib/github/grand2-products/RCWX/model/kana/kana/voice.pth", "kana"),
    ]

    results = {}
    for model_path, model_name in models:
        if not Path(model_path).exists():
            print(f"Model not found: {model_path}")
            continue
        results[model_name] = test_realtime_exact(model_path, model_name)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    all_pass = True
    for name, result in results.items():
        if result is None:
            print(f"  {name}: SKIPPED")
            all_pass = False
        else:
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  {name}: {status}")
            print(f"    max_jump={result['max_jump']:.4f}")
            print(f"    latency={result['latency_ms']:.0f}ms")
            print(f"    infer={result['avg_infer_ms']:.0f}ms")
            if not result["passed"]:
                all_pass = False

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
