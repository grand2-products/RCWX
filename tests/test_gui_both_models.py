"""Test actual GUI code path with both models."""

import sys
import numpy as np
from pathlib import Path
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeConfig, RealtimeVoiceChanger
from rcwx.audio.buffer import ChunkBuffer
from rcwx.audio.crossfade import SOLAState, apply_sola_crossfade
from rcwx.audio.resample import resample


def test_model_gui_path(model_path: str, model_name: str):
    """Test a model using actual GUI code path."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {model_name}")
    print(f"Path: {model_path}")
    print("=" * 70)

    # Load voice
    voice_path = Path("sample_data/kakita.wav")
    if not voice_path.exists():
        print("No test audio")
        return None

    sr, voice_data = wavfile.read(voice_path)
    if voice_data.dtype == np.int16:
        voice_data = voice_data.astype(np.float32) / 32768.0
    if len(voice_data.shape) > 1:
        voice_data = voice_data[:, 0]
    voice_data = voice_data[:int(5 * sr)]

    # Create pipeline
    pipeline = RVCPipeline(model_path, device="xpu", use_compile=False)
    pipeline.load()
    print(f"Model SR: {pipeline.sample_rate}Hz")

    # Create config with low-latency settings
    rt_config = RealtimeConfig(
        mic_sample_rate=48000,
        output_sample_rate=48000,
        chunk_sec=0.35,
        pitch_shift=0,
        use_f0=True,
        f0_method="rmvpe",
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=True,
        context_sec=0.05,
        extra_sec=0.0,
        crossfade_sec=0.05,
        lookahead_sec=0.0,
        use_sola=True,
        prebuffer_chunks=1,
    )

    # Create changer
    changer = RealtimeVoiceChanger(pipeline, config=rt_config)

    # Initialize
    pipeline.clear_cache()
    changer.mic_chunk_samples = int(changer.config.mic_sample_rate * changer.config.chunk_sec)
    changer.mic_context_samples = int(changer.config.mic_sample_rate * changer.config.context_sec)
    changer.mic_lookahead_samples = int(changer.config.mic_sample_rate * changer.config.lookahead_sec)

    changer.input_buffer = ChunkBuffer(
        changer.mic_chunk_samples,
        crossfade_samples=0,
        context_samples=changer.mic_context_samples,
        lookahead_samples=changer.mic_lookahead_samples,
    )

    changer.output_crossfade_samples = int(
        changer.config.output_sample_rate * changer.config.crossfade_sec
    )

    # Use SOLAState like realtime.py
    sola_state = SOLAState.create(
        changer.output_crossfade_samples,
        changer.config.output_sample_rate,
    )

    # Warmup
    print("Warmup...")
    mic_total = (changer.mic_chunk_samples + changer.mic_context_samples +
                 changer.mic_lookahead_samples)
    warmup_samples = int(mic_total * changer.config.input_sample_rate /
                        changer.config.mic_sample_rate)

    warmup_audio = np.zeros(warmup_samples, dtype=np.float32)
    try:
        _ = pipeline.infer(warmup_audio, input_sr=changer.config.input_sample_rate,
                           pitch_shift=0, f0_method="rmvpe", index_rate=0.0,
                           voice_gate_mode="off", use_feature_cache=False)
    except:
        pass

    t = np.arange(warmup_samples) / changer.config.input_sample_rate
    warmup_audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    for _ in range(3):
        try:
            _ = pipeline.infer(warmup_audio, input_sr=changer.config.input_sample_rate,
                               pitch_shift=0, f0_method="rmvpe", index_rate=0.0,
                               voice_gate_mode="off", use_feature_cache=True)
        except:
            pass

    # Reset
    changer.stats.reset()
    changer.input_buffer.clear()
    changer.output_buffer.clear()
    sola_state = SOLAState.create(
        changer.output_crossfade_samples,
        changer.config.output_sample_rate,
    )

    # Process
    print("Processing...")
    outputs = []
    mic_block_size = 4096
    input_pos = 0
    chunk_count = 0

    while input_pos < len(voice_data):
        block = voice_data[input_pos:input_pos + mic_block_size]
        if len(block) < mic_block_size:
            block = np.pad(block, (0, mic_block_size - len(block)))
        changer.input_buffer.add_input(block)
        input_pos += mic_block_size

        while changer.input_buffer.has_chunk():
            chunk = changer.input_buffer.get_chunk()
            if chunk is None:
                break

            # EXACT CODE FROM _inference_thread
            if changer.config.mic_sample_rate != changer.config.input_sample_rate:
                chunk = resample(chunk, changer.config.mic_sample_rate, changer.config.input_sample_rate)

            try:
                output = pipeline.infer(
                    chunk, input_sr=changer.config.input_sample_rate,
                    pitch_shift=changer.config.pitch_shift,
                    f0_method=changer.config.f0_method if changer.config.use_f0 else "none",
                    index_rate=changer.config.index_rate,
                    voice_gate_mode=changer.config.voice_gate_mode,
                    energy_threshold=changer.config.energy_threshold,
                    use_feature_cache=changer.config.use_feature_cache,
                )
            except Exception as e:
                print(f"  Inference error: {e}")
                continue

            if pipeline.sample_rate != changer.config.output_sample_rate:
                output = resample(output, pipeline.sample_rate, changer.config.output_sample_rate)

            max_val = np.max(np.abs(output))
            if max_val > 1.0:
                output = np.tanh(output)

            # EXACT CODE FROM realtime.py _inference_thread
            # RVC-style SOLA crossfade on raw output (before trim)
            cf_result = apply_sola_crossfade(output, sola_state)
            output = cf_result.audio

            # Skip silent outputs
            rms = np.sqrt(np.mean(output**2))
            if rms < 0.001:
                continue

            outputs.append(output)
            chunk_count += 1

    print(f"Chunks: {chunk_count}")

    if len(outputs) < 2:
        print("Not enough output")
        return None

    full_output = np.concatenate(outputs)

    # Analyze
    pos = 0
    jumps = []
    for i, o in enumerate(outputs):
        if i > 0:
            jump = abs(full_output[pos] - full_output[pos-1])
            jumps.append((i, jump))
            if jump > 0.05:
                print(f"  Boundary {i}: pos={pos}, jump={jump:.6f}")
        pos += len(o)

    max_jump = max(j for _, j in jumps) if jumps else 0
    avg_jump = np.mean([j for _, j in jumps]) if jumps else 0

    print(f"\nBoundary jumps: max={max_jump:.6f}, avg={avg_jump:.6f}")
    passed = max_jump < 0.1
    print(f"Result: {'PASS' if passed else 'FAIL'}")

    return passed


def main():
    models = [
        ("C:/lib/github/grand2-products/RCWX/model/kurumi/kurumi.pth", "kurumi"),
        ("C:/lib/github/grand2-products/RCWX/model/kana/kana/voice.pth", "kana"),
    ]

    results = {}
    for model_path, model_name in models:
        if Path(model_path).exists():
            results[model_name] = test_model_gui_path(model_path, model_name)
        else:
            print(f"Model not found: {model_path}")
            results[model_name] = None

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for name, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
