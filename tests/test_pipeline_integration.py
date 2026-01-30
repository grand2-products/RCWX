"""Integration test: Mock mic input -> Full pipeline -> Output analysis."""

import numpy as np
import sys
from pathlib import Path
from scipy.io import wavfile
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeConfig, RealtimeVoiceChanger
from rcwx.audio.buffer import ChunkBuffer
from rcwx.audio.crossfade import SOLAState, apply_sola_crossfade
from rcwx.audio.resample import resample


class MockMicInput:
    """Mock microphone input that feeds audio in small blocks."""

    def __init__(self, audio: np.ndarray, sample_rate: int, block_size: int = 4096):
        self.audio = audio.astype(np.float32)
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.position = 0

    def read_block(self) -> Optional[np.ndarray]:
        """Read next block of audio (simulating mic callback)."""
        if self.position >= len(self.audio):
            return None

        end = min(self.position + self.block_size, len(self.audio))
        block = self.audio[self.position:end]

        # Pad if needed
        if len(block) < self.block_size:
            block = np.pad(block, (0, self.block_size - len(block)))

        self.position = end
        return block

    def reset(self):
        self.position = 0


class OutputCollector:
    """Collect and analyze output chunks."""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.chunks = []
        self.chunk_boundaries = []

    def add_chunk(self, audio: np.ndarray):
        if len(audio) > 0:
            if self.chunks:
                # Record boundary position
                total_so_far = sum(len(c) for c in self.chunks)
                self.chunk_boundaries.append(total_so_far)
            self.chunks.append(audio.copy())

    def get_full_output(self) -> np.ndarray:
        if not self.chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.chunks)

    def analyze(self) -> dict:
        """Analyze output for discontinuities and volume drops."""
        if not self.chunks:
            return {"valid": False, "error": "No output"}

        full = self.get_full_output()

        # 1. Boundary jump analysis
        boundary_jumps = []
        for i, pos in enumerate(self.chunk_boundaries):
            if pos > 0 and pos < len(full):
                jump = abs(full[pos] - full[pos - 1])
                boundary_jumps.append({
                    "index": i + 1,
                    "position": pos,
                    "time_ms": pos * 1000 / self.sample_rate,
                    "jump": jump,
                })

        # 2. Volume analysis around boundaries
        window_ms = 20
        window_samples = int(self.sample_rate * window_ms / 1000)

        volume_analysis = []
        for i, pos in enumerate(self.chunk_boundaries):
            if pos >= window_samples and pos + window_samples <= len(full):
                before = full[pos - window_samples:pos]
                after = full[pos:pos + window_samples]
                at_boundary = full[max(0, pos - window_samples//2):min(len(full), pos + window_samples//2)]

                before_rms = np.sqrt(np.mean(before**2))
                after_rms = np.sqrt(np.mean(after**2))
                boundary_rms = np.sqrt(np.mean(at_boundary**2))

                avg_rms = (before_rms + after_rms) / 2
                if avg_rms > 0.001:
                    drop_pct = (1 - boundary_rms / avg_rms) * 100
                else:
                    drop_pct = 0

                volume_analysis.append({
                    "index": i + 1,
                    "before_rms": before_rms,
                    "after_rms": after_rms,
                    "boundary_rms": boundary_rms,
                    "drop_pct": drop_pct,
                })

        # 3. Overall RMS profile
        rms_window_ms = 50
        rms_window = int(self.sample_rate * rms_window_ms / 1000)
        rms_profile = []
        for i in range(0, len(full) - rms_window, rms_window // 2):
            rms = np.sqrt(np.mean(full[i:i + rms_window]**2))
            rms_profile.append(rms)

        return {
            "valid": True,
            "total_samples": len(full),
            "duration_sec": len(full) / self.sample_rate,
            "num_chunks": len(self.chunks),
            "chunk_lengths": [len(c) for c in self.chunks],
            "boundary_jumps": boundary_jumps,
            "volume_analysis": volume_analysis,
            "max_jump": max(b["jump"] for b in boundary_jumps) if boundary_jumps else 0,
            "avg_jump": np.mean([b["jump"] for b in boundary_jumps]) if boundary_jumps else 0,
            "max_volume_drop": max(v["drop_pct"] for v in volume_analysis) if volume_analysis else 0,
            "rms_profile": rms_profile,
        }


def run_integration_test(
    model_path: str,
    input_audio: np.ndarray,
    input_sr: int,
    config: RealtimeConfig,
    label: str,
    mode: str = "none",  # "none", "standard", "rvc_sola"
) -> dict:
    """Run full integration test with mock mic input."""
    print(f"\n{'=' * 70}")
    print(f"Integration Test: {label}")
    print(f"{'=' * 70}")

    # Create pipeline
    pipeline = RVCPipeline(model_path, device="xpu", use_compile=False)
    pipeline.load()
    print(f"Model SR: {pipeline.sample_rate}Hz")

    # Create mock mic
    mic = MockMicInput(input_audio, input_sr, block_size=4096)

    # Create buffers (exactly like RealtimeVoiceChanger)
    mic_chunk_samples = int(config.mic_sample_rate * config.chunk_sec)
    mic_context_samples = int(config.mic_sample_rate * config.context_sec)
    mic_lookahead_samples = int(config.mic_sample_rate * config.lookahead_sec)

    input_buffer = ChunkBuffer(
        mic_chunk_samples,
        crossfade_samples=0,
        context_samples=mic_context_samples,
        lookahead_samples=mic_lookahead_samples,
    )

    output_crossfade_samples = int(config.output_sample_rate * config.crossfade_sec)
    output_extra_samples = int(config.output_sample_rate * config.extra_sec)
    output_context_samples = int(config.output_sample_rate * config.context_sec)

    if mode == "sola":
        sola_state = SOLAState.create(output_crossfade_samples, config.output_sample_rate)
    # mode == "none": no crossfade

    output_collector = OutputCollector(config.output_sample_rate)

    print(f"Config: chunk={config.chunk_sec}s, crossfade={config.crossfade_sec}s, "
          f"context={config.context_sec}s, extra={config.extra_sec}s, mode={mode}")
    print(f"Samples: chunk={mic_chunk_samples}, cf={output_crossfade_samples}, "
          f"context={output_context_samples}, extra={output_extra_samples}")

    # Warmup
    print("Warmup...")
    mic_total = mic_chunk_samples + mic_context_samples + mic_lookahead_samples
    warmup_samples = int(mic_total * config.input_sample_rate / config.mic_sample_rate)

    warmup_audio = np.zeros(warmup_samples, dtype=np.float32)
    try:
        _ = pipeline.infer(
            warmup_audio,
            input_sr=config.input_sample_rate,
            pitch_shift=0,
            f0_method="rmvpe",
            index_rate=0.0,
            voice_gate_mode="off",
            use_feature_cache=False,
        )
    except:
        pass

    # Process
    print("Processing...")
    chunk_count = 0

    while True:
        block = mic.read_block()
        if block is None:
            break

        input_buffer.add_input(block)

        while input_buffer.has_chunk():
            chunk = input_buffer.get_chunk()
            if chunk is None:
                break

            # Resample to 16kHz if needed
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

            # Clip
            max_val = np.max(np.abs(output))
            if max_val > 1.0:
                output = np.tanh(output)

            if mode == "sola":
                # RVC-style SOLA crossfade on raw output
                cf_result = apply_sola_crossfade(output, sola_state)
                result_audio = cf_result.audio
            else:
                # No crossfade - just use raw output
                result_audio = output

            # Skip silent
            rms = np.sqrt(np.mean(result_audio**2))
            if rms < 0.001:
                continue

            output_collector.add_chunk(result_audio)
            chunk_count += 1

    # Flush remaining SOLA buffer
    if mode == "sola" and sola_state.sola_buffer is not None:
        # Output remaining sola_buffer with fade out
        tail = sola_state.sola_buffer * np.linspace(1, 0, len(sola_state.sola_buffer), dtype=np.float32)
        output_collector.add_chunk(tail)

    print(f"Processed {chunk_count} chunks")

    # Analyze
    analysis = output_collector.analyze()

    if not analysis["valid"]:
        print(f"ERROR: {analysis.get('error', 'Unknown')}")
        return analysis

    print(f"\nResults:")
    print(f"  Total output: {analysis['total_samples']} samples ({analysis['duration_sec']:.2f}s)")
    print(f"  Chunk count: {analysis['num_chunks']}")
    print(f"  Chunk lengths: {analysis['chunk_lengths'][:5]}..." if len(analysis['chunk_lengths']) > 5 else f"  Chunk lengths: {analysis['chunk_lengths']}")

    print(f"\nBoundary Analysis:")
    print(f"  Max jump: {analysis['max_jump']:.6f} (target < 0.1)")
    print(f"  Avg jump: {analysis['avg_jump']:.6f}")

    if analysis["boundary_jumps"]:
        big_jumps = [b for b in analysis["boundary_jumps"] if b["jump"] > 0.05]
        if big_jumps:
            print(f"  Large jumps (>0.05):")
            for b in big_jumps[:5]:
                print(f"    Boundary {b['index']}: jump={b['jump']:.4f} at {b['time_ms']:.0f}ms")

    print(f"\nVolume Analysis:")
    print(f"  Max volume drop: {analysis['max_volume_drop']:.1f}%")

    if analysis["volume_analysis"]:
        big_drops = [v for v in analysis["volume_analysis"] if v["drop_pct"] > 10]
        if big_drops:
            print(f"  Significant drops (>10%):")
            for v in big_drops[:5]:
                print(f"    Boundary {v['index']}: drop={v['drop_pct']:.1f}%")

    # Save output for manual inspection
    output_path = f"test_integration_{label.replace(' ', '_')}.wav"
    full_output = output_collector.get_full_output()
    if len(full_output) > 0:
        wavfile.write(output_path, config.output_sample_rate, full_output)
        print(f"\nSaved output to: {output_path}")

    # Determine pass/fail (jump test only - volume drops can occur naturally in speech)
    jump_pass = analysis["max_jump"] < 0.1
    analysis["jump_pass"] = jump_pass
    analysis["overall_pass"] = jump_pass

    print(f"\n  Jump test: {'PASS' if jump_pass else 'FAIL'}")

    return analysis


def main():
    # Load test audio
    voice_path = Path("sample_data/kakita.wav")
    if not voice_path.exists():
        print(f"Test audio not found: {voice_path}")
        return False

    sr, voice_data = wavfile.read(voice_path)
    if voice_data.dtype == np.int16:
        voice_data = voice_data.astype(np.float32) / 32768.0
    if len(voice_data.shape) > 1:
        voice_data = voice_data[:, 0]
    # Use first 5 seconds
    voice_data = voice_data[:int(5 * sr)]

    print(f"Input: {len(voice_data)/sr:.2f}s @ {sr}Hz")

    # Test models
    models = [
        ("C:/lib/github/grand2-products/RCWX/model/kurumi/kurumi.pth", "kurumi"),
        ("C:/lib/github/grand2-products/RCWX/model/kana/kana/voice.pth", "kana"),
    ]

    # Test configurations: (config, mode)
    # mode: "none" (no crossfade), "sola" (SOLA crossfade)
    configs = {
        "no_crossfade": (RealtimeConfig(
            mic_sample_rate=sr,
            output_sample_rate=48000,
            chunk_sec=0.35,
            context_sec=0.05,
            extra_sec=0.0,
            crossfade_sec=0.0,
            lookahead_sec=0.0,
            use_sola=False,
        ), "none"),
        "sola": (RealtimeConfig(
            mic_sample_rate=sr,
            output_sample_rate=48000,
            chunk_sec=0.35,
            context_sec=0.05,
            extra_sec=0.0,
            crossfade_sec=0.05,
            lookahead_sec=0.0,
            use_sola=True,
        ), "sola"),
    }

    results = {}

    # Test one model with different configs first
    test_model = models[0]
    if Path(test_model[0]).exists():
        for config_name, (config, mode) in configs.items():
            label = f"{test_model[1]}_{config_name}"
            results[label] = run_integration_test(
                test_model[0],
                voice_data,
                sr,
                config,
                label,
                mode=mode,
            )

    # Then test other models with RVC SOLA
    for model_path, model_name in models[1:]:
        if not Path(model_path).exists():
            print(f"Model not found: {model_path}")
            continue

        config, mode = configs["sola"]
        results[model_name] = run_integration_test(
            model_path,
            voice_data,
            sr,
            config,
            model_name,
            mode=mode,
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_pass = True
    for name, result in results.items():
        if not result.get("valid"):
            print(f"  {name}: INVALID")
            all_pass = False
        else:
            status = "PASS" if result["overall_pass"] else "FAIL"
            print(f"  {name}: {status} (max_jump={result['max_jump']:.4f})")
            if not result["overall_pass"]:
                all_pass = False

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
