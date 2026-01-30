"""Integration test: Verify streaming (chunked) output matches batch output."""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.audio.buffer import ChunkBuffer
from rcwx.audio.crossfade import SOLAState, apply_sola_crossfade
from rcwx.audio.resample import resample
from rcwx.device import get_device
from rcwx.pipeline.inference import RVCPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_test_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load and resample audio file."""
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        audio = resample(audio, sr, target_sr)
    return audio.astype(np.float32)


def process_batch(pipeline: RVCPipeline, audio: np.ndarray, pitch_shift: int = 0) -> np.ndarray:
    """Process entire audio in one batch."""
    audio_tensor = torch.from_numpy(audio).float()
    output = pipeline.infer(
        audio_tensor,
        pitch_shift=pitch_shift,
        f0_method="rmvpe",
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=False,
    )
    return output


def process_streaming(
    pipeline: RVCPipeline,
    audio: np.ndarray,
    pitch_shift: int = 0,
    chunk_sec: float = 0.35,
    context_sec: float = 0.05,
    crossfade_sec: float = 0.05,
    use_sola: bool = True,
    input_sr: int = 16000,
) -> np.ndarray:
    """Process audio in streaming chunks, simulating real-time behavior."""
    model_sr = pipeline.sample_rate  # Usually 40000 or 48000

    # Calculate sample counts
    chunk_samples = int(input_sr * chunk_sec)
    context_samples = int(input_sr * context_sec)

    # Output processing parameters
    out_context_samples = int(model_sr * context_sec)
    out_crossfade_samples = int(model_sr * crossfade_sec)

    # Create input buffer (simulates ChunkBuffer behavior)
    input_buffer = ChunkBuffer(
        chunk_samples=chunk_samples,
        crossfade_samples=0,
        context_samples=context_samples,
        lookahead_samples=0,
    )

    # SOLA state for crossfading
    sola_state = SOLAState.create(out_crossfade_samples, model_sr) if use_sola else None

    # Output accumulator
    outputs = []
    prev_output = None

    # Simulate streaming: feed audio in small blocks (like audio callback)
    block_size = 1024  # Typical audio callback size
    pos = 0
    chunk_count = 0

    while pos < len(audio):
        # Simulate audio input callback
        block = audio[pos : pos + block_size]
        if len(block) < block_size:
            block = np.pad(block, (0, block_size - len(block)))
        pos += block_size

        # Add to input buffer
        input_buffer.add_input(block)

        # Check if we have enough for a chunk
        while input_buffer.has_chunk():
            chunk = input_buffer.get_chunk()
            if chunk is None:
                break

            chunk_count += 1

            # Process chunk through pipeline
            chunk_tensor = torch.from_numpy(chunk).float()
            output = pipeline.infer(
                chunk_tensor,
                pitch_shift=pitch_shift,
                f0_method="rmvpe",
                index_rate=0.0,
                voice_gate_mode="off",
                use_feature_cache=True,  # Enable for streaming continuity
            )

            # Trim context from output (w-okada style)
            if out_context_samples > 0 and len(output) > 2 * out_context_samples:
                output = output[out_context_samples:-out_context_samples]

            # Apply crossfade with previous output
            if prev_output is not None and out_crossfade_samples > 0:
                if use_sola and sola_state is not None:
                    output = apply_sola_crossfade(sola_state, prev_output, output)
                else:
                    # Simple linear crossfade
                    cf_len = min(out_crossfade_samples, len(prev_output), len(output))
                    if cf_len > 0:
                        fade_out = np.linspace(1, 0, cf_len, dtype=np.float32)
                        fade_in = np.linspace(0, 1, cf_len, dtype=np.float32)
                        output[:cf_len] = prev_output[-cf_len:] * fade_out + output[:cf_len] * fade_in

            outputs.append(output)
            prev_output = output

    if not outputs:
        return np.array([], dtype=np.float32)

    # Concatenate all outputs
    result = np.concatenate(outputs)
    logger.info(f"Streaming: processed {chunk_count} chunks, output length: {len(result)}")
    return result


def compare_outputs(
    batch_output: np.ndarray,
    streaming_output: np.ndarray,
    tolerance_db: float = -20.0,
) -> dict:
    """Compare batch and streaming outputs."""
    # Align lengths (streaming may be slightly different due to chunking)
    min_len = min(len(batch_output), len(streaming_output))
    if min_len == 0:
        return {"error": "Empty output"}

    batch = batch_output[:min_len]
    streaming = streaming_output[:min_len]

    # Calculate difference
    diff = batch - streaming
    diff_rms = np.sqrt(np.mean(diff**2))
    batch_rms = np.sqrt(np.mean(batch**2))

    # Calculate SNR (signal-to-noise ratio where "noise" is the difference)
    if diff_rms > 0:
        snr_db = 20 * np.log10(batch_rms / diff_rms)
    else:
        snr_db = float("inf")

    # Calculate correlation
    if batch_rms > 0 and np.sqrt(np.mean(streaming**2)) > 0:
        correlation = np.corrcoef(batch, streaming)[0, 1]
    else:
        correlation = 0.0

    # Check if within tolerance
    passed = snr_db >= -tolerance_db  # tolerance_db is negative, so we compare >= -(-20) = 20

    return {
        "batch_length": len(batch_output),
        "streaming_length": len(streaming_output),
        "compared_length": min_len,
        "diff_rms": diff_rms,
        "batch_rms": batch_rms,
        "snr_db": snr_db,
        "correlation": correlation,
        "passed": passed,
        "tolerance_db": tolerance_db,
    }


def run_test(
    model_path: Path,
    audio_path: Path,
    output_dir: Path,
    pitch_shift: int = 0,
):
    """Run the streaming vs batch comparison test."""
    output_dir.mkdir(exist_ok=True)

    # Load model
    logger.info(f"Loading model: {model_path}")
    device = get_device("auto")
    pipeline = RVCPipeline(model_path, device=device)
    pipeline.load()
    logger.info(f"Model loaded, sample rate: {pipeline.sample_rate}Hz")

    # Load audio
    logger.info(f"Loading audio: {audio_path}")
    audio = load_test_audio(audio_path, target_sr=16000)
    logger.info(f"Audio loaded: {len(audio)} samples, {len(audio)/16000:.2f}s")

    # Process batch
    logger.info("Processing batch...")
    batch_output = process_batch(pipeline, audio, pitch_shift)
    logger.info(f"Batch output: {len(batch_output)} samples")

    # Process streaming with different settings
    test_configs = [
        {"chunk_sec": 0.35, "context_sec": 0.05, "crossfade_sec": 0.05, "use_sola": True},
        {"chunk_sec": 0.35, "context_sec": 0.05, "crossfade_sec": 0.05, "use_sola": False},
        {"chunk_sec": 0.2, "context_sec": 0.03, "crossfade_sec": 0.03, "use_sola": True},
        {"chunk_sec": 0.5, "context_sec": 0.08, "crossfade_sec": 0.08, "use_sola": True},
    ]

    results = []
    for i, config in enumerate(test_configs):
        logger.info(f"\nTest {i+1}: {config}")
        streaming_output = process_streaming(pipeline, audio, pitch_shift, **config)

        # Compare
        comparison = compare_outputs(batch_output, streaming_output)
        comparison["config"] = config
        results.append(comparison)

        # Save outputs
        batch_path = output_dir / f"batch_output.wav"
        streaming_path = output_dir / f"streaming_{i+1}.wav"

        if i == 0:  # Save batch only once
            wavfile.write(batch_path, pipeline.sample_rate, batch_output)
            logger.info(f"Saved: {batch_path}")

        wavfile.write(streaming_path, pipeline.sample_rate, streaming_output)
        logger.info(f"Saved: {streaming_path}")

        # Report
        logger.info(f"  SNR: {comparison['snr_db']:.1f} dB")
        logger.info(f"  Correlation: {comparison['correlation']:.4f}")
        logger.info(f"  Passed: {comparison['passed']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for i, result in enumerate(results):
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        all_passed = all_passed and result["passed"]
        print(f"Test {i+1}: {status} (SNR: {result['snr_db']:.1f}dB, Corr: {result['correlation']:.4f})")
        print(f"         Config: chunk={result['config']['chunk_sec']}s, "
              f"ctx={result['config']['context_sec']}s, sola={result['config']['use_sola']}")

    print("=" * 60)
    if all_passed:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
    print("=" * 60)

    return all_passed, results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test streaming vs batch processing")
    parser.add_argument("--model", "-m", type=Path, required=True, help="RVC model path")
    parser.add_argument("--audio", "-a", type=Path, required=True, help="Input audio path")
    parser.add_argument("--output", "-o", type=Path, default=Path("test_output"), help="Output directory")
    parser.add_argument("--pitch", "-p", type=int, default=0, help="Pitch shift (semitones)")
    args = parser.parse_args()

    if not args.model.exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)
    if not args.audio.exists():
        logger.error(f"Audio not found: {args.audio}")
        sys.exit(1)

    passed, _ = run_test(args.model, args.audio, args.output, args.pitch)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
