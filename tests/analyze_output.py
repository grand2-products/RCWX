"""Analyze actual output for discontinuities."""

import sys
import numpy as np
from pathlib import Path
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_discontinuities(audio: np.ndarray, sr: int, window_ms: float = 1.0):
    """Find sample-level discontinuities (clicks/pops)."""
    window = int(sr * window_ms / 1000)

    # Calculate sample-to-sample differences
    diff = np.abs(np.diff(audio.astype(np.float64)))

    # Find large jumps (potential clicks)
    threshold = np.std(diff) * 5  # 5 sigma
    clicks = np.where(diff > threshold)[0]

    print(f"Sample-level analysis:")
    print(f"  Max diff: {np.max(diff):.6f}")
    print(f"  Mean diff: {np.mean(diff):.6f}")
    print(f"  Std diff: {np.std(diff):.6f}")
    print(f"  Threshold (5σ): {threshold:.6f}")
    print(f"  Potential clicks: {len(clicks)}")

    # Show top 10 largest jumps
    top_indices = np.argsort(diff)[-10:][::-1]
    print(f"\n  Top 10 largest jumps:")
    chunk_samples = int(sr * 0.5)
    for i, pos in enumerate(top_indices):
        time_ms = pos / sr * 1000
        chunk_num = pos // chunk_samples
        pos_in_chunk = pos % chunk_samples
        print(f"    {i+1}. sample {pos} ({time_ms:.1f}ms), chunk {chunk_num}, "
              f"pos_in_chunk={pos_in_chunk}, jump={diff[pos]:.6f}, "
              f"values: {audio[pos]:.4f} -> {audio[pos+1]:.4f}")

    return clicks


def analyze_energy_profile(audio: np.ndarray, sr: int, chunk_sec: float = 0.5):
    """Analyze energy at chunk boundaries."""
    chunk_samples = int(sr * chunk_sec)
    n_chunks = len(audio) // chunk_samples

    print(f"\nEnergy at chunk boundaries (chunk={chunk_sec}s):")

    window = int(sr * 0.005)  # 5ms window

    for i in range(1, min(n_chunks, 10)):
        pos = i * chunk_samples
        if pos + window > len(audio):
            break

        before = audio[pos - window:pos]
        after = audio[pos:pos + window]

        before_rms = np.sqrt(np.mean(before.astype(np.float64) ** 2))
        after_rms = np.sqrt(np.mean(after.astype(np.float64) ** 2))

        if before_rms > 1e-6:
            ratio = after_rms / before_rms
            disc = abs(1 - ratio) * 100
            print(f"  Boundary {i}: {before_rms:.4f} -> {after_rms:.4f}, disc={disc:.1f}%")


def plot_waveform_at_boundaries(audio: np.ndarray, sr: int, chunk_sec: float = 0.5):
    """Print waveform values around chunk boundaries."""
    chunk_samples = int(sr * chunk_sec)

    print(f"\nWaveform at first boundary (samples {chunk_samples-10} to {chunk_samples+10}):")

    pos = chunk_samples
    if pos + 10 < len(audio):
        samples = audio[pos - 10:pos + 10]
        print(f"  Before boundary: {audio[pos-5:pos]}")
        print(f"  After boundary:  {audio[pos:pos+5]}")
        print(f"  Jump at boundary: {abs(audio[pos] - audio[pos-1]):.6f}")


def main():
    # Check for test output file
    test_output = Path("tests/test_realtime_direct_output.wav")
    if not test_output.exists():
        print(f"Test output not found: {test_output}")
        print("Run test_realtime_changer_direct.py first")
        return

    print("=" * 70)
    print("Analyzing test output for discontinuities")
    print("=" * 70)

    sr, audio = wavfile.read(test_output)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    print(f"\nFile: {test_output}")
    print(f"Duration: {len(audio)/sr:.2f}s @ {sr}Hz")
    print(f"Samples: {len(audio)}")
    print(f"Range: [{audio.min():.4f}, {audio.max():.4f}]")

    # Find clicks
    clicks = analyze_discontinuities(audio, sr)

    # Analyze energy at boundaries
    analyze_energy_profile(audio, sr)

    # Show waveform at boundaries
    plot_waveform_at_boundaries(audio, sr)

    # Check for zero crossings at boundaries
    chunk_samples = int(sr * 0.5)
    print(f"\nZero-crossing analysis at boundaries:")
    for i in range(1, 5):
        pos = i * chunk_samples
        if pos >= len(audio):
            break
        before_sign = np.sign(audio[pos-1])
        after_sign = np.sign(audio[pos])
        crossing = "YES" if before_sign != after_sign else "NO"
        print(f"  Boundary {i}: before={audio[pos-1]:.4f}, after={audio[pos]:.4f}, zero-cross={crossing}")


if __name__ == "__main__":
    main()
