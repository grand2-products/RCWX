"""Check input file for issues."""

import sys
import numpy as np
from pathlib import Path
from scipy.io import wavfile

def main():
    input_file = Path("sample_data/pure_sine.wav")
    if not input_file.exists():
        print(f"Input not found: {input_file}")
        return

    sr, audio = wavfile.read(input_file)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    print(f"Input: {input_file}")
    print(f"Duration: {len(audio)/sr:.2f}s @ {sr}Hz")
    print(f"Range: [{audio.min():.4f}, {audio.max():.4f}]")

    # Check for discontinuities
    diff = np.abs(np.diff(audio.astype(np.float64)))
    threshold = np.std(diff) * 5
    clicks = np.where(diff > threshold)[0]

    print(f"\nSample-level analysis:")
    print(f"  Max diff: {np.max(diff):.6f}")
    print(f"  Mean diff: {np.mean(diff):.6f}")
    print(f"  Threshold (5σ): {threshold:.6f}")
    print(f"  Potential clicks: {len(clicks)}")

    # Frequency analysis
    from scipy.fft import rfft, rfftfreq
    n = len(audio)
    yf = np.abs(rfft(audio))
    xf = rfftfreq(n, 1/sr)
    peak_freq = xf[np.argmax(yf[1:])+1]  # Skip DC
    print(f"\nPeak frequency: {peak_freq:.1f} Hz")

if __name__ == "__main__":
    main()
