"""Final boundary energy verification test."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.audio.crossfade import CrossfadeState, apply_crossfade


def test_realistic_rvc_profile():
    """Test with realistic RVC energy profile (head low, tail high)."""
    print("=" * 70)
    print("Final Boundary Energy Verification")
    print("=" * 70)

    # Realistic parameters
    sr = 48000
    chunk_sec = 0.5
    cf_sec = 0.05
    n_chunks = 10

    chunk_len = int(sr * chunk_sec)
    cf_samples = int(sr * cf_sec)

    print(f"\nParameters:")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Chunk: {chunk_sec}s ({chunk_len} samples)")
    print(f"  Crossfade: {cf_sec}s ({cf_samples} samples)")
    print(f"  Chunks: {n_chunks}")

    # Create chunks with RVC-like energy profile
    # Based on actual measurements: head ~0.16, tail ~0.20 (20% increase)
    def create_rvc_chunk(chunk_num: int) -> np.ndarray:
        t = np.arange(chunk_len) / sr
        # Simple sine wave base signal
        freq = 220
        base = np.sin(2 * np.pi * freq * t).astype(np.float32)

        # RVC energy envelope: 20% gradient from head to tail
        envelope = np.linspace(0.16, 0.20, chunk_len, dtype=np.float32)
        return base * envelope

    # Process chunks
    state = CrossfadeState(cf_samples=cf_samples)
    outputs = []

    print(f"\nProcessing {n_chunks} chunks...")
    for i in range(n_chunks):
        chunk = create_rvc_chunk(i)
        result = apply_crossfade(chunk, state)
        outputs.append(result.audio)

    # Concatenate
    full_output = np.concatenate(outputs)

    # Analyze all boundaries
    print(f"\nBoundary Analysis:")
    print("-" * 50)

    boundary_discs = []
    window = 100  # 2ms window for energy measurement

    for i in range(1, n_chunks):
        # Position of boundary in concatenated output
        pos = sum(len(outputs[j]) for j in range(i))

        # Energy just before and after boundary
        before = full_output[pos - window:pos]
        after = full_output[pos:pos + window]

        before_rms = np.sqrt(np.mean(before ** 2))
        after_rms = np.sqrt(np.mean(after ** 2))

        ratio = after_rms / before_rms if before_rms > 1e-6 else 0
        disc = abs(1 - ratio) * 100
        boundary_discs.append(disc)

        if i <= 5 or i == n_chunks - 1:
            print(f"  Boundary {i:2d}: {before_rms:.4f} -> {after_rms:.4f}, disc={disc:.1f}%")

    if n_chunks > 6:
        print(f"  ... ({n_chunks - 6} more boundaries) ...")

    # Overall statistics
    print(f"\nResults:")
    print(f"  Mean discontinuity: {np.mean(boundary_discs):.1f}%")
    print(f"  Max discontinuity:  {np.max(boundary_discs):.1f}%")
    print(f"  Min discontinuity:  {np.min(boundary_discs):.1f}%")
    print(f"  Std discontinuity:  {np.std(boundary_discs):.1f}%")
    print(f"\n  Target: < 10%")

    # Overall energy stability (no drift)
    first_chunk_energy = np.sqrt(np.mean(outputs[0] ** 2))
    last_chunk_energy = np.sqrt(np.mean(outputs[-1] ** 2))
    energy_drift = abs(last_chunk_energy - first_chunk_energy) / first_chunk_energy * 100

    print(f"\n  Energy drift (first->last): {energy_drift:.1f}%")
    print(f"  (Should be small to avoid volume drift)")

    # Pass criteria
    max_disc = np.max(boundary_discs)
    passed = max_disc < 10.0

    print(f"\n{'=' * 50}")
    print(f"{'[PASS]' if passed else '[FAIL]'} Max discontinuity {max_disc:.1f}% {'<' if passed else '>='} 10%")
    print(f"{'=' * 50}")

    return passed


if __name__ == "__main__":
    success = test_realistic_rvc_profile()
    sys.exit(0 if success else 1)
