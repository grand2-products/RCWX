"""Test boundary energy continuity with fixed crossfade."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.audio.crossfade import CrossfadeState, apply_crossfade


def test_energy_compensation():
    """Test that crossfade properly compensates for RVC energy discontinuity."""
    print("=" * 60)
    print("Testing crossfade energy compensation")
    print("=" * 60)

    # Simulate RVC output characteristics:
    # - Head energy: ~0.16
    # - Tail energy: ~0.20
    # This creates 20% discontinuity at boundaries

    chunk_len = 24000  # 500ms at 48kHz
    cf_samples = 2400  # 50ms crossfade

    # Create chunks with RVC-like energy profile
    def create_rvc_like_chunk(chunk_num: int) -> np.ndarray:
        """Create a chunk that mimics RVC output energy profile."""
        t = np.arange(chunk_len) / 48000
        # Base signal (sine wave)
        freq = 220 * (1 + chunk_num * 0.1)  # Slightly different freq per chunk
        base = np.sin(2 * np.pi * freq * t).astype(np.float32)

        # Energy envelope: low at head (0.16), high at tail (0.20)
        # Linear ramp from 0.16 to 0.20
        envelope = np.linspace(0.16, 0.20, chunk_len, dtype=np.float32)
        return base * envelope

    # Process multiple chunks
    state = CrossfadeState(cf_samples=cf_samples)
    outputs = []
    n_chunks = 5

    print(f"\nProcessing {n_chunks} chunks with cf_samples={cf_samples}...")

    for i in range(n_chunks):
        chunk = create_rvc_like_chunk(i)

        # Measure input energy at boundaries
        head_energy = np.sqrt(np.mean(chunk[:cf_samples] ** 2))
        tail_energy = np.sqrt(np.mean(chunk[-cf_samples:] ** 2))

        result = apply_crossfade(chunk, state)
        outputs.append(result.audio)

        # Measure output energy at boundaries
        out_head = np.sqrt(np.mean(result.audio[:cf_samples] ** 2))
        out_tail = np.sqrt(np.mean(result.audio[-cf_samples:] ** 2))

        print(f"  Chunk {i}: in_head={head_energy:.3f}, in_tail={tail_energy:.3f} "
              f"-> out_head={out_head:.3f}, out_tail={out_tail:.3f}")

    # Concatenate and analyze boundary continuity
    full_output = np.concatenate(outputs)

    print(f"\nAnalyzing boundaries (cf_samples={cf_samples})...")

    # Analyze energy at each chunk boundary
    discontinuities = []
    for i in range(1, n_chunks):
        # Position just before and after boundary in concatenated output
        pos = i * len(outputs[0])  # Approximate position

        # Get energy in small windows around boundary
        window = 480  # 10ms window
        if pos - window >= 0 and pos + window < len(full_output):
            before = full_output[pos - window:pos]
            after = full_output[pos:pos + window]

            before_rms = np.sqrt(np.mean(before ** 2))
            after_rms = np.sqrt(np.mean(after ** 2))

            if before_rms > 1e-6:
                ratio = after_rms / before_rms
                discontinuity = abs(1.0 - ratio) * 100
                discontinuities.append(discontinuity)
                print(f"  Boundary {i}: before={before_rms:.4f}, after={after_rms:.4f}, "
                      f"ratio={ratio:.3f}, discontinuity={discontinuity:.1f}%")

    if discontinuities:
        max_disc = max(discontinuities)
        avg_disc = np.mean(discontinuities)
        print(f"\nResults:")
        print(f"  Max discontinuity: {max_disc:.1f}%")
        print(f"  Avg discontinuity: {avg_disc:.1f}%")
        print(f"  Target: < 10%")

        passed = max_disc < 10.0
        print(f"\n{'[PASS]' if passed else '[FAIL]'} Max discontinuity {'<' if passed else '>='} 10%")
        return passed

    print("[FAIL] Could not analyze boundaries")
    return False


def test_without_compensation():
    """Show what happens without energy compensation (baseline)."""
    print("\n" + "=" * 60)
    print("Baseline: What would happen WITHOUT energy compensation")
    print("=" * 60)

    chunk_len = 24000
    cf_samples = 2400

    def create_rvc_like_chunk(chunk_num: int) -> np.ndarray:
        t = np.arange(chunk_len) / 48000
        freq = 220 * (1 + chunk_num * 0.1)
        base = np.sin(2 * np.pi * freq * t).astype(np.float32)
        envelope = np.linspace(0.16, 0.20, chunk_len, dtype=np.float32)
        return base * envelope

    # Simulate naive crossfade (no energy compensation)
    chunks = [create_rvc_like_chunk(i) for i in range(5)]

    print("\nWithout energy compensation:")
    for i in range(1, len(chunks)):
        prev_tail = chunks[i-1][-cf_samples:]
        curr_head = chunks[i][:cf_samples]

        prev_rms = np.sqrt(np.mean(prev_tail ** 2))
        curr_rms = np.sqrt(np.mean(curr_head ** 2))
        ratio = curr_rms / prev_rms
        discontinuity = abs(1.0 - ratio) * 100

        print(f"  Boundary {i}: prev_tail={prev_rms:.4f}, curr_head={curr_rms:.4f}, "
              f"discontinuity={discontinuity:.1f}%")

    print("\n(This shows the raw 20% discontinuity from RVC output)")


if __name__ == "__main__":
    test_without_compensation()
    print("\n")
    success = test_energy_compensation()
    sys.exit(0 if success else 1)
