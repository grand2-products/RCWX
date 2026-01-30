"""Precise boundary energy test with actual crossfade points."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.audio.crossfade import CrossfadeState, apply_crossfade


def test_precise_boundary():
    """Test energy at exact crossfade boundaries."""
    print("=" * 60)
    print("Precise boundary energy test")
    print("=" * 60)

    chunk_len = 24000  # 500ms at 48kHz
    cf_samples = 2400  # 50ms crossfade

    # Create chunks with RVC-like energy profile
    def create_chunk(chunk_num: int) -> np.ndarray:
        t = np.arange(chunk_len) / 48000
        freq = 220
        base = np.sin(2 * np.pi * freq * t).astype(np.float32)
        # Energy: head 0.16, tail 0.20
        envelope = np.linspace(0.16, 0.20, chunk_len, dtype=np.float32)
        return base * envelope

    state = CrossfadeState(cf_samples=cf_samples)
    chunks = [create_chunk(i) for i in range(5)]
    outputs = []

    print(f"\n1. Raw chunk energy profile (before crossfade):")
    for i, chunk in enumerate(chunks):
        head_rms = np.sqrt(np.mean(chunk[:100] ** 2))
        tail_rms = np.sqrt(np.mean(chunk[-100:] ** 2))
        print(f"   Chunk {i}: head={head_rms:.4f}, tail={tail_rms:.4f}, drop={100*(1-head_rms/tail_rms):.1f}%")

    print(f"\n2. Processing with crossfade (cf_samples={cf_samples})...")
    for i, chunk in enumerate(chunks):
        result = apply_crossfade(chunk.copy(), state)
        outputs.append(result.audio)

    # Measure at exact transition points
    print(f"\n3. Energy at exact boundary transitions:")

    all_discontinuities = []
    for i in range(1, len(outputs)):
        # End of previous chunk (last 100 samples)
        prev_end = outputs[i-1][-100:]
        prev_end_rms = np.sqrt(np.mean(prev_end ** 2))

        # Start of current chunk (first 100 samples of blended region)
        curr_start = outputs[i][:100]
        curr_start_rms = np.sqrt(np.mean(curr_start ** 2))

        ratio = curr_start_rms / prev_end_rms if prev_end_rms > 1e-6 else 0
        disc = abs(1 - ratio) * 100
        all_discontinuities.append(disc)

        print(f"   Boundary {i}: prev_end={prev_end_rms:.4f} -> curr_start={curr_start_rms:.4f}, "
              f"ratio={ratio:.3f}, disc={disc:.1f}%")

    # Also check within crossfade region (should be smooth)
    print(f"\n4. Energy profile WITHIN crossfade region (should be smooth):")
    for i in range(1, min(3, len(outputs))):
        cf_region = outputs[i][:cf_samples]
        n_windows = 5
        window_size = cf_samples // n_windows
        profile = []
        for w in range(n_windows):
            start = w * window_size
            end = start + window_size
            rms = np.sqrt(np.mean(cf_region[start:end] ** 2))
            profile.append(rms)
        print(f"   Chunk {i} crossfade: {' -> '.join([f'{p:.4f}' for p in profile])}")

    # Check internal discontinuity (end of crossfade -> rest of chunk)
    print(f"\n5. Internal discontinuity (end of crossfade -> rest of chunk):")
    internal_discs = []
    for i in range(1, len(outputs)):
        # End of crossfade region
        cf_end = outputs[i][cf_samples-100:cf_samples]
        cf_end_rms = np.sqrt(np.mean(cf_end ** 2))

        # Start of rest region (right after crossfade)
        rest_start = outputs[i][cf_samples:cf_samples+100]
        rest_start_rms = np.sqrt(np.mean(rest_start ** 2))

        ratio = rest_start_rms / cf_end_rms if cf_end_rms > 1e-6 else 0
        disc = abs(1 - ratio) * 100
        internal_discs.append(disc)

        print(f"   Chunk {i}: cf_end={cf_end_rms:.4f} -> rest={rest_start_rms:.4f}, "
              f"ratio={ratio:.3f}, internal_disc={disc:.1f}%")

    print(f"\n6. Results:")
    max_boundary_disc = max(all_discontinuities)
    max_internal_disc = max(internal_discs) if internal_discs else 0
    print(f"   Max boundary discontinuity: {max_boundary_disc:.1f}%")
    print(f"   Max internal discontinuity: {max_internal_disc:.1f}%")
    print(f"   Target: < 10%")

    # The boundary discontinuity should be small (crossfade handles it)
    # The internal discontinuity is the real issue
    passed = max_boundary_disc < 10.0
    print(f"\n{'[PASS]' if passed else '[FAIL]'} Boundary discontinuity {'<' if passed else '>='} 10%")

    if max_internal_disc > 10.0:
        print(f"[WARNING] Internal discontinuity {max_internal_disc:.1f}% > 10%")
        print("          This occurs at the junction between crossfade region and rest of chunk")

    return passed


if __name__ == "__main__":
    success = test_precise_boundary()
    sys.exit(0 if success else 1)
