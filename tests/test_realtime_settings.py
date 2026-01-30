"""Test that settings are applied in real-time."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeVoiceChanger, RealtimeConfig
from rcwx.audio.crossfade import SOLAState


def test_realtime_settings():
    """Test that settings can be changed while running."""
    print("Testing real-time settings changes...")

    model_path = "C:/lib/github/grand2-products/RCWX/model/kurumi/kurumi.pth"
    if not Path(model_path).exists():
        print("Model not found")
        return False

    # Create pipeline and changer
    pipeline = RVCPipeline(model_path, device="xpu", use_compile=False)
    pipeline.load()

    changer = RealtimeVoiceChanger(pipeline)

    # Simulate running state
    changer._running = True
    changer._recalculate_buffers()
    changer._sola_state = SOLAState.create(
        changer.output_crossfade_samples,
        changer.config.output_sample_rate,
    )

    print(f"\nInitial settings:")
    print(f"  context_sec: {changer.config.context_sec}")
    print(f"  crossfade_sec: {changer.config.crossfade_sec}")
    print(f"  use_sola: {changer.config.use_sola}")
    print(f"  mic_context_samples: {changer.mic_context_samples}")
    print(f"  output_crossfade_samples: {changer.output_crossfade_samples}")
    print(f"  sola_state: {changer._sola_state is not None}")

    # Test 1: Change context
    print("\n--- Test 1: Change context ---")
    old_context = changer.mic_context_samples
    changer.set_context(0.1)  # 100ms
    new_context = changer.mic_context_samples
    print(f"  context_sec: {changer.config.context_sec}")
    print(f"  mic_context_samples: {old_context} -> {new_context}")
    assert changer.config.context_sec == 0.1, "context_sec not updated"
    assert new_context != old_context, "mic_context_samples not recalculated"
    print("  PASS")

    # Test 2: Disable SOLA
    print("\n--- Test 2: Disable SOLA ---")
    assert changer._sola_state is not None, "SOLA state should exist"
    changer.set_sola(False)
    print(f"  use_sola: {changer.config.use_sola}")
    print(f"  sola_state: {changer._sola_state}")
    assert changer.config.use_sola == False, "use_sola not updated"
    assert changer._sola_state is None, "SOLA state should be None when disabled"
    print("  PASS")

    # Test 3: Re-enable SOLA
    print("\n--- Test 3: Re-enable SOLA ---")
    changer.set_sola(True)
    print(f"  use_sola: {changer.config.use_sola}")
    print(f"  sola_state: {changer._sola_state is not None}")
    assert changer.config.use_sola == True, "use_sola not updated"
    assert changer._sola_state is not None, "SOLA state should be created"
    print("  PASS")

    # Test 4: Change crossfade
    print("\n--- Test 4: Change crossfade ---")
    old_cf = changer.output_crossfade_samples
    old_sola_buffer = changer._sola_state.sola_buffer_frame
    changer.set_crossfade(0.1)  # 100ms
    new_cf = changer.output_crossfade_samples
    new_sola_buffer = changer._sola_state.sola_buffer_frame
    print(f"  crossfade_sec: {changer.config.crossfade_sec}")
    print(f"  output_crossfade_samples: {old_cf} -> {new_cf}")
    print(f"  sola_buffer_frame: {old_sola_buffer} -> {new_sola_buffer}")
    assert changer.config.crossfade_sec == 0.1, "crossfade_sec not updated"
    assert new_cf != old_cf, "output_crossfade_samples not recalculated"
    print("  PASS")

    # Test 5: Change extra
    print("\n--- Test 5: Change extra ---")
    changer.set_extra(0.02)  # 20ms
    print(f"  extra_sec: {changer.config.extra_sec}")
    assert changer.config.extra_sec == 0.02, "extra_sec not updated"
    print("  PASS")

    # Test 6: Change lookahead
    print("\n--- Test 6: Change lookahead ---")
    old_lookahead = changer.mic_lookahead_samples
    changer.set_lookahead(0.05)  # 50ms
    new_lookahead = changer.mic_lookahead_samples
    print(f"  lookahead_sec: {changer.config.lookahead_sec}")
    print(f"  mic_lookahead_samples: {old_lookahead} -> {new_lookahead}")
    assert changer.config.lookahead_sec == 0.05, "lookahead_sec not updated"
    assert new_lookahead != old_lookahead, "mic_lookahead_samples not recalculated"
    print("  PASS")

    print("\n" + "="*50)
    print("All real-time settings tests PASSED")
    return True


if __name__ == "__main__":
    success = test_realtime_settings()
    sys.exit(0 if success else 1)
