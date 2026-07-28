"""Verify the SOLA synthesis margin: decoder overlap, alignment, and that
the GUI estimate reports the same figure the runtime actually reserves.

Every check calls the production helpers.  Earlier revisions re-implemented
the formula inline, so they kept passing while the real one diverged.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rcwx.audio.sola import SolaState
from rcwx.gui.widgets.latency_settings import _auto_params
from rcwx.pipeline.realtime_config import DEFAULT_DECODER_OVERLAP_FRAMES, sola_margin_ms
from rcwx.pipeline.realtime_unified import (
    RealtimeConfig,
    RealtimeVoiceChangerUnified,
    _compute_sola_extra_model,
    _effective_decoder_overlap_frames,
)


def test_sola_extra_uses_one_search_window() -> None:
    extra = _compute_sola_extra_model(
        48000,
        48000,
        crossfade_samples_out=480,
        search_samples_out=720,
        decoder_overlap_frames=0,
    )

    # 25ms is rounded up to the decoder's 10ms (480-sample) boundary.
    assert extra == 1440
    print("PASS: crossfade+search counted once, aligned to the 10ms grid")


def test_aggressive_reduces_decoder_overlap() -> None:
    """Aggressive uses 2 frames (20ms), not the configured 5 and not 0.

    overlap=0 was audible as sustained-tone modulation at the chunk
    boundary; see _effective_decoder_overlap_frames.
    """
    assert _effective_decoder_overlap_frames("aggressive", 5) == 2
    assert _effective_decoder_overlap_frames("normal", 5) == 5
    print("PASS: aggressive=2 frames, normal=configured")


def test_decoder_overlap_increases_sola_extra() -> None:
    """decoder_overlap_frames=5 adds exactly 5 * zc_model samples."""
    model_sr, output_sr = 32000, 48000
    zc_model = model_sr // 100  # 320
    crossfade_samples_out = int(output_sr * 0.05)
    search_samples_out = int(output_sr * 10.0 / 1000)

    def extra(frames: int) -> int:
        return _compute_sola_extra_model(
            model_sr, output_sr, crossfade_samples_out, search_samples_out, frames
        )

    diff = extra(5) - extra(0)
    assert diff == 5 * zc_model, f"Expected {5 * zc_model}, got {diff}"
    print(f"PASS: decoder_overlap adds {diff} samples ({diff * 1000 / model_sr:.0f}ms)")


def test_sola_extra_aligned_at_every_model_rate() -> None:
    """The margin must land on the model's zero-crossing grid at every rate.

    trim_left in infer_streaming assumes a whole number of frames; a
    sub-frame residue shifts the output boundary.
    """
    for model_sr in (32000, 40000, 48000):
        for output_sr in (44100, 48000):
            zc_model = model_sr // 100
            cf = int(output_sr * 0.08)
            search = int(output_sr * 10.0 / 1000)
            extra = _compute_sola_extra_model(model_sr, output_sr, cf, search, 5)
            assert extra % zc_model == 0, (
                f"{model_sr}->{output_sr}: {extra} not aligned to {zc_model}"
            )
            # Must still cover the crossfade+search the splice consumes.
            required_out = cf + search
            assert extra * output_sr >= required_out * model_sr, (
                f"{model_sr}->{output_sr}: margin {extra} below required {required_out}"
            )
    print("PASS: margin aligned and sufficient at 32k/40k/48k x 44.1k/48k")


def test_decoder_overlap_default_is_5() -> None:
    cfg = RealtimeConfig()
    assert cfg.decoder_overlap_frames == 5, f"Expected 5, got {cfg.decoder_overlap_frames}"
    print("PASS: default decoder_overlap_frames == 5")


def test_gui_estimate_matches_runtime_margin() -> None:
    """The GUI's latency estimate must report the margin the runtime reserves.

    These were separate formulas once, and the GUI's omitted the decoder
    overlap entirely -- understating Normal by 60ms.
    """
    for mode, chunk_sec in [
        ("aggressive", 0.02),
        ("aggressive", 0.10),
        ("normal", 0.04),
        ("normal", 0.30),
        ("normal", 0.60),
    ]:
        auto = _auto_params(chunk_sec, mode)
        estimate = sola_margin_ms(
            auto["crossfade_sec"],
            auto["sola_search_ms"],
            mode,
            DEFAULT_DECODER_OVERLAP_FRAMES,
        )
        # The runtime figure is rate-dependent in samples but not in ms.
        for model_sr, output_sr in [(48000, 44100), (40000, 48000), (32000, 44100)]:
            extra = _compute_sola_extra_model(
                model_sr,
                output_sr,
                int(output_sr * auto["crossfade_sec"]),
                int(output_sr * auto["sola_search_ms"] / 1000),
                _effective_decoder_overlap_frames(
                    mode, DEFAULT_DECODER_OVERLAP_FRAMES
                ),
            )
            actual_ms = extra * 1000.0 / model_sr
            assert abs(actual_ms - estimate) < 1e-6, (
                f"{mode} {chunk_sec * 1000:.0f}ms at {model_sr}->{output_sr}: "
                f"estimate {estimate}ms != runtime {actual_ms}ms"
            )
    print("PASS: GUI estimate == runtime margin at every rate")


def test_gui_margin_fits_the_hop() -> None:
    """SOLA needs offset+crossfade of margin per chunk; the rest is headroom.

    If this ever fails, sola_crossfade silently abandons its fixed-length
    branch and the output length drifts away from one hop.
    """
    output_sr = 44100
    for mode, chunk_sec in [("aggressive", 0.02), ("normal", 0.04), ("normal", 0.30)]:
        auto = _auto_params(chunk_sec, mode)
        cf = int(output_sr * auto["crossfade_sec"])
        search = int(output_sr * auto["sola_search_ms"] / 1000)
        margin_out = sola_margin_ms(
            auto["crossfade_sec"], auto["sola_search_ms"], mode,
            DEFAULT_DECODER_OVERLAP_FRAMES,
        ) * output_sr / 1000.0
        assert margin_out >= search + cf, (
            f"{mode} {chunk_sec * 1000:.0f}ms: margin {margin_out} < "
            f"worst-case offset+crossfade {search + cf}"
        )
    print("PASS: margin covers worst-case offset+crossfade in every preset")


def _rebuilt(**config_kwargs) -> RealtimeVoiceChangerUnified:
    """A VC with only the state _rebuild_sola touches (no model, no warmup)."""
    settings = {
        "chunk_sec": 0.02,
        "f0_method": "swiftf0",
        "latency_mode": "aggressive",
        "crossfade_sec": 0.010,
        "sola_search_ms": 10.0,
    }
    settings.update(config_kwargs)
    vc = RealtimeVoiceChangerUnified.__new__(RealtimeVoiceChangerUnified)
    vc.config = RealtimeConfig(**settings)
    vc.pipeline = SimpleNamespace(sample_rate=48000)
    vc._runtime_output_sample_rate = 44100
    vc._sola_state = SolaState()
    vc._rebuild_sola()
    return vc


def test_no_margin_when_the_splice_never_runs() -> None:
    """A margin nothing consumes becomes unbounded ring growth.

    infer_streaming emits hop+margin, and only sola_crossfade gives the extra
    back.  With SOLA off -- or a zero-length crossfade, which makes
    sola_crossfade a passthrough -- every chunk would overfill the ring until
    drift control hard-skips it, which is audible.
    """
    assert _rebuilt()._sola_extra_model > 0, "the normal path still needs its margin"
    assert _rebuilt(use_sola=False)._sola_extra_model == 0
    assert _rebuilt(crossfade_sec=0.0)._sola_extra_model == 0
    print("PASS: margin is produced only when the splice consumes it")


def test_sola_resize_drops_stale_holdback() -> None:
    """Changing the crossfade length must drop the old hold-back.

    The buffer is exactly crossfade_samples long; keeping one sized for the
    previous window broadcast-mismatches the new Hann curves and raises
    ValueError on the inference thread.
    """
    state = SolaState(crossfade_samples=441, search_samples=441)
    state.buffer = np.zeros(441, dtype=np.float32)
    state._ensure_window()

    state.resize(882, 441)
    assert state.buffer is None, "stale hold-back survived a crossfade change"
    assert state._hann_fade_in is None, "stale Hann window survived"
    assert state.crossfade_samples == 882

    # Search-only changes carry no state, so the hold-back is preserved.
    state.buffer = np.zeros(882, dtype=np.float32)
    state.resize(882, 220)
    assert state.buffer is not None, "hold-back dropped for a search-only change"
    assert state.search_samples == 220
    print("PASS: resize drops the hold-back only when the crossfade changes")


if __name__ == "__main__":
    test_sola_extra_uses_one_search_window()
    test_aggressive_reduces_decoder_overlap()
    test_decoder_overlap_increases_sola_extra()
    test_sola_extra_aligned_at_every_model_rate()
    test_decoder_overlap_default_is_5()
    test_gui_estimate_matches_runtime_margin()
    test_gui_margin_fits_the_hop()
    test_no_margin_when_the_splice_never_runs()
    test_sola_resize_drops_stale_holdback()
    print("\nAll decoder overlap / SOLA margin tests passed.")
