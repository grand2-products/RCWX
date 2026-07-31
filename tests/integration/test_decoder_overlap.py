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

from rcwx.audio.sola import SolaState, sola_crossfade
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

    The rates the GUI actually produces all divide evenly, so the round-up is
    a no-op there and asserting it on those alone proves nothing -- an earlier
    version of this test passed with BOTH the round-up and the decoder-overlap
    term deleted.  The 15ms crossfade case below does not divide evenly
    (32000/44100 puts the requirement at 799.6 samples), so it pins all three
    terms to hand-computed values.
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

    # model 32000 -> out 44100, crossfade 15ms (661) + search 10ms (441):
    #   required_model = ceil(1102 * 32000/44100) = ceil(799.63) = 800
    #   + 5 frames * 320                          = 2400
    #   rounded up to the 320 grid (7.5 -> 8)     = 2560
    # Deleting the round-up gives 2400; deleting the overlap term gives 960.
    cf_15ms = int(44100 * 0.015)
    search_10ms = int(44100 * 10.0 / 1000)
    assert _compute_sola_extra_model(32000, 44100, cf_15ms, search_10ms, 5) == 2560
    assert _compute_sola_extra_model(32000, 44100, cf_15ms, search_10ms, 0) == 960
    print("PASS: margin aligned at every rate, round-up and overlap pinned")


def test_decoder_overlap_default_is_5() -> None:
    cfg = RealtimeConfig()
    assert cfg.decoder_overlap_frames == 5, f"Expected 5, got {cfg.decoder_overlap_frames}"
    print("PASS: default decoder_overlap_frames == 5")


def test_gui_derived_latency_values_are_pinned() -> None:
    """Pin the values _auto_params hands the runtime.

    Everything else here only checks that the GUI and the runtime agree with
    each other, so both could drift together unnoticed -- mutating
    ``sola_search_ms`` to 20.0 or the crossfade floor to 30ms killed no test.
    These are the numbers the latency budget is actually built from.
    """
    # crossfade + search <= 20ms is what keeps the margin one 10ms model
    # frame smaller; raising either silently costs 10ms of latency.
    for mode in ("aggressive", "normal"):
        for chunk_sec in (0.02, 0.04, 0.10, 0.16, 0.30, 0.60):
            auto = _auto_params(chunk_sec, mode)
            assert auto["sola_search_ms"] == 10.0, (
                f"{mode} {chunk_sec}: search {auto['sola_search_ms']} != 10.0"
            )
            assert auto["use_sola"] is True

    # crossfade is 10% of the chunk on a 10ms grid, floored at 10 and capped
    # at 20; the floor is what holds cf+search at 20ms for small chunks.
    expected_crossfade_ms = {0.02: 10, 0.04: 10, 0.10: 10, 0.16: 20, 0.30: 20, 0.60: 20}
    for chunk_sec, cf_ms in expected_crossfade_ms.items():
        auto = _auto_params(chunk_sec, "normal")
        assert round(auto["crossfade_sec"] * 1000) == cf_ms, (
            f"chunk {chunk_sec}: crossfade {auto['crossfade_sec'] * 1000}ms != {cf_ms}ms"
        )

    # The resulting margins, which are what the user pays in latency.
    assert sola_margin_ms(0.010, 10.0, "aggressive", DEFAULT_DECODER_OVERLAP_FRAMES) == 40.0
    assert sola_margin_ms(0.020, 10.0, "normal", DEFAULT_DECODER_OVERLAP_FRAMES) == 80.0
    print("PASS: GUI-derived search/crossfade and resulting margins pinned")


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


def test_every_gui_preset_emits_exactly_one_hop() -> None:
    """Run each GUI preset through the real splice and check the length.

    Asserting ``margin >= crossfade + search`` instead was worthless: that
    inequality is true for any non-negative decoder overlap and any ceiling,
    so it held even with the round-up and the overlap term both deleted.
    Feeding real chunks through ``sola_crossfade`` is the property that
    actually matters -- when the margin is starved the fixed-length branch is
    silently abandoned and the emitted length drifts away from one hop.
    """
    output_sr, model_sr = 44100, 48000
    rng = np.random.default_rng(0)
    for mode, chunk_sec in [
        ("aggressive", 0.02),
        ("aggressive", 0.10),
        ("normal", 0.04),
        ("normal", 0.30),
    ]:
        auto = _auto_params(chunk_sec, mode)
        cf = int(output_sr * auto["crossfade_sec"])
        search = int(output_sr * auto["sola_search_ms"] / 1000)
        hop_out = int(round(chunk_sec * output_sr))
        margin_model = _compute_sola_extra_model(
            model_sr, output_sr, cf, search,
            _effective_decoder_overlap_frames(mode, DEFAULT_DECODER_OVERLAP_FRAMES),
        )
        margin_out = int(round(margin_model * output_sr / model_sr))

        state = SolaState(crossfade_samples=cf, search_samples=search)
        n = 40
        signal = (
            np.sin(2 * np.pi * 110 * np.arange((n + 4) * hop_out) / output_sr)
            + 0.3 * rng.standard_normal((n + 4) * hop_out)
        ).astype(np.float32)

        emitted = 0
        first = 1 + -(-margin_out // hop_out)  # first chunk with enough history
        for i in range(first, n):
            end = i * hop_out
            out = sola_crossfade(
                signal[end - hop_out - margin_out:end], state, target_len=hop_out
            )
            assert len(out) == hop_out, (
                f"{mode} {chunk_sec * 1000:.0f}ms chunk {i}: emitted {len(out)}, "
                f"expected one hop ({hop_out}) -- fixed-length branch abandoned"
            )
            emitted += len(out)
        assert emitted == (n - first) * hop_out
    print("PASS: every GUI preset emits exactly one hop through the real splice")


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


def test_resized_returns_a_new_state_without_the_stale_holdback() -> None:
    """Resizing must publish a NEW state and drop a hold-back the change invalidated.

    The buffer is exactly crossfade_samples long; keeping one sized for the
    previous window broadcast-mismatches the new Hann curves and raises
    ValueError on the inference thread.  Returning a new object matters too:
    the inference thread reads this state without a lock, so it must never
    observe a half-updated one.
    """
    original = SolaState(crossfade_samples=441, search_samples=441)
    original.buffer = np.zeros(441, dtype=np.float32)
    original._ensure_window()

    changed = original.resized(882, 441)
    assert changed is not original, "resized mutated in place instead of rebinding"
    assert original.buffer is not None, "resized modified the state it was called on"
    assert changed.buffer is None, "stale hold-back survived a crossfade change"
    assert changed._hann_fade_in is None, "stale Hann window survived"
    assert changed.crossfade_samples == 882

    # Search-only changes carry no state, so the hold-back comes across.
    held = np.zeros(882, dtype=np.float32)
    changed.buffer = held
    same_window = changed.resized(882, 220)
    assert same_window is not changed
    assert same_window.buffer is held, "hold-back dropped for a search-only change"
    assert same_window.search_samples == 220
    print("PASS: resized rebinds, and drops the hold-back only on a crossfade change")


if __name__ == "__main__":
    test_sola_extra_uses_one_search_window()
    test_aggressive_reduces_decoder_overlap()
    test_decoder_overlap_increases_sola_extra()
    test_sola_extra_aligned_at_every_model_rate()
    test_decoder_overlap_default_is_5()
    test_gui_derived_latency_values_are_pinned()
    test_gui_estimate_matches_runtime_margin()
    test_every_gui_preset_emits_exactly_one_hop()
    test_no_margin_when_the_splice_never_runs()
    test_resized_returns_a_new_state_without_the_stale_holdback()
    print("\nAll decoder overlap / SOLA margin tests passed.")
