"""
Tests for the Phase 0 page-turn detector.

Runs headless: synthetic grayscale frames stand in for the camera, so the
capture logic can be exercised without a webcam (or a display, or a 4K
recording). Frames are built as noise fields whose *difference* between
consecutive frames is what the detector measures — a still book is the same
field repeated, a page turn is a fresh one.

Run standalone (`python tests/test_live_state.py`) or under pytest.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from live_state import SETTLED, TURNING, WAITING, LiveDetector, build_spreads  # noqa: E402


H, W = 90, 160
FPS = 30.0
# settle_time * fps = 6 frames of stillness, with a 5-frame smoothing window;
# small enough that a test's frame counts stay readable.
KW = dict(fps=FPS, settle_threshold=2.0, turn_threshold=5.0,
          settle_time=0.2, smoothing_window=5)


def still_frame(seed=0):
    """A fixed field for a given seed — repeating it produces zero motion."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 90, size=(H, W), endpoint=False, dtype=np.int16)
    return (base + 80).clip(0, 255).astype(np.uint8)


def flat_frame():
    """A smooth gradient: almost no high-frequency content, so low sharpness."""
    return np.tile(np.linspace(60, 200, W), (H, 1)).astype(np.uint8)


def detailed_frame():
    """The gradient plus sparse speckle — much sharper, but barely any motion.

    Perturbing 1/16 of the pixels by 20 levels puts the mean absolute
    difference from ``flat_frame`` at ~1.25, under the 2.0 settle threshold, so
    this frame can appear inside a still run without breaking it — while its
    Laplacian variance is orders of magnitude higher. That separation is what
    lets a test assert *which* frame the detector picks.
    """
    f = flat_frame().astype(np.int16)
    f[::4, ::4] += 20
    return f.clip(0, 255).astype(np.uint8)


def run(det, frames, start=0):
    """Feed frames to the detector; return the ticks and the captures made."""
    ticks = [det.update(start + i, f) for i, f in enumerate(frames)]
    return ticks, [t.capture for t in ticks if t.capture is not None]


def steady(seed, n):
    """``n`` identical frames — a book sitting still."""
    return [still_frame(seed)] * n


def churn(n, seed0=1000):
    """``n`` frames that each differ wildly — a page mid-turn."""
    return [still_frame(seed0 + i) for i in range(n)]


# ── Core state machine ───────────────────────────────────────


def test_settles_from_waiting_and_captures_once():
    det = LiveDetector(**KW)
    ticks, caps = run(det, steady(1, 20))
    assert len(caps) == 1, f"expected one initial capture, got {len(caps)}"
    assert caps[0].reason == "initial"
    assert ticks[-1].state == SETTLED
    # The very first frame has no predecessor, so motion is defined as 0 and
    # the settle counter may start immediately; what matters is that it takes
    # at least settle_frames before anything is captured.
    assert caps[0].frame_index >= 0
    assert det.state == SETTLED


def test_page_turn_produces_a_second_capture_and_a_peak():
    det = LiveDetector(**KW)
    run(det, steady(1, 20))                      # settle -> "initial"
    _, mid = run(det, churn(10), start=20)       # page turn
    assert mid == [], "no capture should happen mid-turn"
    assert det.state == TURNING
    assert len(det.turn_frames) == 1

    _, caps = run(det, steady(2, 20), start=30)  # settle again
    assert len(caps) == 1
    assert caps[0].reason == "settle"
    assert det.state == SETTLED


def test_stillness_alone_does_not_recapture():
    """A settled book left untouched must not emit a stream of duplicates."""
    det = LiveDetector(**KW)
    _, caps = run(det, steady(1, 200))
    assert len(caps) == 1


def test_turn_without_settling_captures_nothing_more():
    det = LiveDetector(**KW)
    run(det, steady(1, 20))
    _, caps = run(det, churn(60), start=20)
    assert caps == []
    assert det.state == TURNING


def test_three_turns_give_four_captures():
    det = LiveDetector(**KW)
    frames = steady(0, 20)
    for i in range(3):
        frames += churn(10, seed0=1000 + 100 * i) + steady(10 + i, 20)
    _, caps = run(det, frames)
    assert len(caps) == 4, [c.reason for c in caps]
    assert [c.reason for c in caps] == ["initial", "settle", "settle", "settle"]
    assert len(det.turn_frames) == 3


# ── Frame selection ──────────────────────────────────────────


def test_capture_picks_the_sharpest_frame_in_the_window():
    """The chosen frame is the sharpest of the settle window, not the last one.

    This is the behaviour that matters in practice: the camera is still
    refocusing as the operator's hand leaves, so the frame that trips the
    settle threshold is usually not the best one available.
    """
    det = LiveDetector(**KW)
    # settle_frames is 6, so the capture commits on frame 5 with frames 0-5 in
    # the window. Frame 3 is the sharp one; frames 4 and 5 follow it and must
    # still lose.
    frames = [flat_frame()] * 12
    frames[3] = detailed_frame()
    ticks, caps = run(det, frames)
    assert caps, "expected a capture"
    assert caps[0].frame_index == 3, f"picked {caps[0].frame_index}, wanted 3"
    assert ticks[3].sharpness > 10 * ticks[0].sharpness


def test_capture_index_stays_inside_the_resolvable_window():
    """A caller buffering settle_frames frames can always resolve the choice.

    p0_live_capture keeps exactly ``settle_frames`` full-res frames; if the
    detector could name anything older, the capture would silently drop.
    """
    det = LiveDetector(**KW)
    ticks, caps = run(det, steady(1, 20))
    committed_at = next(t.frame_index for t in ticks if t.capture is not None)
    assert committed_at - caps[0].frame_index < det.settle_frames


# ── Operator controls ────────────────────────────────────────


def test_pause_suppresses_auto_capture():
    det = LiveDetector(**KW)
    det.set_paused(True)
    _, caps = run(det, steady(1, 40))
    assert caps == []
    assert det.state == WAITING

    det.set_paused(False)
    _, caps = run(det, steady(1, 40), start=40)
    assert len(caps) == 1


def test_force_capture_marks_the_book_settled():
    """A manual capture must not be followed by a duplicate when it settles."""
    det = LiveDetector(**KW)
    run(det, steady(1, 3))          # too few frames to auto-capture yet
    forced = det.force_capture()
    assert forced is not None and forced.reason == "manual"
    assert det.state == SETTLED

    _, caps = run(det, steady(1, 40), start=3)
    assert caps == [], "settling after a manual capture should not re-capture"


def test_force_capture_before_any_frame_is_a_no_op():
    assert LiveDetector(**KW).force_capture() is None


# ── Artifacts ────────────────────────────────────────────────


def test_signals_have_one_value_per_frame():
    det = LiveDetector(**KW)
    run(det, steady(1, 25))
    assert det.motion_signal().shape == (25,)
    assert det.smoothed_signal().shape == (25,)
    assert det.motion_signal()[0] == 0.0     # no predecessor for frame 0


def test_empty_detector_emits_empty_artifacts():
    det = LiveDetector(**KW)
    assert det.motion_signal().shape == (0,)
    assert det.smoothed_signal().shape == (0,)
    assert det.peaks().shape == (0,)
    # No turns at all is still one spread covering the whole recording.
    assert len(det.spreads(0)) == 1


def test_spreads_partition_the_recording_without_gaps():
    spreads = build_spreads(np.array([50, 120]), 200, FPS)
    assert [(s["start_frame"], s["end_frame"]) for s in spreads] == \
        [(0, 50), (50, 120), (120, 200)]
    assert [s["spread_index"] for s in spreads] == [1, 2, 3]
    assert sum(s["frame_count"] for s in spreads) == 200


def test_peaks_are_sorted_int64():
    det = LiveDetector(**KW)
    frames = steady(0, 20)
    for i in range(2):
        frames += churn(10, seed0=1000 + 100 * i) + steady(10 + i, 20)
    run(det, frames)
    peaks = det.peaks()
    assert peaks.dtype == np.int64
    assert list(peaks) == sorted(peaks)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
