"""Drift budget (utils.drift_due) — the Phase-4 re-anchor reminder.

Tracking error grows with the horizon since a box was last anchored and
never self-corrects, so the watchdog asks for a re-anchor every
DRIFT_HORIZON_LIMIT keyframes. These pin the cadence: a reminder must not
fire on the anchor itself, must fire once per budget rather than on every
frame past it, and must count a consensus box's horizon from the session
start (it was voted, not placed on a frame).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from utils import (DRIFT_HORIZON_LIMIT, DRIFT_SHIFT_FRAC,  # noqa: E402
                   drift_due, shift_due)

L = DRIFT_HORIZON_LIMIT
S = DRIFT_SHIFT_FRAC
passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok    {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")


def test_anchor_frame_never_reminds():
    check("test_anchor_frame_never_reminds", not drift_due(7, 7))


def test_fires_exactly_once_per_budget():
    fires = [i for i in range(200) if drift_due(i, 10)]
    expected = [i for i in range(10 + L, 200, L)]
    check("test_fires_exactly_once_per_budget",
          fires == expected and len(fires) >= 2)


def test_quiet_inside_the_budget():
    quiet = [i for i in range(11, 10 + L) if drift_due(i, 10)]
    check("test_quiet_inside_the_budget", quiet == [])


def test_consensus_counts_from_session_start():
    # No anchor frame owns a consensus box, so its horizon is the index.
    check("test_consensus_counts_from_session_start",
          drift_due(L, None) and not drift_due(L - 1, None))


def test_a_correction_resets_the_budget():
    # Re-anchoring at 30 must push the next reminder to 30 + L, not leave
    # the old anchor's schedule in place.
    check("test_a_correction_resets_the_budget",
          drift_due(30 + L, 30) and not drift_due(10 + L, 30))


def test_limit_is_overridable():
    check("test_limit_is_overridable",
          drift_due(15, 10, limit=5) and not drift_due(15, 10, limit=7))


def test_shift_quiet_inside_its_budget():
    check("test_shift_quiet_inside_its_budget", not shift_due(S * 0.9, 0.0))


def test_shift_fires_once_the_budget_is_spent():
    check("test_shift_fires_once_the_budget_is_spent", shift_due(S, 0.0))


def test_shift_measures_from_the_mark_not_zero():
    # After reminding at S, drifting to 1.5*S must stay quiet: only another
    # full budget's travel earns a second reminder.
    check("test_shift_measures_from_the_mark_not_zero",
          not shift_due(1.5 * S, S) and shift_due(2 * S, S))


def test_a_fast_rig_reminds_before_the_horizon():
    # The case the frame count misses: 1.6% of travel inside 9 keyframes.
    fast = [shift_due(0.016, 0.0), drift_due(9, 0)]
    check("test_a_fast_rig_reminds_before_the_horizon",
          fast == [True, False])


def test_a_still_rig_is_not_nagged_by_travel():
    # A box that never moves must rely on the horizon alone.
    check("test_a_still_rig_is_not_nagged_by_travel",
          not shift_due(0.0, 0.0) and drift_due(L, 0))


if __name__ == "__main__":
    print("test_drift_budget.py")
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(1 if failed else 0)
