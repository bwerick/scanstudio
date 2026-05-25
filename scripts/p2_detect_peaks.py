#!/usr/bin/env python3
"""
Phase 2: Detect Page Turn Peaks

Loads the motion signal from Phase 1, detects page turn peaks, rescues missed
turns in long spreads via valley analysis, and segments the video into spreads.

Usage:
  python scripts/02_detect_peaks.py output/audiq5
  python scripts/02_detect_peaks.py output/audiq5 --peak-height 4.5
  python scripts/02_detect_peaks.py output/audiq5 --min-distance 1.2 --valley-threshold 4.0

Inputs (from Phase 1):
  - output/<name>/motion/smoothed_signal.npy
  - output/<name>/motion/motion_signal.npy
  - output/<name>/motion/metadata.json

Outputs (in output/<name>/peaks/):
  - peaks.npy              Frame indices of detected page turns
  - spreads.json           List of spread boundaries with metadata
  - peaks_metadata.json    Parameters used and detection stats
  - peaks_plot.png         Motion signal with peaks + spread duration histogram

Requirements:
  pip install numpy scipy matplotlib
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import log, ensure_dir, check_overwrite


# ── Tuned defaults from ground truth analysis ────────────────────────────
# All ideal page turns had motion >= 6.26, so 5.0 is a safe threshold.
# Good frames had median motion 0.92, so valleys below 2.0 are stable.
DEFAULT_PEAK_HEIGHT = 5.0
DEFAULT_PEAK_DISTANCE_SEC = 1.5
DEFAULT_PEAK_PROMINENCE = 3.0
DEFAULT_LONG_SPREAD_SEC = 3.0
DEFAULT_VALLEY_MOTION = 2.0
DEFAULT_VALLEY_MIN_SEC = 0.15
DEFAULT_VALLEY_PEAK_THRESHOLD = 4.0


def detect_main_peaks(
    smoothed: np.ndarray,
    fps: float,
    height: float,
    distance_sec: float,
    prominence: float,
) -> np.ndarray:
    """Pass 1: Detect main page turn peaks using scipy find_peaks."""
    peaks, _ = find_peaks(
        smoothed,
        height=height,
        distance=int(distance_sec * fps),
        prominence=prominence,
    )
    log(f"Pass 1: {len(peaks)} page turns detected")
    return peaks


def rescue_missed_turns(
    smoothed: np.ndarray,
    fps: float,
    peaks: np.ndarray,
    long_spread_sec: float,
    valley_motion: float,
    valley_min_sec: float,
    valley_peak_threshold: float,
) -> np.ndarray:
    """
    Pass 2: Check long spreads for missed page turns via valley analysis.

    A missed turn shows up as two stable valleys separated by a motion peak
    within a single spread that's unusually long.
    """
    spreads = build_spread_list(peaks, len(smoothed))
    additional = []

    for i, sp in enumerate(spreads):
        s, e = sp["start_frame"], sp["end_frame"]
        dur = (e - s) / fps
        if dur < long_spread_sec:
            continue

        region = smoothed[s:e]

        # Find valleys: contiguous regions below valley_motion threshold
        below = region < valley_motion
        valleys = []
        in_v = False
        vs = 0
        for k in range(len(below)):
            if below[k] and not in_v:
                vs = k
                in_v = True
            elif not below[k] and in_v:
                if (k - vs) >= int(valley_min_sec * fps):
                    valleys.append((vs, k - 1))
                in_v = False
        if in_v and (len(below) - vs) >= int(valley_min_sec * fps):
            valleys.append((vs, len(below) - 1))

        # Check for two valleys separated by significant motion
        if len(valleys) >= 2:
            for v in range(len(valleys) - 1):
                gap_start = valleys[v][1]
                gap_end = valleys[v + 1][0]
                if gap_end > gap_start:
                    gap_motion = np.max(region[gap_start:gap_end])
                    if gap_motion >= valley_peak_threshold:
                        split_offset = gap_start + np.argmax(region[gap_start:gap_end])
                        split_frame = s + split_offset
                        min_dist = min(abs(split_frame - p) for p in peaks)
                        if min_dist > int(0.8 * fps):
                            additional.append(split_frame)
                            log(
                                f"  Rescued: spread {i+1} at {split_frame/fps:.1f}s "
                                f"(dur={dur:.1f}s, peak={gap_motion:.1f})"
                            )

    if additional:
        all_peaks = np.sort(np.concatenate([peaks, np.array(additional)]))
        log(
            f"Pass 2: {len(additional)} missed turns rescued → {len(all_peaks)} total peaks"
        )
        return all_peaks
    else:
        log(f"Pass 2: No missed turns found")
        return peaks


def build_spread_list(peaks: np.ndarray, total_len: int) -> list[dict]:
    """Build a list of spread dicts from peak positions."""
    boundaries = []
    # First spread: start to first peak
    boundaries.append((0, int(peaks[0])))
    # Middle spreads
    for i in range(len(peaks) - 1):
        boundaries.append((int(peaks[i]), int(peaks[i + 1])))
    # Last spread: last peak to end
    boundaries.append((int(peaks[-1]), total_len))

    spreads = []
    for idx, (s, e) in enumerate(boundaries):
        spreads.append(
            {
                "spread_index": idx + 1,
                "start_frame": s,
                "end_frame": e,
                "frame_count": e - s,
            }
        )
    return spreads


def generate_plot(
    smoothed: np.ndarray,
    peaks: np.ndarray,
    spreads: list[dict],
    fps: float,
    output_path: str,
):
    """Generate diagnostic plot with peaks marked and spread duration histogram."""
    times = np.arange(len(smoothed)) / fps

    fig, axes = plt.subplots(3, 1, figsize=(22, 14))

    # Full signal with peaks
    ax = axes[0]
    ax.plot(times, smoothed, linewidth=0.4, color="steelblue", label="Smoothed motion")
    ax.plot(
        peaks / fps,
        smoothed[peaks],
        "rv",
        markersize=5,
        label=f"Page turns ({len(peaks)})",
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Motion")
    ax.set_title(f"Peak Detection — {len(peaks)} page turns → {len(spreads)} spreads")
    ax.legend(loc="upper right")

    # First 120s zoomed
    ax = axes[1]
    mask = times < 120
    ax.plot(times[mask], smoothed[mask], linewidth=0.7, color="steelblue")
    peak_mask = peaks[peaks / fps < 120]
    ax.plot(peak_mask / fps, smoothed[peak_mask], "rv", markersize=7)
    # Shade spreads alternating
    for i, sp in enumerate(spreads):
        ts = sp["start_frame"] / fps
        te = sp["end_frame"] / fps
        if ts > 120:
            break
        if i % 2 == 0:
            ax.axvspan(ts, min(te, 120), alpha=0.08, color="green")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Motion")
    ax.set_title("First 120 Seconds — Spreads alternately shaded")

    # Spread duration histogram
    ax = axes[2]
    durations = [sp["duration_sec"] for sp in spreads]
    ax.hist(durations, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(
        x=np.median(durations),
        color="red",
        linestyle="--",
        label=f"Median: {np.median(durations):.2f}s",
    )
    ax.set_xlabel("Spread duration (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Spread Durations")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    log(f"  Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Detect page turn peaks from motion signal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("output_dir", help="Base output directory (e.g. output/audiq5)")
    parser.add_argument(
        "--peak-height",
        type=float,
        default=DEFAULT_PEAK_HEIGHT,
        help=f"Min peak height (default: {DEFAULT_PEAK_HEIGHT})",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=DEFAULT_PEAK_DISTANCE_SEC,
        help=f"Min seconds between peaks (default: {DEFAULT_PEAK_DISTANCE_SEC})",
    )
    parser.add_argument(
        "--prominence",
        type=float,
        default=DEFAULT_PEAK_PROMINENCE,
        help=f"Min peak prominence (default: {DEFAULT_PEAK_PROMINENCE})",
    )
    parser.add_argument(
        "--long-spread",
        type=float,
        default=DEFAULT_LONG_SPREAD_SEC,
        help=f"Spreads longer than this are checked for missed turns (default: {DEFAULT_LONG_SPREAD_SEC})",
    )
    parser.add_argument(
        "--valley-motion",
        type=float,
        default=DEFAULT_VALLEY_MOTION,
        help=f"Motion below this = stable valley (default: {DEFAULT_VALLEY_MOTION})",
    )
    parser.add_argument(
        "--valley-min-sec",
        type=float,
        default=DEFAULT_VALLEY_MIN_SEC,
        help=f"Min valley duration in seconds (default: {DEFAULT_VALLEY_MIN_SEC})",
    )
    parser.add_argument(
        "--valley-peak",
        type=float,
        default=DEFAULT_VALLEY_PEAK_THRESHOLD,
        help=f"Min motion between valleys to confirm missed turn (default: {DEFAULT_VALLEY_PEAK_THRESHOLD})",
    )
    args = parser.parse_args()

    log("=" * 60)
    log("PHASE 2: Detect Page Turn Peaks")
    log("=" * 60)

    base = Path(args.output_dir)
    motion_dir = base / "motion"
    peaks_dir = ensure_dir(base / "peaks")

    # Verify Phase 1 outputs exist
    smoothed_path = motion_dir / "smoothed_signal.npy"
    raw_path = motion_dir / "motion_signal.npy"
    meta_path = motion_dir / "metadata.json"

    for p in [smoothed_path, raw_path, meta_path]:
        if not p.exists():
            log(f"ERROR: Phase 1 output not found: {p}")
            log(f"Run Phase 1 first: python scripts/01_motion_signal.py <video>")
            sys.exit(1)

    # Check overwrite
    out_peaks = peaks_dir / "peaks.npy"
    if out_peaks.exists():
        if not check_overwrite(out_peaks):
            log("Skipped. Existing peaks preserved.")
            return

    # Load
    log("")
    log("Loading Phase 1 outputs...")
    smoothed = np.load(str(smoothed_path))
    metadata = json.loads(meta_path.read_text())
    fps = metadata["fps"]
    log(f"  Smoothed signal: {len(smoothed)} values, {fps} fps")

    # Pass 1: main peak detection
    log("")
    log(
        f"Pass 1: find_peaks(height={args.peak_height}, distance={args.min_distance}s, "
        f"prominence={args.prominence})"
    )
    peaks = detect_main_peaks(
        smoothed, fps, args.peak_height, args.min_distance, args.prominence
    )

    # Pass 2: valley rescue
    log("")
    log(
        f"Pass 2: Valley rescue (long_spread>{args.long_spread}s, "
        f"valley_motion<{args.valley_motion}, valley_peak>{args.valley_peak})"
    )
    peaks = rescue_missed_turns(
        smoothed,
        fps,
        peaks,
        long_spread_sec=args.long_spread,
        valley_motion=args.valley_motion,
        valley_min_sec=args.valley_min_sec,
        valley_peak_threshold=args.valley_peak,
    )

    # Build spreads
    log("")
    spreads = build_spread_list(peaks, len(smoothed))
    # Add duration and time fields
    for sp in spreads:
        sp["duration_sec"] = round(sp["frame_count"] / fps, 3)
        sp["start_time"] = round(sp["start_frame"] / fps, 2)
        sp["end_time"] = round(sp["end_frame"] / fps, 2)

    durations = [sp["duration_sec"] for sp in spreads]
    long_count = sum(1 for d in durations if d > args.long_spread)

    log(f"Segmentation: {len(spreads)} spreads")
    log(f"  Duration range: {min(durations):.2f}s — {max(durations):.2f}s")
    log(f"  Median: {np.median(durations):.2f}s")
    log(f"  Long (>{args.long_spread}s): {long_count}")

    # Save
    log("")
    log("Saving outputs...")

    np.save(str(out_peaks), peaks)
    log(f"  Peaks: {out_peaks} ({len(peaks)} peaks)")

    out_spreads = peaks_dir / "spreads.json"
    out_spreads.write_text(json.dumps(spreads, indent=2))
    log(f"  Spreads: {out_spreads} ({len(spreads)} spreads)")

    peaks_meta = {
        "source_motion_dir": str(motion_dir),
        "fps": fps,
        "parameters": {
            "peak_height": args.peak_height,
            "min_distance_sec": args.min_distance,
            "prominence": args.prominence,
            "long_spread_sec": args.long_spread,
            "valley_motion": args.valley_motion,
            "valley_min_sec": args.valley_min_sec,
            "valley_peak_threshold": args.valley_peak,
        },
        "results": {
            "total_peaks": int(len(peaks)),
            "total_spreads": len(spreads),
            "long_spreads": long_count,
            "duration_min": round(min(durations), 3),
            "duration_max": round(max(durations), 3),
            "duration_median": round(float(np.median(durations)), 3),
        },
    }
    out_meta = peaks_dir / "peaks_metadata.json"
    out_meta.write_text(json.dumps(peaks_meta, indent=2))
    log(f"  Metadata: {out_meta}")

    # Plot
    log("")
    log("Generating plot...")
    out_plot = peaks_dir / "peaks_plot.png"
    generate_plot(smoothed, peaks, spreads, fps, str(out_plot))

    # Done
    log("")
    log("=" * 60)
    log("PHASE 2 COMPLETE")
    log(f"  Peaks:    {out_peaks} ({len(peaks)} page turns)")
    log(f"  Spreads:  {out_spreads} ({len(spreads)} spreads)")
    log(f"  Metadata: {out_meta}")
    log(f"  Plot:     {out_plot}")
    log("=" * 60)


if __name__ == "__main__":
    main()
