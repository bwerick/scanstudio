#!/usr/bin/env python3
"""
Phase 2: Detect Page Turn Peaks

Usage:
  python scripts/p2_detect_peaks.py output/mybook
  python scripts/p2_detect_peaks.py output/mybook --peak-height 4.5
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

from utils import log, ProjectPaths, check_overwrite


DEFAULT_PEAK_HEIGHT = 5.0
DEFAULT_PEAK_DISTANCE_SEC = 1.5
DEFAULT_PEAK_PROMINENCE = 3.0
DEFAULT_LONG_SPREAD_SEC = 3.0
DEFAULT_VALLEY_MOTION = 2.0
DEFAULT_VALLEY_MIN_SEC = 0.15
DEFAULT_VALLEY_PEAK_THRESHOLD = 4.0


def build_spread_list(peaks, total_len):
    boundaries = [(0, int(peaks[0]))]
    for i in range(len(peaks) - 1):
        boundaries.append((int(peaks[i]), int(peaks[i + 1])))
    boundaries.append((int(peaks[-1]), total_len))
    return [{"spread_index": i+1, "start_frame": s, "end_frame": e, "frame_count": e-s}
            for i, (s, e) in enumerate(boundaries)]


def rescue_missed_turns(smoothed, fps, peaks, long_sec, valley_motion, valley_min_sec, valley_peak):
    spreads = build_spread_list(peaks, len(smoothed))
    additional = []
    for i, sp in enumerate(spreads):
        s, e = sp["start_frame"], sp["end_frame"]
        if (e - s) / fps < long_sec:
            continue
        region = smoothed[s:e]
        below = region < valley_motion
        valleys = []
        in_v, vs = False, 0
        for k in range(len(below)):
            if below[k] and not in_v:
                vs, in_v = k, True
            elif not below[k] and in_v:
                if (k - vs) >= int(valley_min_sec * fps):
                    valleys.append((vs, k - 1))
                in_v = False
        if in_v and (len(below) - vs) >= int(valley_min_sec * fps):
            valleys.append((vs, len(below) - 1))
        if len(valleys) >= 2:
            for v in range(len(valleys) - 1):
                gs, ge = valleys[v][1], valleys[v + 1][0]
                if ge > gs:
                    gm = np.max(region[gs:ge])
                    if gm >= valley_peak:
                        sf = s + gs + np.argmax(region[gs:ge])
                        if min(abs(sf - p) for p in peaks) > int(0.8 * fps):
                            additional.append(sf)
                            log(f"  Rescued: spread {i+1} at {sf/fps:.1f}s (peak={gm:.1f})")
    if additional:
        all_peaks = np.sort(np.concatenate([peaks, np.array(additional)]))
        log(f"Pass 2: {len(additional)} rescued → {len(all_peaks)} total peaks")
        return all_peaks
    log("Pass 2: No missed turns")
    return peaks


def generate_plot(smoothed, peaks, spreads, fps, output_path):
    times = np.arange(len(smoothed)) / fps
    fig, axes = plt.subplots(3, 1, figsize=(22, 14))

    ax = axes[0]
    ax.plot(times, smoothed, linewidth=0.4, color="steelblue")
    ax.plot(peaks / fps, smoothed[peaks], "rv", markersize=5, label=f"Page turns ({len(peaks)})")
    ax.set_title(f"Peak Detection — {len(peaks)} turns → {len(spreads)} spreads")
    ax.legend()

    ax = axes[1]
    mask = times < 120
    ax.plot(times[mask], smoothed[mask], linewidth=0.7, color="steelblue")
    pm = peaks[peaks / fps < 120]
    ax.plot(pm / fps, smoothed[pm], "rv", markersize=7)
    for i, sp in enumerate(spreads):
        ts, te = sp["start_frame"] / fps, sp["end_frame"] / fps
        if ts > 120: break
        if i % 2 == 0: ax.axvspan(ts, min(te, 120), alpha=0.08, color="green")
    ax.set_title("First 120s — Spreads shaded")

    ax = axes[2]
    durs = [sp["duration_sec"] for sp in spreads]
    ax.hist(durs, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(x=np.median(durs), color="red", linestyle="--", label=f"Median: {np.median(durs):.2f}s")
    ax.set_title("Spread Duration Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Detect page turn peaks")
    parser.add_argument("output_dir", help="Base output directory (e.g. output/mybook)")
    parser.add_argument("--peak-height", type=float, default=DEFAULT_PEAK_HEIGHT)
    parser.add_argument("--min-distance", type=float, default=DEFAULT_PEAK_DISTANCE_SEC)
    parser.add_argument("--prominence", type=float, default=DEFAULT_PEAK_PROMINENCE)
    parser.add_argument("--long-spread", type=float, default=DEFAULT_LONG_SPREAD_SEC)
    parser.add_argument("--valley-motion", type=float, default=DEFAULT_VALLEY_MOTION)
    parser.add_argument("--valley-min-sec", type=float, default=DEFAULT_VALLEY_MIN_SEC)
    parser.add_argument("--valley-peak", type=float, default=DEFAULT_VALLEY_PEAK_THRESHOLD)
    args = parser.parse_args()

    log("=" * 60)
    log("PHASE 2: Detect Page Turn Peaks")
    log("=" * 60)

    paths = ProjectPaths(args.output_dir)

    for p in [paths.data / "smoothed_signal.npy", paths.json / "metadata.json"]:
        if not p.exists():
            log(f"ERROR: {p} not found. Run Phase 1 first.")
            sys.exit(1)

    out_peaks = paths.data / "peaks.npy"
    if out_peaks.exists():
        if not check_overwrite(out_peaks):
            log("Skipped.")
            return

    smoothed = np.load(str(paths.data / "smoothed_signal.npy"))
    metadata = json.loads((paths.json / "metadata.json").read_text())
    fps = metadata["fps"]
    log(f"  {len(smoothed)} values, {fps} fps")

    peaks, _ = find_peaks(smoothed, height=args.peak_height,
                           distance=int(args.min_distance * fps), prominence=args.prominence)
    log(f"Pass 1: {len(peaks)} peaks")

    peaks = rescue_missed_turns(smoothed, fps, peaks, args.long_spread,
                                 args.valley_motion, args.valley_min_sec, args.valley_peak)

    spreads = build_spread_list(peaks, len(smoothed))
    for sp in spreads:
        sp["duration_sec"] = round(sp["frame_count"] / fps, 3)
        sp["start_time"] = round(sp["start_frame"] / fps, 2)
        sp["end_time"] = round(sp["end_frame"] / fps, 2)

    np.save(str(out_peaks), peaks)
    (paths.json / "spreads.json").write_text(json.dumps(spreads, indent=2))

    peaks_meta = {"fps": fps, "parameters": {
        "peak_height": args.peak_height, "min_distance_sec": args.min_distance,
        "prominence": args.prominence, "long_spread_sec": args.long_spread,
    }, "results": {"total_peaks": int(len(peaks)), "total_spreads": len(spreads)}}
    (paths.json / "peaks_metadata.json").write_text(json.dumps(peaks_meta, indent=2))

    generate_plot(smoothed, peaks, spreads, fps, str(paths.plots / "peaks_plot.png"))

    durs = [sp["duration_sec"] for sp in spreads]
    log(f"Segmentation: {len(spreads)} spreads, median={np.median(durs):.2f}s")
    log("PHASE 2 COMPLETE")


if __name__ == "__main__":
    main()
