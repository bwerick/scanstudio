"""
Peak detection and spread segmentation.

Finds page turn peaks in the motion signal and segments
the video into spreads (regions between page turns).
"""

import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def detect_peaks(smoothed: np.ndarray, fps: float,
                  height: float = 5.0, distance_sec: float = 1.5,
                  prominence: float = 3.0) -> np.ndarray:
    """Find page turn peaks in the smoothed motion signal."""
    peaks, _ = find_peaks(
        smoothed,
        height=height,
        distance=int(distance_sec * fps),
        prominence=prominence,
    )
    return peaks


def rescue_missed_turns(smoothed: np.ndarray, fps: float, peaks: np.ndarray,
                         long_spread_sec: float = 3.0, valley_motion: float = 2.0,
                         valley_min_sec: float = 0.15,
                         valley_peak_threshold: float = 4.0) -> np.ndarray:
    """Check long spreads for missed page turns via valley analysis."""
    spreads = build_spreads(peaks, len(smoothed), fps)
    additional = []

    for sp in spreads:
        s, e = sp["start_frame"], sp["end_frame"]
        if (e - s) / fps < long_spread_sec:
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
                    if gm >= valley_peak_threshold:
                        sf = s + gs + np.argmax(region[gs:ge])
                        if min(abs(sf - p) for p in peaks) > int(0.8 * fps):
                            additional.append(sf)

    if additional:
        return np.sort(np.concatenate([peaks, np.array(additional)]))
    return peaks


def build_spreads(peaks: np.ndarray, signal_length: int,
                   fps: float) -> list[dict]:
    """Build spread list from peak positions."""
    boundaries = [(0, int(peaks[0]))]
    for i in range(len(peaks) - 1):
        boundaries.append((int(peaks[i]), int(peaks[i + 1])))
    boundaries.append((int(peaks[-1]), signal_length))

    spreads = []
    for idx, (s, e) in enumerate(boundaries):
        spreads.append({
            "spread_index": idx + 1,
            "start_frame": s,
            "end_frame": e,
            "frame_count": e - s,
            "duration_sec": round((e - s) / fps, 3),
            "start_time": round(s / fps, 2),
            "end_time": round(e / fps, 2),
        })
    return spreads


def plot_peaks(smoothed: np.ndarray, peaks: np.ndarray,
                spreads: list[dict], fps: float, output_path: str | Path):
    """Generate peak detection diagnostic plot."""
    times = np.arange(len(smoothed)) / fps

    fig, axes = plt.subplots(3, 1, figsize=(22, 14))

    ax = axes[0]
    ax.plot(times, smoothed, linewidth=0.4, color="steelblue")
    ax.plot(peaks / fps, smoothed[peaks], "rv", markersize=5,
            label=f"Page turns ({len(peaks)})")
    ax.set_title(f"Peak Detection — {len(peaks)} turns → {len(spreads)} spreads")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean Pixel Difference")
    ax.legend()

    ax = axes[1]
    mask = times < 120
    ax.plot(times[mask], smoothed[mask], linewidth=0.7, color="steelblue")
    pm = peaks[peaks / fps < 120]
    ax.plot(pm / fps, smoothed[pm], "rv", markersize=7)
    for i, sp in enumerate(spreads):
        ts = sp["start_frame"] / fps
        te = sp["end_frame"] / fps
        if ts > 120:
            break
        if i % 2 == 0:
            ax.axvspan(ts, min(te, 120), alpha=0.08, color="green")
    ax.set_title("First 120s — Spreads shaded")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean Pixel Difference")

    ax = axes[2]
    durs = [sp["duration_sec"] for sp in spreads]
    ax.hist(durs, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(x=np.median(durs), color="red", linestyle="--",
               label=f"Median: {np.median(durs):.2f}s")
    ax.set_title("Spread Duration Distribution")
    ax.set_xlabel("Spread Duration (s)")
    ax.set_ylabel("Count")
    ax.legend()

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
