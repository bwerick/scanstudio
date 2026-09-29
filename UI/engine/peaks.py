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

from .plot_theme import use_theme, style_axes


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
                spreads: list[dict], fps: float, output_path: str | Path,
                themes: tuple = ("dark", "light")):
    """
    Peak detection diagnostic: the signal with detected turns, a detail
    window with spreads shaded, and scan pacing — which is where problems
    actually surface (short spreads are corrections, long ones missed turns).
    """
    output_path = Path(output_path)
    times = np.arange(len(smoothed)) / fps
    durs = np.array([sp["duration_sec"] for sp in spreads])
    idx = np.arange(1, len(durs) + 1)
    pk = peaks[peaks < len(smoothed)]

    span = min(30.0, times[-1])
    z0 = min(60.0, max(0.0, times[-1] - span))
    z1 = z0 + span
    m = (times >= z0) & (times <= z1)

    for theme in themes:
        C = use_theme(theme)
        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 1],
                               hspace=0.62, wspace=0.28)

        # ── full signal ──
        ax = fig.add_subplot(gs[0, :])
        ax.fill_between(times, smoothed, color=C["accent"], alpha=0.15, lw=0, zorder=1)
        ax.plot(times, smoothed, color=C["accent"], lw=0.55, zorder=2)
        ax.plot(pk / fps, smoothed[pk], "v", ms=3, color=C["pink"], mec="none", zorder=3)
        ax.axvspan(z0, z1, color=C["muted"], alpha=0.13, lw=0, zorder=0)
        style_axes(ax, "Page Turn Detection",
                   f"{len(pk)} turns · {len(spreads)} spreads · "
                   f"median {np.median(durs):.1f}s each  ·  shaded band shown below",
                   "Time (seconds)", "Mean pixel difference")
        ax.set_xlim(0, times[-1]); ax.set_ylim(0, None)

        # ── detail window, spreads shaded ──
        ax2 = fig.add_subplot(gs[1, :])
        if m.sum() > 2:
            b = [0] + list(pk) + [len(smoothed)]
            for i in range(len(b) - 1):
                if i % 2 == 0:
                    ax2.axvspan(b[i] / fps, b[i + 1] / fps,
                                color=C["panel"], lw=0, zorder=0)
            ax2.fill_between(times[m], smoothed[m], color=C["accent"],
                             alpha=0.2, lw=0, zorder=1)
            ax2.plot(times[m], smoothed[m], color=C["accent"], lw=1.3, zorder=2)
            pm = pk[(pk / fps >= z0) & (pk / fps <= z1)]
            ax2.plot(pm / fps, smoothed[pm], "v", ms=6, color=C["pink"],
                     mec="none", zorder=3)
            ax2.set_xlim(z0, z1)
        ax2.set_ylim(0, None)
        style_axes(ax2, f"Detail — {z0:.0f}s to {z1:.0f}s",
                   "alternating shading marks each detected spread",
                   "Time (seconds)", "Mean pixel difference")

        # ── pacing ──
        ax3 = fig.add_subplot(gs[2, :2])
        ax3.scatter(idx, durs, s=14, color=C["accent"], alpha=0.7, ec="none", zorder=3)
        if len(durs) >= 5:
            w = max(3, min(15, len(durs) // 6))
            roll = np.array([np.median(durs[max(0, i - w // 2):i + w // 2 + 1])
                             for i in range(len(durs))])
            ax3.plot(idx, roll, color=C["accent_2"], lw=2, zorder=4,
                     label="rolling median")
        ax3.axhline(np.median(durs), color=C["muted"], ls="--", lw=1, zorder=2)
        short, long_ = durs < 1.5, durs > 4.0
        if short.any():
            ax3.scatter(idx[short], durs[short], s=38, facecolor="none",
                        ec=C["yellow"], lw=1.3, zorder=5,
                        label=f"{short.sum()} short — likely correction")
        if long_.any():
            ax3.scatter(idx[long_], durs[long_], s=38, facecolor="none",
                        ec=C["red"], lw=1.3, zorder=5,
                        label=f"{long_.sum()} long — possible missed turn")
        style_axes(ax3, "Scan Pacing", "outliers are where problems cluster",
                   "Spread number", "Duration (s)")
        ax3.set_xlim(0, len(durs) + 1)
        ax3.legend(loc="upper right", fontsize=8.5)

        # ── duration distribution ──
        ax4 = fig.add_subplot(gs[2, 2])
        ax4.hist(durs, bins=min(30, max(6, len(durs) // 4)),
                 color=C["purple"], alpha=0.8, ec="none", zorder=3)
        ax4.axvline(np.median(durs), color=C["muted"], ls="--", lw=1.2, zorder=4)
        style_axes(ax4, "Duration Spread", "", "Seconds", "Count")

        suffix = "" if theme == "dark" else "_light"
        plt.savefig(str(output_path.with_name(
            output_path.stem + suffix + output_path.suffix)))
        plt.close()
