"""
Motion signal computation.

Computes frame-to-frame mean absolute pixel difference and generates
a smoothed signal for peak detection.
"""

import time
import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from .plot_theme import use_theme, style_axes


def compute_motion_signal(video_path: str, analysis_height: int = 360,
                           on_progress=None) -> tuple[np.ndarray, dict]:
    """
    Compute frame-to-frame mean absolute pixel difference.

    Args:
        video_path: path to video file
        analysis_height: downscale height for computation
        on_progress: optional callback(frame_idx, total_frames) for progress updates

    Returns:
        (diffs, metadata) where diffs is 1D array of diff values
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = analysis_height / orig_h
    aw = int(orig_w * scale)

    metadata = {
        "video_path": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": total_frames / fps,
        "original_width": orig_w,
        "original_height": orig_h,
        "analysis_width": aw,
        "analysis_height": analysis_height,
    }

    diffs = []
    prev_gray = None
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (aw, analysis_height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diffs.append(float(np.mean(cv2.absdiff(prev_gray, gray))))
        prev_gray = gray
        frame_idx += 1
        if on_progress and frame_idx % 500 == 0:
            on_progress(frame_idx, total_frames)

    cap.release()
    elapsed = time.time() - t0
    metadata["frames_processed"] = frame_idx
    metadata["processing_time_sec"] = round(elapsed, 1)

    return np.array(diffs), metadata


def smooth_signal(diffs: np.ndarray, window: int = 15) -> np.ndarray:
    """Apply uniform rolling average to the motion signal."""
    return uniform_filter1d(diffs, size=window)


def plot_motion_signal(diffs: np.ndarray, smoothed: np.ndarray,
                        fps: float, output_path: str | Path,
                        themes: tuple = ("dark", "light")):
    """
    Diagnostic motion plot: full timeline, a detail window, and the
    distribution. Renders one file per theme so the UI can match the OS
    light/dark preference (`motion_plot.png` + `motion_plot_light.png`).
    """
    output_path = Path(output_path)
    times = np.arange(len(diffs)) / fps

    # detail window — a slice short enough that peaks and valleys separate
    span = min(30.0, times[-1])
    z0 = min(60.0, max(0.0, times[-1] - span))
    z1 = z0 + span
    m = (times >= z0) & (times <= z1)

    for theme in themes:
        C = use_theme(theme)
        fig, axes = plt.subplots(3, 1, figsize=(16, 9),
                                 gridspec_kw={"height_ratios": [1.25, 1, 1],
                                              "hspace": 0.62})

        # ── full timeline ──
        ax = axes[0]
        ax.fill_between(times, smoothed, color=C["accent"], alpha=0.15, lw=0, zorder=1)
        ax.plot(times, smoothed, color=C["accent"], lw=0.55, zorder=2)
        ax.axvspan(z0, z1, color=C["muted"], alpha=0.13, lw=0, zorder=0)
        style_axes(ax, "Motion Signal",
                   f"{len(diffs)} frame pairs · {times[-1]:.0f}s · "
                   f"median {np.median(diffs):.2f}  ·  shaded band shown below",
                   "Time (seconds)", "Mean pixel difference")
        ax.set_xlim(0, times[-1]); ax.set_ylim(0, None)

        # ── detail window, its own panel so nothing is hidden ──
        ax2 = axes[1]
        if m.sum() > 2:
            ax2.fill_between(times[m], smoothed[m], color=C["accent"],
                             alpha=0.2, lw=0, zorder=1)
            ax2.plot(times[m], smoothed[m], color=C["accent"], lw=1.3, zorder=2)
            ax2.set_xlim(z0, z1)
        ax2.set_ylim(0, None)
        style_axes(ax2, f"Detail — {z0:.0f}s to {z1:.0f}s",
                   "valleys are stable pages; spikes are page turns",
                   "Time (seconds)", "Mean pixel difference")

        # ── distribution ──
        ax3 = axes[2]
        ax3.hist(diffs, bins=140, color=C["accent"], alpha=0.75, ec="none", zorder=3)
        for val, col, lbl in [
            (np.median(diffs), C["muted"], f"median {np.median(diffs):.2f}"),
            (np.percentile(diffs, 95), C["yellow"], f"95th {np.percentile(diffs, 95):.2f}"),
        ]:
            ax3.axvline(val, color=col, ls="--", lw=1.2, zorder=4, label=lbl)
        style_axes(ax3, "Distribution",
                   "low values are stable pages; the long tail is page turns",
                   "Mean pixel difference", "Frame count")
        ax3.legend(loc="upper right")

        suffix = "" if theme == "dark" else "_light"
        plt.savefig(str(output_path.with_name(
            output_path.stem + suffix + output_path.suffix)))
        plt.close()
