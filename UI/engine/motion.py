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
                        fps: float, output_path: str | Path):
    """Generate the diagnostic motion signal plot."""
    times = np.arange(len(diffs)) / fps

    fig, axes = plt.subplots(3, 1, figsize=(22, 13))

    ax = axes[0]
    ax.plot(times, smoothed, linewidth=0.4, color="steelblue", label="Smoothed")
    ax.plot(times, diffs, linewidth=0.15, color="lightblue", alpha=0.4, label="Raw")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean Pixel Difference")
    ax.set_title(f"Motion Signal — {len(diffs)} frame pairs, {times[-1]:.0f}s duration")
    ax.legend(loc="upper right")

    ax = axes[1]
    mask = times < 120
    ax.plot(times[mask], smoothed[mask], linewidth=0.7, color="steelblue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean Pixel Difference")
    ax.set_title("First 120 Seconds — Zoomed")

    ax = axes[2]
    ax.hist(diffs, bins=150, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(x=np.median(diffs), color="red", linestyle="--",
               label=f"Median: {np.median(diffs):.2f}")
    ax.axvline(x=np.percentile(diffs, 95), color="orange", linestyle="--",
               label=f"95th: {np.percentile(diffs, 95):.2f}")
    ax.set_xlabel("Mean Pixel Difference")
    ax.set_ylabel("Count")
    ax.set_title("Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
