#!/usr/bin/env python3
"""
Phase 1: Compute Motion Signal

Reads a video file, computes the frame-to-frame mean absolute pixel difference
at reduced resolution, and saves the motion signal for downstream processing.

Usage:
  python scripts/01_motion_signal.py recordings/bookflip_20260220_140323.mp4
  python scripts/01_motion_signal.py recordings/bookflip_20260220_140323.mp4 --analysis-height 480
  python scripts/01_motion_signal.py recordings/bookflip_20260220_140323.mp4 --output-dir output/my_custom_dir

Inputs:
  - A video file (MP4, MOV, etc.)

Outputs (in output/<video_name>/motion/):
  - motion_signal.npy      Raw frame-to-frame diffs (length = total_frames - 1)
  - smoothed_signal.npy    Uniform-filtered smoothed signal
  - metadata.json          Video properties and parameters used
  - motion_plot.png        Diagnostic plot (full signal, first 120s zoom, histogram)

Requirements:
  pip install opencv-python numpy scipy matplotlib
"""

import argparse
import json
import time
import sys

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import log, derive_output_dir, ensure_dir, check_overwrite


def compute_motion_signal(
    video_path: str, analysis_height: int
) -> tuple[np.ndarray, dict]:
    """
    Compute frame-to-frame mean absolute difference for every frame in the video.
    Downscales each frame to analysis_height before computing.

    Returns:
        diffs: 1D numpy array of mean pixel differences (length = total_frames - 1)
        metadata: dict with video properties
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = analysis_height / orig_h
    analysis_width = int(orig_w * scale)

    log(
        f"Video: {orig_w}x{orig_h} @ {fps:.1f}fps, {total_frames} frames ({total_frames/fps:.1f}s)"
    )
    log(f"Analysis resolution: {analysis_width}x{analysis_height}")

    metadata = {
        "video_path": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": total_frames / fps,
        "original_width": orig_w,
        "original_height": orig_h,
        "analysis_width": analysis_width,
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
        small = cv2.resize(
            frame, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            diffs.append(np.mean(diff))
        prev_gray = gray
        frame_idx += 1
        if frame_idx % 3000 == 0:
            elapsed = time.time() - t0
            pct = frame_idx / total_frames * 100
            eta = elapsed / frame_idx * (total_frames - frame_idx)
            log(
                f"  Progress: {frame_idx}/{total_frames} ({pct:.0f}%) — ETA: {eta:.0f}s"
            )

    cap.release()
    elapsed = time.time() - t0
    log(f"  Done. {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.0f} fps)")

    metadata["frames_processed"] = frame_idx
    metadata["processing_time_sec"] = round(elapsed, 1)

    return np.array(diffs), metadata


def generate_plot(
    diffs: np.ndarray, smoothed: np.ndarray, fps: float, output_path: str
):
    """Generate the diagnostic motion signal plot."""
    times = np.arange(len(diffs)) / fps

    fig, axes = plt.subplots(3, 1, figsize=(22, 13))

    # Full signal
    ax = axes[0]
    ax.plot(times, smoothed, linewidth=0.4, color="steelblue", label="Smoothed")
    ax.plot(times, diffs, linewidth=0.15, color="lightblue", alpha=0.4, label="Raw")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mean pixel diff")
    ax.set_title(f"Motion Signal — {len(diffs)} frame pairs, {times[-1]:.0f}s duration")
    ax.legend(loc="upper right")

    # First 120 seconds
    ax = axes[1]
    mask = times < 120
    ax.plot(
        times[mask], smoothed[mask], linewidth=0.7, color="steelblue", label="Smoothed"
    )
    ax.plot(times[mask], diffs[mask], linewidth=0.3, color="lightblue", alpha=0.5)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mean pixel diff")
    ax.set_title("First 120 Seconds — Zoomed")

    # Histogram
    ax = axes[2]
    ax.hist(diffs, bins=150, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(
        x=np.median(diffs),
        color="red",
        linestyle="--",
        label=f"Median: {np.median(diffs):.2f}",
    )
    ax.axvline(
        x=np.percentile(diffs, 95),
        color="orange",
        linestyle="--",
        label=f"95th pct: {np.percentile(diffs, 95):.2f}",
    )
    ax.set_xlabel("Mean pixel diff")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Frame-to-Frame Differences")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    log(f"  Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Compute motion signal from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument(
        "--analysis-height",
        type=int,
        default=360,
        help="Resolution height for motion analysis (default: 360)",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=15,
        help="Smoothing window in frames (default: 15 = 0.5s at 30fps)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: derived from video filename)",
    )
    args = parser.parse_args()

    log("=" * 60)
    log("PHASE 1: Compute Motion Signal")
    log("=" * 60)

    # Resolve output directory
    base_output = derive_output_dir(args.video, args.output_dir)
    motion_dir = ensure_dir(base_output / "motion")
    log(f"Output: {motion_dir}")

    # Check for existing outputs
    signal_path = motion_dir / "motion_signal.npy"
    smoothed_path = motion_dir / "smoothed_signal.npy"
    metadata_path = motion_dir / "metadata.json"
    plot_path = motion_dir / "motion_plot.png"

    if signal_path.exists():
        if not check_overwrite(signal_path):
            log("Skipped. Existing motion signal preserved.")
            return

    # Compute
    log("")
    diffs, metadata = compute_motion_signal(args.video, args.analysis_height)

    # Smooth
    log("")
    log(f"Smoothing with window={args.smoothing_window}...")
    smoothed = uniform_filter1d(diffs, size=args.smoothing_window)
    metadata["smoothing_window"] = args.smoothing_window

    # Save
    log("")
    log("Saving outputs...")

    np.save(str(signal_path), diffs)
    log(f"  Motion signal: {signal_path} ({len(diffs)} values)")

    np.save(str(smoothed_path), smoothed)
    log(f"  Smoothed signal: {smoothed_path}")

    metadata_path.write_text(json.dumps(metadata, indent=2))
    log(f"  Metadata: {metadata_path}")

    # Signal statistics
    log("")
    log("Signal statistics:")
    log(f"  Min: {diffs.min():.2f}, Max: {diffs.max():.2f}")
    log(f"  Mean: {diffs.mean():.2f}, Median: {np.median(diffs):.2f}")
    log(f"  Std: {diffs.std():.2f}")
    log(
        f"  90th pct: {np.percentile(diffs, 90):.2f}, 95th: {np.percentile(diffs, 95):.2f}"
    )

    # Plot
    log("")
    log("Generating plot...")
    generate_plot(diffs, smoothed, metadata["fps"], str(plot_path))

    # Done
    log("")
    log("=" * 60)
    log("PHASE 1 COMPLETE")
    log(f"  Motion signal:   {signal_path}")
    log(f"  Smoothed signal: {smoothed_path}")
    log(f"  Metadata:        {metadata_path}")
    log(f"  Plot:            {plot_path}")
    log("=" * 60)


if __name__ == "__main__":
    main()
