#!/usr/bin/env python3
"""
Phase 1: Compute Motion Signal

Usage:
  python scripts/p1_motion_signal.py recordings/mybook.mp4
  python scripts/p1_motion_signal.py recordings/mybook.mp4 --output-dir output/custom
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

from utils import log, derive_output_dir, ProjectPaths, check_overwrite


def compute_motion_signal(video_path, analysis_height):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = analysis_height / orig_h
    aw = int(orig_w * scale)

    log(f"Video: {orig_w}x{orig_h} @ {fps:.1f}fps, {total_frames} frames ({total_frames/fps:.1f}s)")
    log(f"Analysis: {aw}x{analysis_height}")

    metadata = {
        "video_path": str(video_path), "fps": fps, "total_frames": total_frames,
        "duration_sec": total_frames / fps, "original_width": orig_w,
        "original_height": orig_h, "analysis_width": aw,
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
        if frame_idx % 3000 == 0:
            elapsed = time.time() - t0
            pct = frame_idx / total_frames * 100
            eta = elapsed / frame_idx * (total_frames - frame_idx)
            log(f"  {frame_idx}/{total_frames} ({pct:.0f}%) — ETA: {eta:.0f}s")

    cap.release()
    elapsed = time.time() - t0
    log(f"  Done. {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.0f} fps)")
    metadata["frames_processed"] = frame_idx
    metadata["processing_time_sec"] = round(elapsed, 1)
    return np.array(diffs), metadata


def generate_plot(diffs, smoothed, fps, output_path):
    times = np.arange(len(diffs)) / fps
    fig, axes = plt.subplots(3, 1, figsize=(22, 13))

    ax = axes[0]
    ax.plot(times, smoothed, linewidth=0.4, color="steelblue", label="Smoothed")
    ax.plot(times, diffs, linewidth=0.15, color="lightblue", alpha=0.4, label="Raw")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean pixel diff")
    ax.set_title(f"Motion Signal — {len(diffs)} frame pairs, {times[-1]:.0f}s duration")
    ax.legend(loc="upper right")

    ax = axes[1]
    mask = times < 120
    ax.plot(times[mask], smoothed[mask], linewidth=0.7, color="steelblue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean pixel diff")
    ax.set_title("First 120 Seconds — Zoomed")

    ax = axes[2]
    ax.hist(diffs, bins=150, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(x=np.median(diffs), color="red", linestyle="--", label=f"Median: {np.median(diffs):.2f}")
    ax.axvline(x=np.percentile(diffs, 95), color="orange", linestyle="--", label=f"95th: {np.percentile(diffs, 95):.2f}")
    ax.set_xlabel("Mean pixel diff")
    ax.set_ylabel("Count")
    ax.set_title("Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Compute motion signal")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--analysis-height", type=int, default=360)
    parser.add_argument("--smoothing-window", type=int, default=15)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    log("=" * 60)
    log("PHASE 1: Compute Motion Signal")
    log("=" * 60)

    paths = ProjectPaths(derive_output_dir(args.video, args.output_dir))
    paths.ensure("data", "json", "plots")
    log(f"Output: {paths.base}")

    signal_path = paths.data / "motion_signal.npy"
    if signal_path.exists():
        if not check_overwrite(signal_path):
            log("Skipped.")
            return

    diffs, metadata = compute_motion_signal(args.video, args.analysis_height)

    smoothed = uniform_filter1d(diffs, size=args.smoothing_window)
    metadata["smoothing_window"] = args.smoothing_window

    np.save(str(signal_path), diffs)
    np.save(str(paths.data / "smoothed_signal.npy"), smoothed)
    (paths.json / "metadata.json").write_text(json.dumps(metadata, indent=2))

    log(f"Signal stats: min={diffs.min():.2f}, max={diffs.max():.2f}, median={np.median(diffs):.2f}")

    generate_plot(diffs, smoothed, metadata["fps"], str(paths.plots / "motion_plot.png"))
    log(f"  Plot saved: {paths.plots / 'motion_plot.png'}")

    log("")
    log("PHASE 1 COMPLETE")


if __name__ == "__main__":
    main()
