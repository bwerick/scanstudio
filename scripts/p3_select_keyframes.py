#!/usr/bin/env python3
"""
Phase 3: Select Keyframes

For each spread, selects the lowest-motion frame (sharpness as tiebreaker).
Saves full-resolution images to images/ and metadata to json/keyframes.json.

Usage:
  python scripts/p3_select_keyframes.py output/mybook recordings/mybook.mp4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import log, ProjectPaths, check_overwrite_dir


def select_keyframe(cap, smoothed, start, end, sample_rate, motion_margin):
    if end <= start:
        return None
    region = smoothed[start:min(end, len(smoothed))]
    if len(region) == 0:
        return None

    min_motion = np.min(region)
    threshold = min_motion + motion_margin
    clean_frames = [start + i for i in range(len(region)) if region[i] < threshold]
    if not clean_frames:
        clean_frames = [start + np.argmin(region)]

    candidates = clean_frames[::sample_rate] or [clean_frames[len(clean_frames) // 2]]

    best_frame, best_sharpness, best_motion = None, -1, float('inf')
    for fi in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        center = gray[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        sharpness = cv2.Laplacian(center, cv2.CV_64F).var()
        if sharpness > best_sharpness:
            best_sharpness = sharpness
            best_frame = fi
            best_motion = smoothed[min(fi, len(smoothed)-1)]

    return best_frame, best_motion, best_sharpness


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Select keyframes")
    parser.add_argument("output_dir", help="Base output directory")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--sample-rate", type=int, default=6)
    parser.add_argument("--motion-margin", type=float, default=0.5)
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="JPEG quality for extracted keyframes")
    args = parser.parse_args()

    log("=" * 60)
    log("PHASE 3: Select Keyframes")
    log("=" * 60)

    paths = ProjectPaths(args.output_dir)
    paths.ensure("images", "json", "plots")

    spreads = json.loads((paths.json / "spreads.json").read_text())
    smoothed = np.load(str(paths.data / "smoothed_signal.npy"))
    metadata = json.loads((paths.json / "metadata.json").read_text())
    fps = metadata["fps"]
    log(f"  {len(spreads)} spreads, {fps} fps")

    if not check_overwrite_dir(paths.images):
        log("Skipped.")
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        log(f"ERROR: Cannot open video: {args.video}")
        sys.exit(1)
    log(f"  Video: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    t0 = time.time()
    keyframe_data = []

    for i, sp in enumerate(spreads):
        result = select_keyframe(cap, smoothed, sp["start_frame"], sp["end_frame"],
                                  args.sample_rate, args.motion_margin)
        if result is None:
            log(f"  WARNING: No keyframe for spread {sp['spread_index']}")
            continue

        frame_idx, motion_val, sharpness = result
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        filename = f"frame{frame_idx:06d}.jpg"
        cv2.imwrite(str(paths.images / filename), frame,
                     [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

        keyframe_data.append({
            "frame_index": frame_idx,
            "time_sec": round(frame_idx / fps, 2),
            "motion_value": round(float(motion_val), 4),
            "sharpness": round(float(sharpness), 1),
            "filename": filename,
            "spread_start": sp["start_frame"],
            "spread_end": sp["end_frame"],
            "spread_duration": sp["duration_sec"],
            "source": "algorithm",
        })

        if (i + 1) % 25 == 0 or i == len(spreads) - 1:
            log(f"  {i+1}/{len(spreads)} — frame {frame_idx} (motion={motion_val:.2f}, sharp={sharpness:.0f})")

    cap.release()
    log(f"  Done. {len(keyframe_data)} keyframes in {time.time()-t0:.1f}s")

    (paths.json / "keyframes.json").write_text(json.dumps(keyframe_data, indent=2))

    # Selection plot
    times = np.arange(len(smoothed)) / fps
    fig, ax = plt.subplots(1, 1, figsize=(22, 6))
    ax.plot(times, smoothed, linewidth=0.4, color="steelblue", alpha=0.7)
    kf_t = [kf["frame_index"] / fps for kf in keyframe_data]
    kf_m = [kf["motion_value"] for kf in keyframe_data]
    ax.plot(kf_t, kf_m, "g^", markersize=4, label=f"Keyframes ({len(keyframe_data)})")
    ax.set_title(f"Keyframe Selection — {len(keyframe_data)} keyframes")
    ax.set_xlabel("Time (s)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(paths.plots / "selection_plot.png"), dpi=150)
    plt.close()

    motions = [kf["motion_value"] for kf in keyframe_data]
    high = [kf for kf in keyframe_data if kf["motion_value"] > 3.0]
    log(f"  Motion: median={np.median(motions):.2f}, max={max(motions):.2f}")
    if high:
        log(f"  ⚠ {len(high)} keyframes with motion > 3.0")

    log("PHASE 3 COMPLETE")


if __name__ == "__main__":
    main()
