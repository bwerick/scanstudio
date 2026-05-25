#!/usr/bin/env python3
"""
Phase 3: Select Keyframes

For each spread identified in Phase 2, selects the best keyframe by finding
the frame with the lowest motion value (our ground truth analysis showed this
is the most reliable way to avoid hand occlusion — low motion = stable page
with no hand movement).

Uses sharpness (Laplacian variance) as a tiebreaker among frames with
similarly low motion values.

Usage:
  python scripts/03_select_keyframes.py output/audiq5 recordings/audiq5.mp4
  python scripts/03_select_keyframes.py output/audiq5 recordings/audiq5.mp4 --sample-rate 3
  python scripts/03_select_keyframes.py output/audiq5 recordings/audiq5.mp4 --jpeg-quality 90

Inputs:
  - output/<n>/peaks/spreads.json     Spread boundaries from Phase 2
  - output/<n>/motion/metadata.json   Video metadata (fps)
  - output/<n>/motion/smoothed_signal.npy  For motion-based frame scoring
  - The original video file            For extracting full-resolution frames

Outputs (in output/<n>/keyframes/):
  - spread_NNNN_frameXXXXXX.jpg       One keyframe image per spread
  - keyframes.json                     Metadata for each selected keyframe
  - selection_plot.png                 Motion signal with selected keyframes marked

Requirements:
  pip install opencv-python numpy scipy matplotlib
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import log, ensure_dir, check_overwrite_dir


def select_keyframe_for_spread(
    cap: cv2.VideoCapture,
    smoothed: np.ndarray,
    start: int,
    end: int,
    sample_rate: int,
    motion_margin: float,
) -> tuple[int, float, float] | None:
    """
    Select the best keyframe from a spread.

    Strategy:
      1. Find the frame with the lowest smoothed motion value in the spread
      2. Among frames within motion_margin of the minimum, pick the sharpest

    Args:
        cap: OpenCV video capture (positioned at any frame)
        smoothed: smoothed motion signal array
        start: start frame of spread
        end: end frame of spread
        sample_rate: evaluate every Nth frame for sharpness
        motion_margin: consider frames within this much of the minimum motion

    Returns:
        (frame_index, motion_value, sharpness) or None
    """
    if end <= start:
        return None

    # Step 1: Find the minimum motion frame in the spread
    region = smoothed[start : min(end, len(smoothed))]
    if len(region) == 0:
        return None

    min_motion = np.min(region)
    min_motion_frame = start + np.argmin(region)

    # Step 2: Find all frames within motion_margin of the minimum
    # These are all "clean" frames where the page is stable
    threshold = min_motion + motion_margin
    clean_mask = region < threshold
    clean_frames = [start + i for i in range(len(region)) if clean_mask[i]]

    if not clean_frames:
        clean_frames = [min_motion_frame]

    # Step 3: Among clean frames, sample every Nth and pick sharpest
    candidates = clean_frames[::sample_rate]
    if not candidates:
        candidates = [clean_frames[len(clean_frames) // 2]]

    best_frame = None
    best_sharpness = -1
    best_motion = float("inf")

    for frame_idx in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # Sharpness on center region (avoid edges/corners)
        center = gray[int(h * 0.1) : int(h * 0.9), int(w * 0.1) : int(w * 0.9)]
        sharpness = cv2.Laplacian(center, cv2.CV_64F).var()
        motion = smoothed[min(frame_idx, len(smoothed) - 1)]

        if sharpness > best_sharpness:
            best_sharpness = sharpness
            best_frame = frame_idx
            best_motion = motion

    return best_frame, best_motion, best_sharpness


def generate_plot(
    smoothed: np.ndarray,
    fps: float,
    spreads: list[dict],
    keyframe_data: list[dict],
    output_path: str,
):
    """Generate plot showing selected keyframes on the motion signal."""
    times = np.arange(len(smoothed)) / fps

    fig, axes = plt.subplots(2, 1, figsize=(22, 10))

    # Full signal with keyframes
    ax = axes[0]
    ax.plot(
        times,
        smoothed,
        linewidth=0.4,
        color="steelblue",
        alpha=0.7,
        label="Smoothed motion",
    )
    kf_frames = [kf["frame_index"] for kf in keyframe_data]
    kf_motions = [kf["motion_value"] for kf in keyframe_data]
    kf_times = [f / fps for f in kf_frames]
    ax.plot(
        kf_times,
        kf_motions,
        "g^",
        markersize=4,
        label=f"Keyframes ({len(keyframe_data)})",
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Motion")
    ax.set_title(
        f"Keyframe Selection — {len(keyframe_data)} keyframes (lowest motion per spread)"
    )
    ax.legend(loc="upper right")

    # First 120s zoomed with keyframes
    ax = axes[1]
    mask = times < 120
    ax.plot(times[mask], smoothed[mask], linewidth=0.7, color="steelblue")
    for kf in keyframe_data:
        t = kf["frame_index"] / fps
        if t < 120:
            ax.axvline(x=t, color="green", alpha=0.3, linewidth=0.8)
            ax.plot(t, kf["motion_value"], "g^", markersize=6)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Motion")
    ax.set_title("First 120 Seconds — Keyframes marked")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Select keyframes from each spread",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("output_dir", help="Base output directory (e.g. output/audiq5)")
    parser.add_argument("video", help="Path to original video file")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=6,
        help="Evaluate every Nth clean frame for sharpness (default: 6)",
    )
    parser.add_argument(
        "--motion-margin",
        type=float,
        default=0.5,
        help="Consider frames within this much of min motion (default: 0.5)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for saved keyframes (default: 95)",
    )
    args = parser.parse_args()

    log("=" * 60)
    log("PHASE 3: Select Keyframes")
    log("=" * 60)

    base = Path(args.output_dir)
    peaks_dir = base / "peaks"
    motion_dir = base / "motion"
    keyframes_dir = ensure_dir(base / "keyframes")

    # Verify inputs
    spreads_path = peaks_dir / "spreads.json"
    smoothed_path = motion_dir / "smoothed_signal.npy"
    meta_path = motion_dir / "metadata.json"

    for p in [spreads_path, smoothed_path, meta_path]:
        if not p.exists():
            log(f"ERROR: Required input not found: {p}")
            sys.exit(1)

    if not Path(args.video).exists():
        log(f"ERROR: Video not found: {args.video}")
        sys.exit(1)

    # Check overwrite
    if not check_overwrite_dir(keyframes_dir):
        log("Skipped. Existing keyframes preserved.")
        return

    # Load
    log("")
    log("Loading inputs...")
    spreads = json.loads(spreads_path.read_text())
    smoothed = np.load(str(smoothed_path))
    metadata = json.loads(meta_path.read_text())
    fps = metadata["fps"]
    log(f"  {len(spreads)} spreads, {len(smoothed)} motion values, {fps} fps")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        log(f"ERROR: Cannot open video: {args.video}")
        sys.exit(1)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log(f"  Video: {orig_w}x{orig_h}")

    # Select keyframes
    log("")
    log(
        f"Selecting keyframes (sample_rate={args.sample_rate}, motion_margin={args.motion_margin})..."
    )
    t0 = time.time()

    keyframe_data = []

    for i, sp in enumerate(spreads):
        start = sp["start_frame"]
        end = sp["end_frame"]

        result = select_keyframe_for_spread(
            cap, smoothed, start, end, args.sample_rate, args.motion_margin
        )

        if result is None:
            log(f"  WARNING: No keyframe found for spread {sp['spread_index']}")
            continue

        frame_idx, motion_val, sharpness = result

        # Read the full-res frame and save
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            log(f"  WARNING: Could not read frame {frame_idx}")
            continue

        filename = f"spread_{sp['spread_index']:04d}_frame{frame_idx:06d}.jpg"
        filepath = keyframes_dir / filename
        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

        kf_entry = {
            "spread_index": sp["spread_index"],
            "frame_index": frame_idx,
            "time_sec": round(frame_idx / fps, 2),
            "motion_value": round(float(motion_val), 4),
            "sharpness": round(float(sharpness), 1),
            "filename": filename,
            "spread_start": start,
            "spread_end": end,
            "spread_duration": sp["duration_sec"],
        }
        keyframe_data.append(kf_entry)

        if (i + 1) % 25 == 0 or i == len(spreads) - 1:
            elapsed = time.time() - t0
            log(
                f"  {i + 1}/{len(spreads)} — frame {frame_idx} "
                f"(motion={motion_val:.2f}, sharp={sharpness:.0f})"
            )

    cap.release()
    elapsed = time.time() - t0
    log(f"  Done. {len(keyframe_data)} keyframes in {elapsed:.1f}s")

    # Save keyframe metadata
    log("")
    log("Saving outputs...")

    kf_json_path = keyframes_dir / "keyframes.json"
    kf_json_path.write_text(json.dumps(keyframe_data, indent=2))
    log(f"  Keyframe metadata: {kf_json_path}")

    # Stats
    motions = [kf["motion_value"] for kf in keyframe_data]
    sharpnesses = [kf["sharpness"] for kf in keyframe_data]

    log("")
    log("Keyframe statistics:")
    log(
        f"  Motion:    min={min(motions):.2f}, max={max(motions):.2f}, "
        f"mean={np.mean(motions):.2f}, median={np.median(motions):.2f}"
    )
    log(
        f"  Sharpness: min={min(sharpnesses):.0f}, max={max(sharpnesses):.0f}, "
        f"mean={np.mean(sharpnesses):.0f}"
    )

    high_motion = [kf for kf in keyframe_data if kf["motion_value"] > 3.0]
    if high_motion:
        log(f"  Keyframes with motion > 3.0 (possible hand): {len(high_motion)}")
        for kf in high_motion:
            log(
                f"    Spread {kf['spread_index']}: frame {kf['frame_index']} "
                f"({kf['time_sec']}s), motion={kf['motion_value']:.2f}"
            )

    # Plot
    log("")
    log("Generating plot...")
    plot_path = keyframes_dir / "selection_plot.png"
    generate_plot(smoothed, fps, spreads, keyframe_data, str(plot_path))
    log(f"  Plot saved: {plot_path}")

    # Summary
    log("")
    log("=" * 60)
    log("PHASE 3 COMPLETE")
    log(f"  Keyframes: {keyframes_dir}/ ({len(keyframe_data)} images)")
    log(f"  Metadata:  {kf_json_path}")
    log(f"  Plot:      {plot_path}")
    log(f"")
    log(f"  Next: Review keyframes with Phase 4")
    log(f"    python scripts/04_review_keyframes.py {args.output_dir} {args.video}")
    log("=" * 60)


if __name__ == "__main__":
    main()
