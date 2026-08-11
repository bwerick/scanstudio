"""
Keyframe selection.

For each spread, selects the frame with the lowest motion value,
using sharpness as a tiebreaker.
"""

import cv2
import numpy as np
from pathlib import Path


def select_keyframe(cap: cv2.VideoCapture, smoothed: np.ndarray,
                     start: int, end: int,
                     sample_rate: int = 6, motion_margin: float = 0.5):
    """
    Select the best keyframe from a spread.

    Returns:
        (frame_index, motion_value, sharpness) or None
    """
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
        center = gray[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
        sharpness = cv2.Laplacian(center, cv2.CV_64F).var()
        if sharpness > best_sharpness:
            best_sharpness = sharpness
            best_frame = fi
            best_motion = float(smoothed[min(fi, len(smoothed) - 1)])

    return best_frame, best_motion, best_sharpness


def extract_frame(video_path: str, frame_index: int) -> np.ndarray | None:
    """Read a single frame from the video at full resolution."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def select_all_keyframes(video_path: str, spreads: list[dict],
                          smoothed: np.ndarray, fps: float,
                          output_dir: Path, jpeg_quality: int = 95,
                          sample_rate: int = 6, motion_margin: float = 0.5,
                          on_progress=None) -> list[dict]:
    """
    Select keyframes for all spreads and save images.

    Returns:
        list of keyframe metadata dicts
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    keyframe_data = []

    for i, sp in enumerate(spreads):
        result = select_keyframe(cap, smoothed, sp["start_frame"], sp["end_frame"],
                                  sample_rate, motion_margin)
        if result is None:
            continue

        frame_idx, motion_val, sharpness = result
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        filename = f"frame{frame_idx:06d}.jpg"
        cv2.imwrite(str(output_dir / filename), frame,
                     [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        keyframe_data.append({
            "frame_index": frame_idx,
            "time_sec": round(frame_idx / fps, 2),
            "motion_value": round(motion_val, 4),
            "sharpness": round(sharpness, 1),
            "filename": filename,
            "spread_start": sp["start_frame"],
            "spread_end": sp["end_frame"],
            "spread_duration": sp["duration_sec"],
            "source": "algorithm",
        })

        if on_progress:
            on_progress(i + 1, len(spreads))

    cap.release()
    return keyframe_data
