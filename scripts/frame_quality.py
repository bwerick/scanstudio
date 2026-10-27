"""Image-only quality signals for ScanStudio keyframes.

The first quality pass is deliberately conservative. It compares only nearby
keyframes and *flags* likely duplicate captures; it never removes an image.
The P4 reviewer remains the authority on whether a frame should be deleted.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_preview(path: Path, size: tuple[int, int] = (96, 96)) -> np.ndarray | None:
    """Load a normalized grayscale preview suitable for inexpensive comparison."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(image, (3, 3), 0)


def dhash(image: np.ndarray) -> np.ndarray:
    """Return a 64-bit difference hash that tolerates small exposure changes."""
    tiny = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    return tiny[:, 1:] > tiny[:, :-1]


def hamming_distance(first: np.ndarray, second: np.ndarray) -> int:
    return int(np.count_nonzero(first != second))


def pixel_difference(first: np.ndarray, second: np.ndarray) -> float:
    """Mean absolute difference, normalized to the 0-1 grayscale range."""
    return float(np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32))) / 255.0)


def compare(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    """Return complementary structural and pixel-difference signals."""
    return {
        "hash_distance": hamming_distance(dhash(first), dhash(second)),
        "pixel_difference": pixel_difference(first, second),
    }


def find_likely_duplicates(
    previews: list[np.ndarray | None],
    max_neighbor_gap: int = 1,
    hash_threshold: int = 4,
    difference_threshold: float = 0.02,
) -> list[dict[str, float | int]]:
    """Flag near-identical keyframes within a small scan-order window.

    Requiring both a perceptual hash match and a low pixel difference avoids
    flagging adjacent pages that happen to have a similar layout. The function
    reports each later frame at most once, matched with its closest earlier
    neighbor.
    """
    flags: list[dict[str, float | int]] = []
    for current_idx, current in enumerate(previews):
        if current is None:
            continue
        best: dict[str, float | int] | None = None
        for previous_idx in range(max(0, current_idx - max_neighbor_gap), current_idx):
            previous = previews[previous_idx]
            if previous is None:
                continue
            scores = compare(previous, current)
            if (
                scores["hash_distance"] <= hash_threshold
                and scores["pixel_difference"] <= difference_threshold
            ):
                candidate: dict[str, float | int] = {
                    "current_index": current_idx,
                    "match_index": previous_idx,
                    **scores,
                }
                if best is None or candidate["pixel_difference"] < best["pixel_difference"]:
                    best = candidate
        if best is not None:
            flags.append(best)
    return flags
