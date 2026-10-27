"""Headless tests for the Phase 3.5 image-only duplicate detector."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from frame_quality import find_likely_duplicates  # noqa: E402


def page(seed: int) -> np.ndarray:
    """A page-like synthetic image with repeatable text-shaped detail."""
    rng = np.random.default_rng(seed)
    image = np.full((180, 130), 220, dtype=np.uint8)
    for row in range(20, 160, 12):
        start = int(rng.integers(12, 30))
        end = int(rng.integers(90, 120))
        cv2.line(image, (start, row), (end, row), int(rng.integers(20, 90)), 2)
    return image


def test_exact_duplicate_is_flagged():
    first = page(1)
    flags = find_likely_duplicates([first, first.copy()])
    assert len(flags) == 1
    assert flags[0]["current_index"] == 1 and flags[0]["match_index"] == 0


def test_different_pages_are_not_flagged():
    flags = find_likely_duplicates([page(1), page(2)])
    assert flags == []


def test_nearby_copy_with_small_exposure_change_is_flagged():
    first = page(3)
    second = cv2.convertScaleAbs(first, alpha=1.0, beta=3)
    flags = find_likely_duplicates([first, second])
    assert len(flags) == 1


def test_only_nearby_frames_are_compared():
    first = page(4)
    flags = find_likely_duplicates([first, page(5), first.copy()])
    assert flags == []
    flags = find_likely_duplicates([first, page(5), first.copy()], max_neighbor_gap=2)
    assert len(flags) == 1


if __name__ == "__main__":
    test_exact_duplicate_is_flagged()
    test_different_pages_are_not_flagged()
    test_nearby_copy_with_small_exposure_change_is_flagged()
    test_only_nearby_frames_are_compared()
    print("all frame quality tests passed")
