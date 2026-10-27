"""Regression checks for the crop API used by both Phase 4 reviewers."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import p5_crop


def test_review_quad_is_rectified():
    image = np.full((160, 240, 3), 230, dtype=np.uint8)
    quad = np.array([[40, 30], [200, 30], [200, 130], [40, 130]], np.float32)
    cropped = p5_crop.crop_to_quad(image, quad, 0)
    assert cropped.shape == (100, 160, 3)
    assert np.all(cropped == 230)


def test_auto_crop_accepts_review_rotation_and_returns_bounds():
    image = np.full((160, 240, 3), 230, dtype=np.uint8)
    mask = np.zeros((160, 240), np.uint8)
    mask[30:130, 40:200] = 255
    with patch.object(p5_crop, "page_mask_robust", return_value=mask):
        cropped, method, bounds = p5_crop.crop_double_page(image, 0, 0.0)
    assert bounds == (40, 30, 160, 100)
    assert cropped.shape == (100, 160, 3)
    assert method == "hsv_deskew"
    assert abs(p5_crop._spread_tilt(mask)) < 0.01


def test_crop_cli_applies_review_box_in_both_modes():
    # The import repair must also carry the operator's reviewed box through
    # the real crop command, rather than silently ignoring their adjustment.
    for mode in ("double", "single"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "json").mkdir()
            image = np.full((160, 240, 3), 230, dtype=np.uint8)
            filename = "frame000001.jpg"
            assert cv2.imwrite(str(root / "images" / filename), image)
            keyframes = [{"filename": filename, "frame_index": 1,
                          "crop_quad": [[0.25, 0.25], [0.75, 0.25],
                                        [0.75, 0.75], [0.25, 0.75]]}]
            (root / "json" / "keyframes.json").write_text(json.dumps(keyframes))
            result = subprocess.run(
                [sys.executable, str(SCRIPTS / "p5_crop.py"), str(root),
                 "--mode", mode, "--padding", "0"],
                capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            assert "manual_quad" in result.stdout
            cropped = cv2.imread(str(root / "images" / filename))
            assert cropped.shape == (80, 120, 3), (mode, cropped.shape)


if __name__ == "__main__":
    test_review_quad_is_rectified()
    test_auto_crop_accepts_review_rotation_and_returns_bounds()
    test_crop_cli_applies_review_box_in_both_modes()
    print("3/3 crop-review contract tests passed")
