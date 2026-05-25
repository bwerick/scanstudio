#!/usr/bin/env python3
"""
Phase 6: Split Spreads into Individual Pages

Splits each keyframe spread at the center into left and right pages,
crops to page boundaries (removing table background), and saves
individual page images in order.

Cover frames (marked with is_cover in review) are kept as single pages.

Usage:
  python scripts/p6_split_pages.py output/audiq5
  python scripts/p6_split_pages.py output/audiq5 --safety-margin 0.03
  python scripts/p6_split_pages.py output/audiq5 --no-crop

Inputs:
  - output/<n>/review/final_keyframes.json    Reviewed keyframe list
  - output/<n>/keyframes/*.jpg                Keyframe images

Outputs (in output/<n>/pages/):
  - page_0001_cover.jpg                        Front cover
  - page_0002_left.jpg                         First left page
  - page_0003_right.jpg                        First right page
  - ...
  - page_NNNN_cover.jpg                        Back cover
  - pages.json                                 Page metadata and ordering

Requirements:
  pip install opencv-python numpy
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from utils import log, ensure_dir, check_overwrite_dir


def detect_page_crop(
    half_img: np.ndarray, side: str, safety_pct: float
) -> tuple[int, int, int, int, str]:
    """
    Detect page boundary in a half-spread image using Otsu thresholding
    with fallback to a conservative fixed crop.

    Args:
        half_img: one half of the spread (BGR)
        side: "left" or "right"
        safety_pct: safety margin as fraction of image dimensions

    Returns:
        (x1, y1, x2, y2, method) where method is "otsu" or "fallback"
    """
    h, w = half_img.shape[:2]

    # --- Attempt: Otsu brightness thresholding ---
    gray = cv2.cvtColor(half_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((15, 15), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    otsu_valid = False
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        x, y, bw, bh = cv2.boundingRect(largest)

        area_ratio = area / (w * h)
        height_ratio = bh / h

        # Validation: region should cover enough of the image and be tall
        if area_ratio > 0.35 and height_ratio > 0.7:
            otsu_valid = True

    if otsu_valid:
        mx = int(w * safety_pct)
        my = int(h * safety_pct)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + bw + mx)
        y2 = min(h, y + bh + my)
        return x1, y1, x2, y2, "otsu"
    else:
        # Fallback: conservative fixed crop
        if side == "left":
            x1 = int(w * 0.25)
            x2 = w
        else:
            x1 = 0
            x2 = int(w * 0.75)
        y1 = int(h * 0.01)
        y2 = int(h * 0.98)
        return x1, y1, x2, y2, "fallback"


def split_and_crop_spread(
    img: np.ndarray, safety_pct: float, do_crop: bool
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Split a spread image at center and optionally crop to page boundaries.

    Returns:
        (left_page, right_page, info_dict)
    """
    h, w = img.shape[:2]
    mid = w // 2

    left_half = img[:, :mid]
    right_half = img[:, mid:]

    info = {"split_x": mid, "original_size": f"{w}x{h}"}

    if do_crop:
        l_x1, l_y1, l_x2, l_y2, l_method = detect_page_crop(
            left_half, "left", safety_pct
        )
        r_x1, r_y1, r_x2, r_y2, r_method = detect_page_crop(
            right_half, "right", safety_pct
        )

        left_page = left_half[l_y1:l_y2, l_x1:l_x2]
        right_page = right_half[r_y1:r_y2, r_x1:r_x2]

        info["left_crop"] = f"({l_x1},{l_y1})-({l_x2},{l_y2})"
        info["left_method"] = l_method
        info["left_size"] = f"{left_page.shape[1]}x{left_page.shape[0]}"
        info["right_crop"] = f"({r_x1},{r_y1})-({r_x2},{r_y2})"
        info["right_method"] = r_method
        info["right_size"] = f"{right_page.shape[1]}x{right_page.shape[0]}"
    else:
        left_page = left_half
        right_page = right_half
        info["left_size"] = f"{left_half.shape[1]}x{left_half.shape[0]}"
        info["right_size"] = f"{right_half.shape[1]}x{right_half.shape[0]}"
        info["left_method"] = "none"
        info["right_method"] = "none"

    return left_page, right_page, info


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: Split spreads into pages and build PDF",
    )
    parser.add_argument("output_dir", help="Base output directory (e.g. output/audiq5)")
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.02,
        help="Crop safety margin as fraction (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Skip page boundary detection, just split at center",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG quality for page images (default: 92)",
    )
    args = parser.parse_args()

    log("=" * 60)
    log("PHASE 6: Split Spreads into Pages")
    log("=" * 60)

    base = Path(args.output_dir)
    keyframes_dir = base / "keyframes"
    review_dir = base / "review"
    pages_dir = ensure_dir(base / "pages")

    # Load final keyframes
    final_path = review_dir / "final_keyframes.json"
    if not final_path.exists():
        log(f"ERROR: {final_path} not found. Run Phase 4 review first.")
        sys.exit(1)

    keyframes = json.loads(final_path.read_text())
    log(f"Loaded {len(keyframes)} keyframes")

    # Load crop bounds from review log (if set in Phase 4)
    review_log_path = review_dir / "review_log.json"
    global_crop = None
    if review_log_path.exists():
        review_log = json.loads(review_log_path.read_text())
        crop_info = review_log.get("crop_bounds", {})
        if "global" in crop_info:
            global_crop = crop_info["global"]
            log(
                f"  Global crop bounds: L={global_crop['left']:.1%}, R={global_crop['right']:.1%}"
            )

    covers = [kf for kf in keyframes if kf.get("is_cover")]
    spreads = [kf for kf in keyframes if not kf.get("is_cover")]
    log(f"  Covers: {len(covers)}, Spreads to split: {len(spreads)}")

    # Check overwrite
    if not check_overwrite_dir(pages_dir):
        log("Skipped.")
        return

    # Process each keyframe
    log("")
    log(
        f"Splitting spreads (crop={'off' if args.no_crop else f'on, margin={args.safety_margin}'})..."
    )
    t0 = time.time()

    page_list = []  # ordered list of pages for PDF
    page_num = 0
    otsu_count = 0
    fallback_count = 0

    for kf in keyframes:
        img_path = keyframes_dir / kf["filename"]
        if not img_path.exists():
            log(f"  WARNING: {kf['filename']} not found, skipping")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            log(f"  WARNING: Cannot read {kf['filename']}, skipping")
            continue

        # Apply crop bounds (pre-trim sides before splitting)
        crop = kf.get("crop_bounds") or global_crop
        if crop and not kf.get("is_cover", False):
            h_img, w_img = img.shape[:2]
            x_left = int(w_img * crop["left"])
            x_right = int(w_img * crop["right"])
            img = img[:, x_left:x_right]

        is_cover = kf.get("is_cover", False)
        frame_idx = kf.get("frame_index", page_num)

        if is_cover:
            # Determine if front cover (first) or back cover (last)
            is_last = kf is keyframes[-1]
            cover_type = "backcover" if is_last else "cover"

            page_num += 1
            filename = f"frame{frame_idx:06d}_{cover_type}.jpg"
            cv2.imwrite(
                str(pages_dir / filename),
                img,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )

            page_list.append(
                {
                    "page_num": page_num,
                    "type": cover_type,
                    "filename": filename,
                    "source": kf["filename"],
                    "size": f"{img.shape[1]}x{img.shape[0]}",
                }
            )
        else:
            # Split into left and right pages
            left, right, info = split_and_crop_spread(
                img, args.safety_margin, not args.no_crop
            )

            # Left page
            page_num += 1
            left_fn = f"frame{frame_idx:06d}_left.jpg"
            cv2.imwrite(
                str(pages_dir / left_fn),
                left,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )

            page_list.append(
                {
                    "page_num": page_num,
                    "type": "left",
                    "filename": left_fn,
                    "source": kf["filename"],
                    "size": info["left_size"],
                    "crop_method": info["left_method"],
                }
            )
            if info["left_method"] == "otsu":
                otsu_count += 1
            elif info["left_method"] == "fallback":
                fallback_count += 1

            # Right page
            page_num += 1
            right_fn = f"frame{frame_idx:06d}_right.jpg"
            cv2.imwrite(
                str(pages_dir / right_fn),
                right,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )

            page_list.append(
                {
                    "page_num": page_num,
                    "type": "right",
                    "filename": right_fn,
                    "source": kf["filename"],
                    "size": info["right_size"],
                    "crop_method": info["right_method"],
                }
            )
            if info["right_method"] == "otsu":
                otsu_count += 1
            elif info["right_method"] == "fallback":
                fallback_count += 1

        if page_num % 50 == 0:
            log(f"  Processed {page_num} pages...")

    elapsed = time.time() - t0
    log(f"  Done. {len(page_list)} pages in {elapsed:.1f}s")
    if not args.no_crop:
        log(f"  Crop methods: otsu={otsu_count}, fallback={fallback_count}")

    # Save page metadata
    pages_json = pages_dir / "pages.json"
    pages_json.write_text(json.dumps(page_list, indent=2))
    log(f"  Page metadata: {pages_json}")

    # Summary
    log("")
    log("=" * 60)
    log("PHASE 6 COMPLETE")
    log(f"  Pages: {pages_dir}/ ({len(page_list)} images)")
    log(f"  Metadata: {pages_json}")
    log("=" * 60)


if __name__ == "__main__":
    main()
