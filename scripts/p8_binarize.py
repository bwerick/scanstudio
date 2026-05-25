#!/usr/bin/env python3
"""
Phase 8: Binarize Pages (Black & White)

Converts page images to clean black-and-white using adaptive gaussian
thresholding for maximum text legibility.

Usage:
  python scripts/p8_binarize.py output/audiq5
  python scripts/p8_binarize.py output/audiq5 --block-size 31 --offset 8
  python scripts/p8_binarize.py output/audiq5 --skip-covers

Inputs:
  - output/<n>/pages/pages.json     Page metadata from Phase 6
  - output/<n>/pages/*.jpg          Page images

Outputs (in output/<n>/bw/):
  - (same filenames as pages/)      Binarized page images
  - bw_metadata.json                Parameters used

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


def binarize_page(img_bgr: np.ndarray, block_size: int, offset: int) -> np.ndarray:
    """
    Binarize a page image using adaptive gaussian thresholding.

    Args:
        img_bgr: input image (BGR)
        block_size: neighborhood size for adaptive threshold (must be odd)
        offset: constant subtracted from the mean (higher = whiter background)

    Returns:
        Binary image (uint8, 0 or 255)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        offset,
    )
    return binary


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8: Binarize pages to black & white",
    )
    parser.add_argument("output_dir", help="Base output directory (e.g. output/audiq5)")
    parser.add_argument(
        "--block-size",
        type=int,
        default=51,
        help="Adaptive threshold block size, must be odd (default: 51)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=10,
        help="Threshold offset — higher = whiter background (default: 10)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG quality for output images (default: 92)",
    )
    parser.add_argument(
        "--skip-covers",
        action="store_true",
        help="Skip cover images (keep them in color)",
    )
    args = parser.parse_args()

    # Ensure block size is odd
    if args.block_size % 2 == 0:
        args.block_size += 1
        log(f"Block size adjusted to {args.block_size} (must be odd)")

    log("=" * 60)
    log("PHASE 8: Binarize Pages")
    log("=" * 60)

    base = Path(args.output_dir)
    pages_dir = base / "pages"
    bw_dir = ensure_dir(base / "bw")

    # Load page list
    pages_json = pages_dir / "pages.json"
    if not pages_json.exists():
        log(f"ERROR: {pages_json} not found. Run Phase 6 first.")
        sys.exit(1)

    pages = json.loads(pages_json.read_text())
    log(f"Loaded {len(pages)} pages")

    # Check overwrite
    if not check_overwrite_dir(bw_dir):
        log("Skipped.")
        return

    # Process
    log(f"Binarizing (block_size={args.block_size}, offset={args.offset})...")
    t0 = time.time()
    processed = 0
    skipped = 0

    for pg in pages:
        src = pages_dir / pg["filename"]
        dst = bw_dir / pg["filename"]

        if not src.exists():
            log(f"  WARNING: {pg['filename']} not found, skipping")
            continue

        is_cover = pg["type"] in ("cover", "backcover")
        if is_cover and args.skip_covers:
            # Copy cover as-is (color)
            import shutil

            shutil.copy2(src, dst)
            skipped += 1
            continue

        img = cv2.imread(str(src))
        if img is None:
            log(f"  WARNING: Cannot read {pg['filename']}, skipping")
            continue

        binary = binarize_page(img, args.block_size, args.offset)
        cv2.imwrite(str(dst), binary, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        processed += 1

        if processed % 50 == 0:
            log(f"  Processed {processed} pages...")

    elapsed = time.time() - t0
    log(f"  Done. {processed} binarized, {skipped} copied as-is in {elapsed:.1f}s")

    # Save metadata
    meta = {
        "block_size": args.block_size,
        "offset": args.offset,
        "jpeg_quality": args.jpeg_quality,
        "skip_covers": args.skip_covers,
        "pages_processed": processed,
        "pages_skipped": skipped,
    }
    meta_path = bw_dir / "bw_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    # Summary
    log("")
    log("=" * 60)
    log("PHASE 8 COMPLETE")
    log(f"  Output:   {bw_dir}/ ({processed + skipped} images)")
    log(f"  Params:   block_size={args.block_size}, offset={args.offset}")
    log(f"  Metadata: {meta_path}")
    log("=" * 60)


if __name__ == "__main__":
    main()
