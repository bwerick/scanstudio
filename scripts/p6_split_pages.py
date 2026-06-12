#!/usr/bin/env python3
"""
Phase 6: Split Spreads into Individual Pages

Double mode: splits each keyframe at center into left/right pages.
Single mode: copies cropped images directly to pages/ (no split).

Usage:
  python scripts/p6_split_pages.py output/mybook
  python scripts/p6_split_pages.py output/mybook --mode single
"""

import argparse
import json
import sys
import time
import shutil
from pathlib import Path

import cv2

from utils import log, ProjectPaths, check_overwrite_dir, detect_gutter, text_skew


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Split spreads into pages")
    parser.add_argument("output_dir", help="Base output directory")
    parser.add_argument(
        "--mode",
        type=str,
        default="double",
        choices=["single", "double"],
        help="'double' splits spreads, 'single' copies pages as-is (default: double)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG quality for page images (default: 92)",
    )
    parser.add_argument(
        "--no-text-deskew",
        action="store_true",
        help="Skip the per-page fine deskew based on text lines",
    )
    args = parser.parse_args()

    log("=" * 60)
    log(
        f"PHASE 6: {'Split Pages' if args.mode == 'double' else 'Copy Pages (single mode)'}"
    )
    log("=" * 60)

    paths = ProjectPaths(args.output_dir)
    paths.ensure("pages")

    kf_path = paths.json / "keyframes.json"
    if not kf_path.exists():
        log(f"ERROR: {kf_path} not found.")
        sys.exit(1)

    keyframes = json.loads(kf_path.read_text())
    covers = [kf for kf in keyframes if kf.get("is_cover")]
    spreads = [kf for kf in keyframes if not kf.get("is_cover")]
    log(
        f"Loaded {len(keyframes)} keyframes ({len(covers)} covers, {len(spreads)} spreads)"
    )

    if not check_overwrite_dir(paths.pages):
        log("Skipped.")
        return

    t0 = time.time()
    page_list = []
    page_num = 0

    for kf in keyframes:
        img_path = paths.images / kf["filename"]
        if not img_path.exists():
            log(f"  WARNING: {kf['filename']} not found")
            continue

        frame_idx = kf.get("frame_index", page_num)
        is_cover = kf.get("is_cover", False)

        if args.mode == "single":
            # Single-page mode: each keyframe = one page, no split
            page_num += 1
            fn = f"frame{frame_idx:06d}_page.jpg"
            shutil.copy2(img_path, paths.pages / fn)
            page_list.append(
                {
                    "page_num": page_num,
                    "type": "page",
                    "filename": fn,
                    "source": kf["filename"],
                }
            )
        elif is_cover:
            # Double-page mode: cover
            is_last = kf is keyframes[-1]
            ctype = "backcover" if is_last else "cover"
            page_num += 1
            fn = f"frame{frame_idx:06d}_{ctype}.jpg"
            img = cv2.imread(str(img_path))
            cv2.imwrite(
                str(paths.pages / fn),
                img,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )
            page_list.append(
                {
                    "page_num": page_num,
                    "type": ctype,
                    "filename": fn,
                    "source": kf["filename"],
                    "size": f"{img.shape[1]}x{img.shape[0]}",
                }
            )
        else:
            # Double-page mode: split at center
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            # Split at the gutter (spine), not the blind midpoint, so an off-center
            # or translated spread still divides cleanly between pages. A manual
            # override from Phase 4 (fraction of width) wins over auto-detection.
            gutter = kf.get("gutter")
            mid = int(round(w * gutter)) if gutter is not None else detect_gutter(img)
            mid = max(1, min(w - 1, mid))
            left_half = img[:, :mid]
            right_half = img[:, mid:]

            for side, half in [("left", left_half), ("right", right_half)]:
                page_num += 1
                # Fine deskew per page: the spread-level rotation (p5) levels
                # the page edge, but text isn't always parallel to it — page
                # curvature near the spine skews each half differently. Level
                # the text lines themselves.
                skew = 0.0
                if not args.no_text_deskew:
                    skew = text_skew(half)
                    if skew:
                        hh, hw = half.shape[:2]
                        M = cv2.getRotationMatrix2D((hw / 2, hh / 2), skew, 1.0)
                        half = cv2.warpAffine(
                            half,
                            M,
                            (hw, hh),
                            flags=cv2.INTER_LINEAR,
                            borderValue=(255, 255, 255),
                        )
                fn = f"frame{frame_idx:06d}_{side}.jpg"
                cv2.imwrite(
                    str(paths.pages / fn),
                    half,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                )
                page_list.append(
                    {
                        "page_num": page_num,
                        "type": side,
                        "filename": fn,
                        "source": kf["filename"],
                        "size": f"{half.shape[1]}x{half.shape[0]}",
                        "deskew_deg": round(skew, 2),
                    }
                )

        if page_num % 50 == 0:
            log(f"  {page_num} pages...")

    elapsed = time.time() - t0
    log(f"  Done. {len(page_list)} pages in {elapsed:.1f}s")

    (paths.json / "pages.json").write_text(json.dumps(page_list, indent=2))

    log("")
    log("PHASE 6 COMPLETE")
    log(f"  Pages: {paths.pages}/ ({len(page_list)} images)")


if __name__ == "__main__":
    main()
