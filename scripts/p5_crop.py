#!/usr/bin/env python3
"""
Phase 5: Crop Keyframes

Two modes:
  double (default): For book spreads — applies crop bounds + Otsu detection
  single: For loose documents — uses GrabCut to segment page from table,
           handles rotation, works with any page color

Modifies images/ in-place. To restore originals, re-run Phase 3.

Usage:
  python scripts/p5_crop.py output/mybook
  python scripts/p5_crop.py output/mybook --mode single
  python scripts/p5_crop.py output/mybook --mode double --safety-margin 0.005
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from utils import log, ProjectPaths, check_overwrite

# ── Double-page crop (books) ─────────────────────────────────


def crop_double_page(img, safety_pct):
    """Otsu-based crop for book spreads."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((15, 15), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        x, y, bw, bh = cv2.boundingRect(largest)

        if area / (w * h) > 0.35 and bh / h > 0.7:
            mx = int(w * safety_pct)
            my = int(h * safety_pct)
            return (
                img[
                    max(0, y - my) : min(h, y + bh + my),
                    max(0, x - mx) : min(w, x + bw + mx),
                ],
                "otsu",
            )

    # Fallback
    mx, my = int(w * 0.02), int(h * 0.02)
    return img[my : h - my, mx : w - mx], "fallback"


# ── Single-page crop (loose documents) ───────────────────────


def order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    r = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    r[0] = pts[np.argmin(s)]
    r[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    r[1] = pts[np.argmin(d)]
    r[3] = pts[np.argmax(d)]
    return r


def crop_single_page(img, padding_pct=0.01):
    """
    GrabCut-based crop for single loose documents.
    Detects the document against the table, handles rotation,
    and applies perspective correction to produce a straight rectangle.
    """
    h, w = img.shape[:2]

    # Work at lower resolution for GrabCut speed
    work_w = 640
    scale = work_w / w
    work = cv2.resize(img, (work_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    wh, ww = work.shape[:2]

    # Initialize GrabCut mask
    mask = np.full((wh, ww), cv2.GC_PR_BGD, dtype=np.uint8)

    # Border strips = definite background (table)
    border = int(min(wh, ww) * 0.08)
    mask[:border, :] = cv2.GC_BGD
    mask[wh - border :, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, ww - border :] = cv2.GC_BGD

    # Center = probable foreground (page)
    cy1, cy2 = int(wh * 0.25), int(wh * 0.75)
    cx1, cx2 = int(ww * 0.25), int(ww * 0.75)
    mask[cy1:cy2, cx1:cx2] = cv2.GC_PR_FGD

    # Run GrabCut
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(work, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    page_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(
        np.uint8
    )

    # Clean up mask
    kernel = np.ones((9, 9), np.uint8)
    page_mask = cv2.morphologyEx(page_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    page_mask = cv2.morphologyEx(page_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find largest contour (skip whole-frame detections)
    contours, _ = cv2.findContours(
        page_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best = None
    for c in contours:
        ratio = cv2.contourArea(c) / (ww * wh)
        if 0.05 < ratio < 0.95:
            best = c
            break

    if best is None:
        # Fallback: return center 80%
        mx, my = int(w * 0.1), int(h * 0.1)
        return img[my : h - my, mx : w - mx], "grabcut_fallback"

    # Get rotated rectangle
    rect = cv2.minAreaRect(best)
    box = cv2.boxPoints(rect).astype(np.float32)

    # Scale corners back to original image coordinates
    box_orig = box / scale

    # Order the corners
    ordered = order_points(box_orig)
    tl, tr, br, bl = ordered

    # Compute output dimensions
    out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    # Add safety padding to avoid clipping edges
    pad_x = int(out_w * padding_pct)
    pad_y = int(out_h * padding_pct)

    # Expand the source quad outward by padding amount
    center = np.mean(ordered, axis=0)
    padded = ordered.copy()
    for i in range(4):
        direction = ordered[i] - center
        length = np.linalg.norm(direction)
        if length > 0:
            direction = direction / length
            padded[i] = ordered[i] + direction * max(pad_x, pad_y)

    # Clamp to image bounds
    padded[:, 0] = np.clip(padded[:, 0], 0, w - 1)
    padded[:, 1] = np.clip(padded[:, 1], 0, h - 1)

    # Perspective transform to straighten the page
    out_w_padded = out_w + 2 * pad_x
    out_h_padded = out_h + 2 * pad_y

    dst = np.array(
        [
            [0, 0],
            [out_w_padded - 1, 0],
            [out_w_padded - 1, out_h_padded - 1],
            [0, out_h_padded - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(padded, dst)
    warped = cv2.warpPerspective(img, M, (out_w_padded, out_h_padded))

    return warped, "grabcut"


# ── Main ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Crop keyframes")
    parser.add_argument("output_dir", help="Base output directory")
    parser.add_argument(
        "--mode",
        type=str,
        default="double",
        choices=["single", "double"],
        help="Crop mode: 'double' for book spreads, 'single' for loose docs (default: double)",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.005,
        help="Otsu safety margin for double mode (default: 0.005)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.03,
        help="Edge padding for single mode (default: 0.03 = 3%%)",
    )
    parser.add_argument(
        "--no-otsu",
        action="store_true",
        help="Double mode only: skip Otsu, only apply crop bounds",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for cropped images (default: 95)",
    )
    args = parser.parse_args()

    log("=" * 60)
    log(f"PHASE 5: Crop Keyframes ({args.mode} mode)")
    log("=" * 60)

    paths = ProjectPaths(args.output_dir)

    kf_path = paths.json / "keyframes.json"
    if not kf_path.exists():
        log(f"ERROR: {kf_path} not found. Run Phase 3 first.")
        sys.exit(1)

    keyframes = json.loads(kf_path.read_text())
    log(f"Loaded {len(keyframes)} keyframes")

    # Load crop bounds from review (for double mode side trim)
    global_crop = None
    if args.mode == "double":
        rl_path = paths.json / "review_log.json"
        if rl_path.exists():
            rl = json.loads(rl_path.read_text())
            for session in rl.get("sessions", []):
                gc = session.get("global_crop")
                if gc:
                    global_crop = gc
        if global_crop:
            log(
                f"  Crop bounds: L={global_crop['left']:.1%}, R={global_crop['right']:.1%}"
            )

    log("")
    t0 = time.time()
    method_counts = {}

    for i, kf in enumerate(keyframes):
        img_path = paths.images / kf["filename"]
        if not img_path.exists():
            log(f"  WARNING: {kf['filename']} not found")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        is_cover = kf.get("is_cover", False)

        if args.mode == "single":
            # Single-page: GrabCut detection + perspective correction
            if is_cover:
                # Still crop covers
                cropped, method = crop_single_page(img, args.padding)
            else:
                cropped, method = crop_single_page(img, args.padding)

        else:
            # Double-page: crop bounds + Otsu
            # Step 1: Apply side crop bounds
            crop = kf.get("crop_bounds") or global_crop
            if crop and not is_cover:
                h_img, w_img = img.shape[:2]
                img = img[:, int(w_img * crop["left"]) : int(w_img * crop["right"])]

            # Step 2: Otsu page detection
            if not args.no_otsu and not is_cover:
                cropped, method = crop_double_page(img, args.safety_margin)
            else:
                cropped = img
                method = "bounds_only" if crop else "none"

        method_counts[method] = method_counts.get(method, 0) + 1

        # Save in-place
        cv2.imwrite(
            str(img_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
        )

        if (i + 1) % 25 == 0 or i == len(keyframes) - 1:
            log(f"  {i + 1}/{len(keyframes)} — {kf['filename']} [{method}]")

    elapsed = time.time() - t0
    log(f"\nDone. {len(keyframes)} images in {elapsed:.1f}s")
    for method, count in sorted(method_counts.items()):
        log(f"  {method}: {count}")

    log("")
    log("PHASE 5 COMPLETE")
    log(f"  To restore originals, re-run Phase 3")


if __name__ == "__main__":
    main()
