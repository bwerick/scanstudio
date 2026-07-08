#!/usr/bin/env python3
"""
Phase 5: Crop Keyframes

Two modes:
  double (default): For book spreads — a manual crop box from Phase-4 review
           (crop_quad, propagated forward from the nearest earlier correction)
           wins; otherwise crop bounds + page-mask auto detection.
  single: For loose documents — a per-frame manual crop_quad wins; otherwise
           GrabCut segments the page from the table. Handles rotation, works
           with any page color.

Modifies images/ in-place. To restore originals, re-run Phase 3.

Usage:
  python scripts/p5_crop.py output/mybook
  python scripts/p5_crop.py output/mybook --mode single
  python scripts/p5_crop.py output/mybook --mode double --safety-margin 0.02
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from utils import (
    log,
    ProjectPaths,
    check_overwrite,
    page_mask,
    resolve_rotation,
    resolve_crop_quad,
)

# Breathing room kept around the detected spread bounds, as a fraction of the
# frame. Bumped up from a hair-thin 0.005 because a tight crop that clips real
# page content is far worse than a little extra table in the margin (the split
# step finds the gutter regardless, and the binarizer drops the border). Phase 4
# imports this so its split preview matches, and it is the per-frame default that
# a keyframe's own ``crop_margin`` override replaces.
DEFAULT_SAFETY_MARGIN = 0.02

# ── Double-page crop (books) ─────────────────────────────────


def _spread_tilt(mask, max_deg=8.0):
    """Estimate spread rotation (degrees) from the mask's top edge.

    The top edge of a flat-lying spread is a near-straight line; its slope is
    the rotation. A robust (Huber) line fit ignores the finger/notch outliers,
    and columns where the page runs off the top of the frame are skipped since
    their "top" is the frame border, not the page. Returns 0 when the estimate
    is implausibly large (mask too ragged to trust).
    """
    h, w = mask.shape
    xs, ys = [], []
    for x in range(0, w, 4):
        col = np.where(mask[:, x] > 0)[0]
        if len(col) and 2 < col[0] < h * 0.45:
            xs.append(x)
            ys.append(int(col[0]))
    if len(xs) < 20:
        return 0.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    angle = float(np.degrees(np.arctan2(vy, vx)))
    return angle if abs(angle) <= max_deg else 0.0


def _mask_bounds(mask, min_frac=0.5):
    """Robust page bounds ``(x, y, w, h)``, ignoring thin protrusions.

    ``cv2.boundingRect`` spans the blob's maximum extent, so a page underneath
    sticking out past the edge (or a finger) drags the crop outward and leaves
    a band of table along the whole side. Instead, keep only the columns/rows
    whose page coverage reaches ``min_frac`` of the peak coverage — a
    protrusion spans a small fraction of the page height/width, so the bounds
    snap to the page proper.
    """
    cols = (mask > 0).sum(axis=0)
    rows = (mask > 0).sum(axis=1)
    xs = np.where(cols >= cols.max() * min_frac)[0]
    ys = np.where(rows >= rows.max() * min_frac)[0]
    if not len(xs) or not len(ys):
        return cv2.boundingRect(mask)
    return int(xs[0]), int(ys[0]), int(xs[-1] + 1 - xs[0]), int(ys[-1] + 1 - ys[0])


def crop_double_page(img, safety_pct, rotation_override=None):
    """Deskew and isolate a book spread from a tinted table.

    Replaces grayscale Otsu (which merges cream pages into light-brown wood)
    with an HSV page mask, measures the spread's tilt from the mask's top edge,
    rotates to deskew, then tight-crops to the page bounds. Robust to rotation
    and translation of the spread within the frame. The downstream split step
    finds the gutter on the result, so this only has to straighten and frame
    the spread.

    ``rotation_override`` (degrees) replaces the auto-measured tilt when the
    operator has corrected the deskew in Phase 4. p4's split preview calls this
    with identical arguments, so the cropped result here matches what was shown.

    Returns ``(cropped, method, (x0, y0, crop_w, crop_h))`` — the crop's origin
    and size in deskewed-frame pixels — so a gutter measured on the crop, or
    the crop box itself, can be mapped back onto the original frame.
    """
    h, w = img.shape[:2]
    mask = page_mask(img)

    if cv2.countNonZero(mask) < 0.2 * w * h:
        # No page-sized bright region found — leave the frame essentially as-is.
        mx, my = int(w * 0.02), int(h * 0.02)
        return (
            img[my : h - my, mx : w - mx],
            "fallback",
            (mx, my, w - 2 * mx, h - 2 * my),
        )

    angle = rotation_override if rotation_override is not None else _spread_tilt(mask)
    if abs(angle) > 0.2:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
        )
        mask = page_mask(img)

    x, y, bw, bh = _mask_bounds(mask)
    mx, my = int(w * safety_pct), int(h * safety_pct)
    x0, x1 = max(0, x - mx), min(w, x + bw + mx)
    y0, y1 = max(0, y - my), min(h, y + bh + my)
    return (
        img[y0:y1, x0:x1],
        "hsv_deskew",
        (x0, y0, x1 - x0, y1 - y0),
    )


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


def detect_page_quad(img):
    """GrabCut-segment a loose page from the table; return its 4 corners.

    Returns an ordered ``(tl, tr, br, bl)`` float32 array in full-image pixel
    coordinates (the page's minimum-area rectangle), or ``None`` when no
    confident page-sized region is found. Both the automatic crop
    (``crop_single_page``) and the Phase-4 manual crop editor seed from this, so
    the box the operator tunes starts exactly where the detector landed.
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

    page = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(
        np.uint8
    )

    # Clean up mask
    kernel = np.ones((9, 9), np.uint8)
    page = cv2.morphologyEx(page, cv2.MORPH_CLOSE, kernel, iterations=2)
    page = cv2.morphologyEx(page, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find largest contour (skip whole-frame detections)
    contours, _ = cv2.findContours(page, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best = None
    for c in contours:
        ratio = cv2.contourArea(c) / (ww * wh)
        if 0.05 < ratio < 0.95:
            best = c
            break

    if best is None:
        return None

    # Rotated rectangle, scaled back to original image coordinates
    box = cv2.boxPoints(cv2.minAreaRect(best)).astype(np.float32) / scale
    return order_points(box)


def crop_to_quad(img, quad, padding_pct=0.01):
    """Perspective-straighten the page bounded by ``quad`` into a rectangle.

    ``quad`` is an ordered ``(tl, tr, br, bl)`` array in image pixels — either a
    detector result or a Phase-4 manual override. The quad is expanded outward
    by ``padding_pct`` (so edges aren't clipped) and warped to a straight,
    axis-aligned rectangle.
    """
    ordered = order_points(np.asarray(quad, dtype=np.float32))
    tl, tr, br, bl = ordered

    # Output dimensions from the quad's side lengths
    out_w = max(1, int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    out_h = max(1, int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))

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
            padded[i] = ordered[i] + (direction / length) * max(pad_x, pad_y)

    # Perspective transform to straighten the page. The source quad may poke
    # past the frame (a manual box drawn to the image edge, or the padding
    # expansion): clamping its corners would squish the content, so instead
    # let the warp sample out-of-bounds pixels and fill them with white.
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
    return cv2.warpPerspective(
        img,
        M,
        (out_w_padded, out_h_padded),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def crop_single_page(img, padding_pct=0.01):
    """Auto-crop a single loose document: detect the page, then straighten it.

    Detects the document against the table (handling rotation) and applies
    perspective correction to produce a straight rectangle. Falls back to the
    center 80% when no confident page region is found.
    """
    quad = detect_page_quad(img)
    if quad is None:
        h, w = img.shape[:2]
        mx, my = int(w * 0.1), int(h * 0.1)
        return img[my : h - my, mx : w - mx], "grabcut_fallback"
    return crop_to_quad(img, quad, padding_pct), "grabcut"


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
        default=DEFAULT_SAFETY_MARGIN,
        help="Default breathing room around the auto-detected spread for double "
        f"mode, as a fraction of the frame (default: {DEFAULT_SAFETY_MARGIN}). "
        "Only used on frames without a manual crop box (crop_quad, drawn with G "
        "in Phase 4); a legacy per-keyframe crop_margin override still takes "
        "precedence over this default.",
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
            # Single-page: a Phase-4 manual crop override (4 corners as
            # fractions of the frame) wins; otherwise GrabCut auto-detection.
            quad = kf.get("crop_quad")
            if quad:
                h_img, w_img = img.shape[:2]
                quad_px = np.array(
                    [[p[0] * w_img, p[1] * h_img] for p in quad], dtype=np.float32
                )
                cropped = crop_to_quad(img, quad_px, args.padding)
                method = "manual_quad"
            else:
                cropped, method = crop_single_page(img, args.padding)

        else:
            # Double-page. A manual crop box from Phase-4 review wins outright:
            # it propagates forward from the nearest earlier correction (the
            # rig doesn't move between page turns), is drawn on the raw frame,
            # and encodes position, size, and tilt at once — so it replaces the
            # side-bounds + auto-detection path entirely for this frame.
            quad = None if is_cover else resolve_crop_quad(keyframes, i)
            if quad is not None:
                h_img, w_img = img.shape[:2]
                quad_px = np.array(
                    [[p[0] * w_img, p[1] * h_img] for p in quad], dtype=np.float32
                )
                cropped = crop_to_quad(img, quad_px, 0.0)
                method = "manual_quad" if kf.get("crop_quad") else "inherited_quad"
            else:
                # Auto path: crop bounds + page-mask detection
                # Step 1: Apply side crop bounds
                crop = kf.get("crop_bounds") or global_crop
                if crop and not is_cover:
                    h_img, w_img = img.shape[:2]
                    img = img[
                        :, int(w_img * crop["left"]) : int(w_img * crop["right"])
                    ]

                # Step 2: Otsu page detection
                if not args.no_otsu and not is_cover:
                    # Manual deskew corrections propagate forward to later
                    # spreads (the rig doesn't move between page turns). A
                    # per-spread crop margin (legacy override from older review
                    # sessions) is a one-off, so it does NOT propagate.
                    rot = resolve_rotation(keyframes, i)
                    margin = kf.get("crop_margin", args.safety_margin)
                    cropped, method, _ = crop_double_page(img, margin, rot)
                else:
                    cropped = img
                    method = "bounds_only" if crop else "none"

        method_counts[method] = method_counts.get(method, 0) + 1

        # Save in-place. High-quality JPEG: this is the binarizer's eventual
        # source, so keep one near-lossless generation rather than stacking many.
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
