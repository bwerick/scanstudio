"""
Page cropping.

Two strategies:
  - Otsu: brightness-based, for book spreads with white pages
  - GrabCut: segmentation-based, for any document color/orientation
"""

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    r = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    r[0] = pts[np.argmin(s)]
    r[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    r[1] = pts[np.argmin(d)]
    r[3] = pts[np.argmax(d)]
    return r


# ── Otsu crop (book spreads) ─────────────────────────────────

def crop_otsu(img: np.ndarray, safety_pct: float = 0.005) -> tuple[np.ndarray, dict]:
    """
    Brightness-based crop using Otsu thresholding.
    Works well for white/cream pages on dark tables.

    Returns:
        (cropped_image, info_dict)
    """
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
            cropped = img[max(0, y - my):min(h, y + bh + my),
                          max(0, x - mx):min(w, x + bw + mx)]
            return cropped, {"method": "otsu", "bbox": [x, y, bw, bh]}

    # Fallback
    mx, my = int(w * 0.02), int(h * 0.02)
    return img[my:h - my, mx:w - mx], {"method": "otsu_fallback"}


def apply_side_trim(img: np.ndarray, left_pct: float,
                     right_pct: float) -> np.ndarray:
    """Trim left and right edges by percentage."""
    h, w = img.shape[:2]
    return img[:, int(w * left_pct):int(w * right_pct)]


# ── GrabCut crop (loose documents) ───────────────────────────

def crop_grabcut(img: np.ndarray,
                  padding_pct: float = 0.03) -> tuple[np.ndarray, dict]:
    """
    GrabCut-based crop with perspective correction.
    Works for any document color against the table.

    Returns:
        (cropped_image, info_dict) where info includes detected corners
    """
    h, w = img.shape[:2]

    # Downscale for speed
    work_w = 640
    scale = work_w / w
    work = cv2.resize(img, (work_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    wh, ww = work.shape[:2]

    # Initialize mask
    mask = np.full((wh, ww), cv2.GC_PR_BGD, dtype=np.uint8)
    border = int(min(wh, ww) * 0.08)
    mask[:border, :] = cv2.GC_BGD
    mask[wh - border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, ww - border:] = cv2.GC_BGD
    cy1, cy2 = int(wh * 0.25), int(wh * 0.75)
    cx1, cx2 = int(ww * 0.25), int(ww * 0.75)
    mask[cy1:cy2, cx1:cx2] = cv2.GC_PR_FGD

    # Run GrabCut
    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(work, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)

    page_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    kernel = np.ones((9, 9), np.uint8)
    page_mask = cv2.morphologyEx(page_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    page_mask = cv2.morphologyEx(page_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(page_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best = None
    for c in contours:
        ratio = cv2.contourArea(c) / (ww * wh)
        if 0.05 < ratio < 0.95:
            best = c
            break

    if best is None:
        mx, my = int(w * 0.1), int(h * 0.1)
        return img[my:h - my, mx:w - mx], {"method": "grabcut_fallback"}

    rect = cv2.minAreaRect(best)
    box = cv2.boxPoints(rect).astype(np.float32)
    box_orig = box / scale
    ordered = order_points(box_orig)
    tl, tr, br, bl = ordered

    out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    pad_x = int(out_w * padding_pct)
    pad_y = int(out_h * padding_pct)

    center = np.mean(ordered, axis=0)
    padded = ordered.copy()
    for i in range(4):
        direction = ordered[i] - center
        length = np.linalg.norm(direction)
        if length > 0:
            padded[i] = ordered[i] + (direction / length) * max(pad_x, pad_y)
    padded[:, 0] = np.clip(padded[:, 0], 0, w - 1)
    padded[:, 1] = np.clip(padded[:, 1], 0, h - 1)

    out_wp = out_w + 2 * pad_x
    out_hp = out_h + 2 * pad_y
    dst = np.array([[0, 0], [out_wp - 1, 0], [out_wp - 1, out_hp - 1],
                     [0, out_hp - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(padded, dst)
    warped = cv2.warpPerspective(img, M, (out_wp, out_hp))

    return warped, {
        "method": "grabcut",
        "corners": ordered.tolist(),
        "angle": float(rect[2]),
    }


# ── Crop preview (for UI) ────────────────────────────────────

def preview_crop(img: np.ndarray, mode: str = "double",
                  side_trim: dict = None,
                  safety_pct: float = 0.005,
                  padding_pct: float = 0.03) -> tuple[np.ndarray, dict]:
    """
    Generate a crop preview without modifying the source image.

    Args:
        img: full keyframe image (BGR)
        mode: "single" or "double"
        side_trim: {"left": 0.15, "right": 0.85} for double mode
        safety_pct: Otsu margin for double mode
        padding_pct: GrabCut padding for single mode

    Returns:
        (cropped_preview, info_dict)
    """
    work = img.copy()

    if mode == "double":
        if side_trim:
            work = apply_side_trim(work, side_trim["left"], side_trim["right"])
        return crop_otsu(work, safety_pct)
    else:
        return crop_grabcut(work, padding_pct)
