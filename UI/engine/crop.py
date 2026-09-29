"""
Page cropping.

Two strategies:
  - Otsu: brightness-based, for book spreads with white pages
  - GrabCut: segmentation-based, for any document color/orientation
"""

import cv2
import numpy as np

from .geometry import (
    order_quad, quad_from_rect, rect_from_quad, quad_to_px, quad_to_norm,
    quad_size, warp_quad, remap_gutter, normalize_quad, normalize_gutter,
    gutter_endpoints, gutter_from_points, split_quads, resolve_crop_quad,
    resolve_gutter_chain, DEFAULT_GUTTER,
)

# Back-compat aliases: geometry.py owns the implementations.
rect_to_quad = quad_from_rect
quad_to_rect = rect_from_quad
quad_px = quad_to_px
resolve_gutter = resolve_gutter_chain


def crop_to_quad(img, quad_pixels, pad_px=0):
    """Warp a pixel-space quad to an upright rectangle."""
    return warp_quad(img, quad_pixels, pad_px)


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
            cx1, cx2 = max(0, x - mx), min(w, x + bw + mx)
            cropped = img[max(0, y - my):min(h, y + bh + my), cx1:cx2]
            return cropped, {"method": "otsu", "bbox": [x, y, bw, bh],
                             "x_range": (cx1 / w, cx2 / w)}

    # Fallback
    mx, my = int(w * 0.02), int(h * 0.02)
    return img[my:h - my, mx:w - mx], {"method": "otsu_fallback",
                                        "x_range": (mx / w, (w - mx) / w)}


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


# ══════════════════════════════════════════════════════════════
# Quad geometry — the single crop representation
#
# A crop is 4 normalized corners in (tl, tr, br, bl) order. An axis-aligned
# rectangle is just a quad, so this generalizes the old {x1,y1,x2,y2} with
# no loss and lets a tilted book be cropped without throwing away the tilt.
#
# The gutter is stored RELATIVE TO THE BOX as two fractions: where the spine
# crosses the box's top edge and its bottom edge. Two consequences:
#   * it has its own angle, independent of the box's tilt (a fanned page
#     stack skews the outer edge relative to the spine, so one shared angle
#     can never satisfy both the crop and the split);
#   * it rides along when the box moves, and needs NO remapping when the
#     crop is applied — the coordinate-space bug that shifted splits into
#     the text cannot occur in this representation.
# ══════════════════════════════════════════════════════════════

CORNER_ORDER = ("tl", "tr", "br", "bl")


# ── Manual crop ──────────────────────────────────────────────

def crop_manual(img: np.ndarray, shape) -> tuple[np.ndarray, dict]:
    """
    Crop to an explicit shape in normalized coords.

    Accepts a quad (4 corners) or a legacy axis-aligned rect
    {"x1","y1","x2","y2"} — the rect is widened to a quad so old projects
    keep working. A quad is warped, so tilt is corrected for free.
    """
    h, w = img.shape[:2]
    if isinstance(shape, dict):
        quad = quad_from_rect(shape)
        kind = "manual_rect"
    else:
        quad = order_quad(shape)
        kind = "manual"

    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    if (max(xs) - min(xs)) * w < 10 or (max(ys) - min(ys)) * h < 10:
        return img, {"method": "manual_invalid", "x_range": (0.0, 1.0)}

    out = warp_quad(img, quad_to_px(quad, w, h))
    return out, {"method": kind, "quad": quad,
                 "x_range": (float(min(xs)), float(max(xs)))}


def _detect_crop(img: np.ndarray, mode: str = "double",
                  side_trim: dict = None,
                  safety_pct: float = 0.005) -> dict:
    """
    Auto-detect the crop region and return it as a normalized quad plus
    its bounding rect. Double mode uses Otsu (the page dominates a bright
    frame); single mode uses line detection, which keeps real tilt.
    """
    H, W = img.shape[:2]
    lo, hi = 0.0, 1.0
    work = img
    if mode == "double" and side_trim:
        lo, hi = float(side_trim["left"]), float(side_trim["right"])
        work = apply_side_trim(img, lo, hi)
    span = hi - lo

    def _out(quad, source, **extra):
        # Map the sub-image quad back onto the full frame's x-axis.
        q = [[lo + float(x) * span, float(y)] for x, y in quad]
        r = rect_from_quad(q)
        r.update({"quad": order_quad(q), "source": source, **extra})
        return r

    if mode == "double":
        h, w = work.shape[:2]
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = np.ones((15, 15), np.uint8)
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)
        cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) / (w * h) > 0.35:
                x, y, bw, bh = cv2.boundingRect(c)
                mx, my = int(w * safety_pct), int(h * safety_pct)
                x, y = max(0, x - mx), max(0, y - my)
                bw = min(w - x, bw + 2 * mx)
                bh = min(h - y, bh + 2 * my)
                return _out(quad_from_rect({"x1": x / w, "y1": y / h,
                                             "x2": (x + bw) / w,
                                             "y2": (y + bh) / h}), "otsu")
        return _out(quad_from_rect({"x1": 0.02, "y1": 0.02,
                                     "x2": 0.98, "y2": 0.98}), "fallback")

    quad, dbg = detect_quad(work)
    if quad is None:
        return _out(quad_from_rect({"x1": 0.05, "y1": 0.05,
                                     "x2": 0.95, "y2": 0.95}), "fallback")
    h, w = work.shape[:2]
    return _out(quad_to_norm(quad, w, h), "lines",
                lines_used=dbg.get("lines_used", 0))


def detect_crop_quad(img: np.ndarray, mode: str = "double",
                      side_trim: dict = None,
                      safety_pct: float = 0.005) -> dict:
    """
    Auto-detect the crop as a quad, so the UI can seed its drag handles
    from the algorithm's guess. Includes the bounding-rect keys for
    callers that only want a rectangle.
    """
    return _detect_crop(img, mode, side_trim, safety_pct)


def detect_crop_rect(img: np.ndarray, mode: str = "double",
                      side_trim: dict = None,
                      safety_pct: float = 0.005) -> dict:
    """Bounding-rect view of `detect_crop_quad`, for legacy callers."""
    return detect_crop_quad(img, mode, side_trim, safety_pct)


# ── Crop preview (for UI) ────────────────────────────────────

def crop_lines(img: np.ndarray, padding_pct: float = 0.004
               ) -> tuple[np.ndarray, dict]:
    """
    Line-based crop. Handles any page color, corrects rotation, and rejects
    hands geometrically. Falls back to GrabCut if no valid quad is found.
    """
    quad, dbg = detect_quad(img)
    if quad is None:
        cropped, info = crop_grabcut(img, padding_pct=max(padding_pct, 0.01))
        info["method"] = "lines_fallback_grabcut"
        return cropped, info
    _W = img.shape[1]
    return warp(img, quad, pad_pct=padding_pct), {
        "method": "lines",
        "x_range": (float(quad[:, 0].min() / _W), float(quad[:, 0].max() / _W)),
        "lines_used": dbg.get("lines_used", 0),
        "cue_scores": {k: round(v, 3) for k, v in dbg.get("scores", {}).items()},
    }


def preview_crop(img: np.ndarray, mode: str = "double",
                  side_trim: dict = None,
                  safety_pct: float = 0.005,
                  padding_pct: float = 0.004,
                  method: str = "auto",
                  manual_rect=None) -> tuple[np.ndarray, dict]:
    """
    Generate a crop preview without modifying the source image.

    Args:
        img: full keyframe image (BGR)
        mode: "single" or "double"
        side_trim: {"left": 0.15, "right": 0.85} applied first in double mode
        method: "auto" | "lines" | "otsu" | "grabcut"

    Returns:
        (cropped_preview, info_dict)
    """
    # An explicit quad (or legacy rect) always wins — it is the override.
    if manual_rect:
        return crop_manual(img, manual_rect)

    work = img.copy()

    trim_lo, trim_hi = 0.0, 1.0
    if mode == "double" and side_trim:
        trim_lo, trim_hi = side_trim["left"], side_trim["right"]
        work = apply_side_trim(work, trim_lo, trim_hi)

    def _compose(res):
        """Re-express the inner crop's x_range in ORIGINAL image coordinates."""
        out, info = res
        if "x_range" in info:
            lo, hi = info["x_range"]
            span = trim_hi - trim_lo
            info["x_range"] = (trim_lo + lo * span, trim_lo + hi * span)
        return out, info

    if method == "otsu":
        return _compose(crop_otsu(work, safety_pct))
    if method == "grabcut":
        return _compose(crop_grabcut(work, padding_pct))
    if method == "lines":
        return _compose(crop_lines(work, padding_pct))

    # auto: Otsu is proven for book spreads (the page dominates a bright frame);
    # line detection handles loose documents of any color/orientation.
    if mode == "double":
        return _compose(crop_otsu(work, safety_pct))
    return _compose(crop_lines(work, padding_pct))


# ══════════════════════════════════════════════════════════════
# Line-based detection (v3)
#
# Combines three cues, auto-weighted per image by how bimodal each is:
#   texture   - wood grain has high local variance, paper is smooth
#   chroma    - Lab a/b distance from the sampled table color
#   luminance - L distance from the table
# Then restricts Canny to a band around the rough mask boundary (killing
# false edges from page content) and fits the four sides with Hough lines.
# Hands cannot vote for a long straight line, so they self-reject.
# ══════════════════════════════════════════════════════════════


def _local_std(gray, k=15):
    f = gray.astype(np.float32)
    mu = cv2.blur(f, (k, k)); mu2 = cv2.blur(f * f, (k, k))
    return np.sqrt(np.maximum(mu2 - mu * mu, 0))


def _norm8(x):
    x = x - x.min(); m = x.max()
    return (x / m * 255).astype(np.uint8) if m > 1e-6 else np.zeros_like(x, np.uint8)


def _otsu_score(u8):
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    f = u8.astype(np.float32)
    lo, hi = f[f <= t], f[f > t]
    if len(lo) < 10 or len(hi) < 10:
        return 0.0
    w0, w1 = len(lo) / f.size, len(hi) / f.size
    return (w0 * w1 * (lo.mean() - hi.mean()) ** 2) / (f.var() + 1e-6)


def skin_mask(bgr, min_area_frac=0.01):
    """Conservative: only large contiguous blobs. Wood grain speckle is dropped."""
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(y, np.array([0, 138, 80], np.uint8),
                       np.array([255, 170, 125], np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8), iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    min_area = min_area_frac * m.size
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lbl == i] = 255
    return cv2.dilate(out, np.ones((11, 11), np.uint8), iterations=2)


def foreground_mask(bgr, skin):
    """Adaptive multi-cue foreground. Returns (mask, info)."""
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tex = _local_std(gray)

    b = int(min(h, w) * 0.04)
    bm = np.zeros((h, w), bool)
    bm[:b, :] = 1; bm[-b:, :] = 1; bm[:, :b] = 1; bm[:, -b:] = 1
    bm[skin > 0] = 0                      # never sample hands as table
    if bm.sum() < 100:
        bm[:b, :] = 1; bm[-b:, :] = 1

    mu = lab[bm].mean(axis=0)
    tmu = tex[bm].mean()

    feats = {
        "texture": np.clip(tmu - tex, 0, None),
        "chroma":  np.sqrt((lab[:, :, 1] - mu[1]) ** 2 + (lab[:, :, 2] - mu[2]) ** 2),
        "lumin":   np.abs(lab[:, :, 0] - mu[0]),
    }

    # weight each cue by its bimodality on this specific image
    score, u8 = {}, {}
    for k, v in feats.items():
        u8[k] = _norm8(v)
        score[k] = _otsu_score(u8[k])

    tot = sum(score.values()) + 1e-6
    combo = np.zeros((h, w), np.float32)
    for k in feats:
        combo += (score[k] / tot) * u8[k].astype(np.float32)

    _, fg = cv2.threshold(_norm8(combo), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg[skin > 0] = 0

    k = np.ones((13, 13), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=4)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=2)

    # fill holes (page content shouldn't punch through)
    ff = fg.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 255)
    fg = fg | cv2.bitwise_not(ff)

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fg = (lbl == big).astype(np.uint8) * 255

    return fg, {"scores": score}


def boundary_edges(bgr, fg, skin):
    h, w = fg.shape
    band_px = max(10, int(min(h, w) * 0.05))
    k = np.ones((band_px, band_px), np.uint8)
    band = cv2.dilate(fg, k) - cv2.erode(fg, k)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 60, 60)
    edges = cv2.Canny(gray, 25, 80)
    edges[band == 0] = 0
    edges[skin > 0] = 0
    return edges, band


def _fit_side_lines(edges, shape):
    h, w = shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=50,
                            minLineLength=int(min(h, w) * 0.20),
                            maxLineGap=int(min(h, w) * 0.15))
    if lines is None:
        return {}
    V, Hh = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if ang < 18 or ang > 162: Hh.append((x1, y1, x2, y2))
        elif 72 < ang < 108:      V.append((x1, y1, x2, y2))

    out = {}
    if V:
        def xm(l):
            x1, y1, x2, y2 = l
            return (x1 + x2) / 2 if y2 == y1 else x1 + (h/2 - y1)/(y2 - y1)*(x2 - x1)
        s = sorted(V, key=xm); out["left"], out["right"] = s[0], s[-1]
    if Hh:
        def ym(l):
            x1, y1, x2, y2 = l
            return (y1 + y2) / 2 if x2 == x1 else y1 + (w/2 - x1)/(x2 - x1)*(y2 - y1)
        s = sorted(Hh, key=ym); out["top"], out["bottom"] = s[0], s[-1]
    return out


def _isect(l1, l2):
    x1,y1,x2,y2 = l1; x3,y3,x4,y4 = l2
    d = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(d) < 1e-6: return None
    px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4))/d
    py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4))/d
    return (px, py)


def detect_quad(bgr, work_w=900):
    H, W = bgr.shape[:2]
    scale = work_w / W
    work = cv2.resize(bgr, (work_w, int(H*scale)), interpolation=cv2.INTER_AREA)
    h, w = work.shape[:2]

    sk = skin_mask(work)
    fg, info = foreground_mask(work, sk)
    edges, band = boundary_edges(work, fg, sk)
    sides = _fit_side_lines(edges, (h, w))

    dbg = {"work": work, "skin": sk, "fg": fg, "edges": edges,
           "sides": sides, "scale": scale, **info}

    ys, xs = np.where(fg > 0)
    if len(xs) == 0:
        return None, dbg
    bx1, bx2, by1, by2 = xs.min(), xs.max(), ys.min(), ys.max()
    bbox_area = max(1, (bx2-bx1)*(by2-by1))

    # Arbitration: masks leak OUTWARD (closing, shadows); Hough lines are precise.
    # Allow a line to TIGHTEN the bbox generously, but only LOOSEN it slightly.
    tighten_max = 0.30 * min(h, w)
    loosen_max  = 0.04 * min(h, w)
    def pick(key, bedge, axis, inward):
        if key not in sides:
            return None
        x1, y1, x2, y2 = sides[key]
        pos = (x1 + x2) / 2 if axis == 0 else (y1 + y2) / 2
        delta = (pos - bedge) * inward        # >0 means tighter
        if 0 <= delta <= tighten_max:  return sides[key]
        if -loosen_max <= delta < 0:   return sides[key]
        return None

    L = pick("left",   bx1, 0, +1) or (bx1, 0, bx1, h)
    R = pick("right",  bx2, 0, -1) or (bx2, 0, bx2, h)
    T = pick("top",    by1, 1, +1) or (0, by1, w, by1)
    B = pick("bottom", by2, 1, -1) or (0, by2, w, by2)
    used = sum(1 for k, v in (("left",L),("right",R),("top",T),("bottom",B))
               if k in sides and v is sides[k])
    dbg["lines_used"] = used

    c = [_isect(T,L), _isect(T,R), _isect(B,R), _isect(B,L)]
    if any(x is None for x in c):
        return None, dbg
    quad = np.array(c, np.float32)
    quad[:,0] = np.clip(quad[:,0], 0, w-1); quad[:,1] = np.clip(quad[:,1], 0, h-1)

    area = cv2.contourArea(quad)
    if area < 0.08*w*h or not (0.40 < area/bbox_area < 1.6):
        return None, dbg

    dbg["quad_work"] = quad.copy()
    return quad/scale, dbg


def warp(bgr, quad, pad_pct=0.004):
    o = order_points(quad); tl,tr,br,bl = o
    ow = int(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl)))
    oh = int(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr)))
    px, py = int(ow*pad_pct), int(oh*pad_pct)
    c = o.mean(0); p = o.copy()
    for i in range(4):
        v = o[i]-c; n = np.linalg.norm(v)
        if n > 0: p[i] = o[i] + v/n*max(px,py)
    p[:,0] = np.clip(p[:,0],0,bgr.shape[1]-1); p[:,1] = np.clip(p[:,1],0,bgr.shape[0]-1)
    ow += 2*px; oh += 2*py
    dst = np.array([[0,0],[ow-1,0],[ow-1,oh-1],[0,oh-1]], np.float32)
    return cv2.warpPerspective(bgr, cv2.getPerspectiveTransform(p,dst), (ow,oh))
