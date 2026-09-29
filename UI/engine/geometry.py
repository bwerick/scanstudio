"""
Geometry shared by crop and split.

Two representations, both normalized to the frame (0-1) so they survive
resizing:

  quad    4 corners in (tl, tr, br, bl) order. A rotated rectangle is a
          quad, and so is an axis-aligned one — the general case costs
          nothing and buys tilt correction for free.

  gutter  {"pos": 0..1, "angle": degrees}. `pos` is the fraction across
          the box's width; `angle` is the spine's tilt *relative to the
          box's vertical edges*, so 0 means "parallel to the page edges".

Keeping the gutter's angle separate from the box's is the whole point.
An open book fans: the outer edge of the page stack is not parallel to
the spine. One shared angle cannot satisfy both, so the crop and the
split each get their own.
"""

import cv2
import numpy as np


# ── Quad basics ──────────────────────────────────────────────

def order_quad(pts) -> list:
    """Order 4 points as (top-left, top-right, bottom-right, bottom-left)."""
    p = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    s, d = p.sum(axis=1), np.diff(p, axis=1).ravel()
    return [
        p[np.argmin(s)].tolist(),   # tl: smallest x+y
        p[np.argmin(d)].tolist(),   # tr: smallest y-x
        p[np.argmax(s)].tolist(),   # br: largest x+y
        p[np.argmax(d)].tolist(),   # bl: largest y-x
    ]


def quad_from_rect(rect: dict) -> list:
    """Axis-aligned {x1,y1,x2,y2} → quad. Used to migrate legacy crop_rect."""
    x1, x2 = sorted((float(rect["x1"]), float(rect["x2"])))
    y1, y2 = sorted((float(rect["y1"]), float(rect["y2"])))
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def rect_from_quad(quad) -> dict:
    """Bounding box of a quad, for coarse comparisons and legacy callers."""
    p = np.asarray(quad, dtype=np.float64)
    return {"x1": float(p[:, 0].min()), "y1": float(p[:, 1].min()),
            "x2": float(p[:, 0].max()), "y2": float(p[:, 1].max())}


def quad_to_px(quad, w: int, h: int) -> np.ndarray:
    return np.asarray(quad, dtype=np.float32) * np.array([w, h], dtype=np.float32)


def quad_to_norm(quad_px, w: int, h: int) -> list:
    p = np.asarray(quad_px, dtype=np.float64) / np.array([w, h], dtype=np.float64)
    return [[float(x), float(y)] for x, y in p]


def quad_size(quad_px) -> tuple:
    """Output (width, height) for a warp: the longer of each opposing pair."""
    tl, tr, br, bl = [np.asarray(p, dtype=np.float64) for p in quad_px]
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    return max(1, int(round(w))), max(1, int(round(h)))


def warp_quad(img: np.ndarray, quad_px, pad_px: int = 0) -> np.ndarray:
    """Perspective-warp the quad to an upright rectangle."""
    q = np.asarray(quad_px, dtype=np.float32).reshape(4, 2)
    if pad_px:
        c = q.mean(axis=0)
        v = q - c
        n = np.linalg.norm(v, axis=1, keepdims=True)
        q = q + np.divide(v, np.maximum(n, 1e-6)) * pad_px
    ow, oh = quad_size(q)
    dst = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]],
                   dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(q, dst), (ow, oh))


# ── Gutter ───────────────────────────────────────────────────

DEFAULT_GUTTER = {"top": 0.5, "bot": 0.5}


def normalize_gutter(g) -> dict:
    """
    Coerce any stored gutter to {"top", "bot"} — the fraction along the
    box's top edge and along its bottom edge.

    Two fractions carry the same two degrees of freedom as a position plus
    an angle, but need no trigonometry and map straight onto dragging the
    line's endpoints. Equal values mean an untilted spine, which is what a
    legacy scalar becomes.
    """
    if g is None:
        return None
    if isinstance(g, (int, float)):
        return {"top": float(g), "bot": float(g)}
    if "top" in g and "bot" in g:
        return {"top": float(g["top"]), "bot": float(g["bot"])}
    if "pos" in g:                      # {pos, angle} from an interim save
        pos, ang = float(g["pos"]), float(g.get("angle", 0.0))
        half = np.tan(np.radians(ang)) * 0.5
        return {"top": pos + half, "bot": pos - half}
    return dict(DEFAULT_GUTTER)


def gutter_endpoints(quad, gutter) -> list:
    """Where the spine meets the box's top and bottom edges."""
    g = normalize_gutter(gutter) or DEFAULT_GUTTER
    tl, tr, br, bl = [np.asarray(p, dtype=np.float64) for p in quad]
    t = float(np.clip(g["top"], 0.0, 1.0))
    b = float(np.clip(g["bot"], 0.0, 1.0))
    return [(tl + t * (tr - tl)).tolist(), (bl + b * (br - bl)).tolist()]


def gutter_from_points(quad, p_top, p_bot) -> dict:
    """Two dragged endpoints -> {top, bot}, each clamped inside the box."""
    tl, tr, br, bl = [np.asarray(p, dtype=np.float64) for p in quad]

    def frac(p, a, b):
        e = b - a
        L2 = float(e @ e)
        if L2 < 1e-12:
            return 0.5
        return float(np.clip(((np.asarray(p, float) - a) @ e) / L2, 0.02, 0.98))

    return {"top": frac(p_top, tl, tr), "bot": frac(p_bot, bl, br)}


def gutter_angle(quad, gutter) -> float:
    """Spine tilt in degrees relative to the box's vertical, for display."""
    (tx, ty), (bx, by) = gutter_endpoints(quad, gutter)
    tl, tr, br, bl = [np.asarray(p, dtype=np.float64) for p in quad]
    v_box = (bl + br) / 2.0 - (tl + tr) / 2.0
    v_cut = np.array([bx - tx, by - ty])
    if np.linalg.norm(v_box) < 1e-9 or np.linalg.norm(v_cut) < 1e-9:
        return 0.0
    a = np.degrees(np.arctan2(v_cut[1], v_cut[0])
                   - np.arctan2(v_box[1], v_box[0]))
    return float((a + 180.0) % 360.0 - 180.0)


def split_quads(quad, gutter) -> tuple:
    """
    Cut a spread along the gutter into (left_quad, right_quad).

    Each half is a quad, so a leaning spine still yields two upright pages
    after warping — no lost content and no black wedges.
    """
    tl, tr, br, bl = [list(map(float, p)) for p in quad]
    g_top, g_bot = gutter_endpoints(quad, gutter)
    return [tl, g_top, g_bot, bl], [g_top, tr, br, g_bot]


def remap_gutter(gutter, x_range: tuple) -> dict:
    """
    Re-express a gutter after a crop that kept `x_range` of the width.

    The spine is a physical feature and cropping is only a framing
    decision, so the gutter has to survive it. Both fractions rescale,
    which preserves the tilt.
    """
    g = normalize_gutter(gutter)
    if g is None:
        return None
    x1, x2 = x_range
    span = x2 - x1
    if span <= 1e-6:
        return g
    return {
        "top": float(np.clip((g["top"] - x1) / span, 0.02, 0.98)),
        "bot": float(np.clip((g["bot"] - x1) / span, 0.02, 0.98)),
    }


# ── Validation / inheritance ─────────────────────────────────

def normalize_quad(value):
    """Coerce a stored value to a clean quad, or None if unusable."""
    if value is None:
        return None
    try:
        p = np.asarray(value, dtype=np.float64).reshape(4, 2)
    except Exception:
        if isinstance(value, dict) and {"x1", "y1", "x2", "y2"} <= set(value):
            return quad_from_rect(value)
        return None
    if not np.isfinite(p).all():
        return None
    return order_quad(p)


def resolve_field(keyframes: list, idx: int, key: str):
    """
    Value in effect at `idx`: its own, else the nearest one set before it.

    Returns (value, anchor_idx). Books drift slowly, so one correction
    should carry forward until another replaces it.
    """
    if idx < 0 or idx >= len(keyframes):
        return None, None
    if keyframes[idx].get(key) is not None:
        return keyframes[idx][key], idx
    for j in range(idx - 1, -1, -1):
        if keyframes[j].get(key) is not None:
            return keyframes[j][key], j
    return None, None


def resolve_crop_quad(keyframes: list, idx: int):
    q, a = resolve_field(keyframes, idx, "crop_quad")
    return normalize_quad(q), a


def resolve_gutter_chain(keyframes: list, idx: int):
    """Gutter in effect at `idx`, accepting the legacy `gutter_pct` float."""
    g, a = resolve_field(keyframes, idx, "gutter")
    if g is None:
        g, a = resolve_field(keyframes, idx, "gutter_pct")
    return normalize_gutter(g), a
