"""Learned corner + gutter regression for double-page spreads (optional).

A small CNN (MobileNetV3-Small head, exported to ONNX) trained on this rig's
own operator-validated Phase-4 geometry: it maps a raw frame to the four
corners of the *page-block* crop box — the operator's convention, which
excludes covers and the fanned page stack, a boundary no generic segmenter
targets — plus the gutter as a fraction of the way from the box's left edge
to its right (the same crop-relative fraction P4 and P6 store).

The model's absolute placement carries a small per-session bias (lighting,
book size, how far the fan protrudes), so the tracker never uses it raw:
predictions are differenced against a prediction on the operator's anchor
frame, and only the *change* moves the box (see ``model_quad_offsets``). The
bias cancels in the difference — the anchor-relative scheme that the
gradient edge-snap experiment showed is the only sound way to refine an
operator's deliberate placement.

Runs on onnxruntime (already a dependency via rembg). Everything degrades
gracefully: no model file, no onnxruntime, or a load failure just returns
None and the callers keep their mask-based behavior. Training lives in
``train_corner_net.py``; the model file is ``models/corner_net.onnx``.
"""
import threading
from pathlib import Path

import cv2
import numpy as np

# Must match train_corner_net.py exactly.
IN_W, IN_H = 448, 256
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def model_path() -> Path | None:
    """The corner model to serve: first ``models/corner_net*.onnx`` found.

    Models are named ``corner_net-<rig>.onnx`` (e.g. ``rm333-rig``, the rig
    the repo's bundled model was trained on) — the label travels in the
    filename and the file's ONNX metadata, and the loader stays
    label-agnostic so a retrain for another rig needs no code change.
    """
    c = sorted(MODELS_DIR.glob("corner_net*.onnx"))
    return c[0] if c else None

# Same lazy-session pattern as utils' _u2net: one load attempt, failures
# remembered, a lock because the watchdog thread and the request thread can
# both trigger the first inference.
_net = {"session": None, "state": "untried"}
_net_lock = threading.Lock()


def available() -> bool:
    """Whether the corner model can serve predictions (loads it if needed)."""
    return _session() is not None


def _session():
    with _net_lock:
        if _net["state"] == "unavailable":
            return None
        if _net["session"] is None:
            path = model_path()
            if path is None:
                _net["state"] = "unavailable"
                return None
            try:
                import onnxruntime as ort

                _net["session"] = ort.InferenceSession(
                    str(path), providers=["CPUExecutionProvider"]
                )
                _net["state"] = "ready"
            except Exception:
                _net["state"] = "unavailable"
                return None
        return _net["session"]


def predict(img) -> tuple["np.ndarray", float] | None:
    """(quad 4x2 fractions tl,tr,br,bl; gutter fraction of box width) or None."""
    sess = _session()
    if sess is None:
        return None
    small = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    x = (small[:, :, ::-1].astype(np.float32) / 255.0 - _MEAN) / _STD
    x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]
    out = sess.run(None, {"image": x})[0][0]
    quad = out[:8].astype(np.float64).reshape(4, 2)
    return quad, float(np.clip(out[8], 0.0, 1.0))


def model_gutter_frac(img, box_frac=None) -> float | None:
    """Model's spine position as a fraction of a crop box's width, or None.

    Feeds ``detect_gutter``'s ``prior``: the model hints where the spine is
    and the shadow scan refines within its tight band, exactly as an
    operator hint would. Without ``box_frac`` the fraction is relative to
    the model's own predicted box; with it, the predicted spine line is
    reprojected onto that box's horizontal axis, so the fraction matches
    the crop the caller actually warped.
    """
    p = predict(img)
    if p is None:
        return None
    quad, g = p
    if box_frac is None:
        return g
    tl, tr, br, bl = quad
    mid = ((tl + g * (tr - tl)) + (bl + g * (br - bl))) / 2.0
    btl, btr, bbr, bbl = [np.asarray(c, dtype=float) for c in box_frac]
    u = ((btr - btl) + (bbr - bbl)) / 2.0
    denom = float(u @ u)
    if denom <= 0:
        return float(g)
    t = float((mid - (btl + bbl) / 2.0) @ u) / denom
    return float(np.clip(t, 0.0, 1.0))


def model_quad_offsets(img, quad_px):
    """Model-measured per-edge boundary offsets around ``quad_px``, or None.

    Drop-in for ``utils.measure_quad_offsets`` inside the Phase-4 watchdog:
    same contract — ``(offsets, reliable)``, indexed top, right, bottom,
    left, offsets along each edge's outward normal in full-res pixels — but
    the boundary is the model's predicted box instead of the page mask's
    blob edge. Anchor frames and tracked frames must both be measured
    through the same path so the model's systematic bias cancels in the
    difference the tracker takes (exactly the mask path's rule).

    The prediction's edge midpoints are projected onto the *reference*
    quad's normals, so a slight predicted rotation degrades into the rigid
    translation the tracker models rather than corrupting it.
    """
    p = predict(img)
    if p is None:
        return None
    h, w = img.shape[:2]
    return quad_offsets_from_pred(p[0] * [w, h], quad_px)


def quad_offsets_from_pred(pred_px, quad_px):
    """Project a predicted quad onto a reference quad's edge normals.

    Split out of ``model_quad_offsets`` so the offline tracker-emulation
    eval can run the byte-identical projection on stored predictions.
    """
    tl, tr, br, bl = [np.asarray(c, dtype=float) for c in np.asarray(quad_px)]
    u = (tr - tl) + (br - bl)
    v = (bl - tl) + (br - tr)
    u /= max(1e-9, np.linalg.norm(u))
    v /= max(1e-9, np.linalg.norm(v))
    ptl, ptr, pbr, pbl = [np.asarray(c, dtype=float) for c in np.asarray(pred_px)]
    # Edge order and outward normals mirror utils.edge_boundary_offsets.
    pairs = (
        ((tl + tr) / 2, (ptl + ptr) / 2, -v),   # top
        ((tr + br) / 2, (ptr + pbr) / 2, u),    # right
        ((bl + br) / 2, (pbl + pbr) / 2, v),    # bottom
        ((tl + bl) / 2, (ptl + pbl) / 2, -u),   # left
    )
    offsets = [float((pm - qm) @ n) for qm, pm, n in pairs]
    return offsets, [True, True, True, True]
