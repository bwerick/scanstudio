"""
Shared utilities for the ScanStudio pipeline.

Provides logging, overwrite prompts, and path helpers used by all scripts.

Directory structure per video:
  output/<video_name>/
  ├── images/    # keyframe images (4K), modified in-place
  ├── pages/     # split/cropped individual pages
  ├── plots/     # all diagnostic plots
  ├── data/      # .npy signal data
  ├── json/      # all metadata, configs, logs
  ├── reports/   # .md and .txt reports
  └── pdf/       # final PDFs
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def log(msg: str):
    """Print a timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def derive_output_dir(video_path: str, output_dir_override: str | None = None) -> Path:
    """
    Derive the per-video output directory from the video filename.
    recordings/foo.mp4 → output/foo/
    """
    if output_dir_override:
        return Path(output_dir_override)
    video_name = Path(video_path).stem
    return Path.cwd() / "output" / video_name


class ProjectPaths:
    """Standardized paths for a video project's output."""

    def __init__(self, output_dir: str | Path):
        self.base = Path(output_dir)
        self.images = self.base / "images"
        self.pages = self.base / "pages"
        self.plots = self.base / "plots"
        self.data = self.base / "data"
        self.json = self.base / "json"
        self.reports = self.base / "reports"
        self.pdf = self.base / "pdf"

    def ensure_all(self):
        """Create all subdirectories."""
        for d in [self.images, self.pages, self.plots, self.data,
                  self.json, self.reports, self.pdf]:
            d.mkdir(parents=True, exist_ok=True)
        return self

    def ensure(self, *dirs: str):
        """Create specific subdirectories by name."""
        for name in dirs:
            getattr(self, name).mkdir(parents=True, exist_ok=True)
        return self


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def bring_to_front(root):
    """Force a Tk window to the foreground and grab keyboard focus.

    On macOS, Tk windows open behind the active app and never steal focus, so a
    review GUI launched mid-pipeline (e.g. P7, which opens only after the long
    crop/split phase when attention has wandered to another window) can come up
    hidden and be dismissed unseen. Lifting, briefly pinning ``-topmost``, then
    forcing focus makes the window unmissable without leaving it permanently
    above everything else.
    """
    root.update_idletasks()
    root.deiconify()
    root.lift()
    root.attributes("-topmost", True)
    root.after(800, lambda: root.attributes("-topmost", False))
    root.focus_force()


def check_overwrite(path: Path) -> bool:
    """Prompt to confirm overwrite if path exists."""
    if not path.exists():
        return True
    while True:
        response = input(f"  '{path}' already exists. Overwrite? [y/n]: ").strip().lower()
        if response in ("y", "yes"):
            return True
        elif response in ("n", "no"):
            return False
        print("  Please enter 'y' or 'n'.")


# ── Page / spread vision helpers ─────────────────────────────
#
# A book page is bright and nearly colorless; a wood (or any tinted) table is
# darker and more saturated. Grayscale Otsu can't tell cream pages from
# light-brown wood — their luma overlaps — so it tends to segment the whole
# frame as "foreground". Thresholding in HSV on saturation + value separates
# them cleanly and is invariant to where the spread sits or how it's rotated.


def page_mask(img, sat_max: int = 70, val_min: int = 150) -> "np.ndarray":
    """Binary mask (uint8 0/255) of the page region against a tinted table.

    Pages are kept where saturation is low and value is high. The value floor
    adapts to lighting via Otsu but is capped at ``val_min`` so a dim capture
    still excludes the table. Only the largest blob is returned, filled solid.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    vt, _ = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((S < sat_max) & (V > min(int(vt), val_min))).astype(np.uint8) * 255

    # OPEN must run before CLOSE: wood-grain highlights pass the threshold as
    # sparse speckle, and closing first solidifies that speckle into blobs that
    # merge with the page. Opening first erases it while the dense page region
    # survives.
    k = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return clean


def book_center_x(mask) -> float | None:
    """Fraction (0–1) of the book's horizontal centre from a page mask.

    The mean of the left and right page edges, taken row by row and reduced by
    median so a slight rotation, page curl, or a hand clipping one edge doesn't
    skew it. This is where the spine sits on a symmetric spread — the same
    midpoint an operator eyeballs by halving the distance between the two outer
    edges. Returns None for an empty mask.
    """
    m = mask > 0
    rows = m.any(axis=1)
    if not rows.any():
        return None
    w = mask.shape[1]
    left = m.argmax(axis=1)                       # first page column per row
    right = (w - 1) - m[:, ::-1].argmax(axis=1)   # last page column per row
    mids = (left[rows] + right[rows]) / 2.0
    return float(np.median(mids)) / w


def detect_gutter(spread, mask=None, prior: float | None = None) -> int:
    """Return the x of the book gutter (spine) in a cropped spread.

    A bound book's two pages are equal width, so the spine sits at the book's
    geometric centre — the mean of its left and right edges (see
    ``book_center_x``), the midpoint an operator eyeballs by halving the gap
    between the outer edges. With no prior hint this *is* the answer: it tracks
    the book as it shifts and is far steadier than the shadow valley between the
    pages, which is faint when the book lies flat. A per-column brightness scan
    latches onto a text-block edge or a left/right page-brightness gradient just
    as readily as the spine, so trusting it tends to drag the line into a page.

    When ``prior`` (a gutter fraction from an earlier manual correction) is
    given, the line pins to it and a tight shadow scan tracks the spine as the
    book drifts, but only follows a column that is a genuine shadow — at least
    ~6% darker than the band's typical column. A faint or flat dip is ignored
    and the operator's hint holds.
    """
    h, w = spread.shape[:2]
    if mask is None:
        mask = page_mask(spread)

    if prior is None:
        gc = book_center_x(mask)
        return int(round((0.5 if gc is None else gc) * w))

    gray = cv2.cvtColor(spread, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mb = mask > 0
    counts = mb.sum(axis=0)
    sums = (gray * mb).sum(axis=0)
    # Columns with no page pixels are treated as bright so they're never chosen.
    col = np.where(counts > 0, sums / np.maximum(counts, 1), 255.0)
    col = np.convolve(col, np.ones(15) / 15, mode="same")

    center = float(np.clip(prior, 0.0, 1.0))
    cx = int(round(center * w))
    lo, hi = max(0, int(w * (center - 0.03))), min(w, int(w * (center + 0.03)))
    if hi <= lo:
        return cx
    seg = col[lo:hi]
    vx = lo + int(np.argmin(seg))

    # Track the spine only via a genuine shadow, not a text-density dip. Real
    # gutters run ~6–10% below the band's median brightness; false valleys only
    # ~2–3%, so this relative test (which also scales with exposure) rejects
    # them and keeps the operator's hint.
    median = float(np.median(seg))
    return vx if (median - float(seg.min())) >= 0.06 * median else cx


def resolve_gutter(keyframes, idx):
    """Gutter fraction in effect for keyframe ``idx`` as a tracking prior, or None.

    Mirrors ``resolve_rotation``: a manual gutter correction in review
    propagates forward as a *prior*, not a fixed value — later spreads re-detect
    the spine in a tight band around it (see ``detect_gutter``), so the line
    follows the book as it shifts while the operator's hint stays the anchor. A
    correction therefore becomes the exception rather than the rule. Returns the
    keyframe's own override, else the nearest earlier one, else None (full auto).
    """
    for kf in reversed(keyframes[: idx + 1]):
        g = kf.get("gutter")
        if g is not None:
            return g
    return None


def resolve_rotation(keyframes, idx):
    """Manual deskew angle (deg) in effect for keyframe ``idx``, or None.

    A rotation correction from review propagates forward: the rig rarely
    moves between page turns, so once the operator dials in an angle it
    applies to every following spread until the next correction (or until a
    reset removes it, at which point the previous correction takes over).
    Returns the keyframe's own override, else the nearest earlier one, else
    None (auto-detect)."""
    for kf in reversed(keyframes[: idx + 1]):
        rot = kf.get("rotation_deg")
        if rot is not None:
            return rot
    return None


def resolve_crop_anchor(keyframes, idx):
    """Manual crop box in effect for keyframe ``idx`` and where it came from.

    Returns ``(quad, anchor_idx)`` — the nearest crop_quad at or before
    ``idx`` and the index of the keyframe that owns it — or ``(None, None)``.
    The anchor index matters to the boundary watchdog: its baseline is
    measured on the anchor frame's own image, so later frames are compared
    against how the boundary sat when the operator drew the box."""
    for j in range(idx, -1, -1):
        q = keyframes[j].get("crop_quad")
        if q is not None:
            return q, j
    return None, None


def resolve_crop_quad(keyframes, idx):
    """Manual crop box in effect for keyframe ``idx`` (double mode), or None.

    The box — 4 corners (tl, tr, br, bl) as fractions of the raw frame, drawn
    in Phase-4 review — propagates forward like ``resolve_rotation``: the rig
    and the book barely move between page turns, so one corrected box holds
    for every following spread until the next correction. Returns the
    keyframe's own box, else the nearest earlier one, else None (auto crop).
    Single mode reads ``crop_quad`` directly without propagation: loose pages
    move and resize between frames, so one page's box says nothing about the
    next."""
    return resolve_crop_anchor(keyframes, idx)[0]


# ── Consensus box + boundary tracker (double mode) ──────────
#
# The intended semantics: the crop box *follows the book*. One *consensus*
# box is voted from a sample of frames — robust to the hands, glare, or
# mid-turn pages that break any single frame — and an operator correction
# becomes the tracking anchor from that frame on. The per-edge boundary
# measurement then keeps the box on the book as it drifts around the frame,
# so minor shifts never need a keypress.
#
# The tracker's motion model is deliberately rigid. Validation against a
# 60-correction session showed that following each measured edge
# independently makes boxes worse (the fanned page stack under the spread
# moves the paper outline even while the book holds still). A moving *book*
# shows up as opposing edges shifting together — rigid translation — while
# the fan shows up as edges moving apart — expansion. ``rigid_shift``
# decomposes the smoothed per-edge deltas into exactly those two parts: the
# translation is applied, the expansion never is, only scored. Phase 4
# flags frames whose non-rigid residual exceeds ~2% of the frame width, or
# that can't be measured at all (occlusion), for an operator's glance —
# everything else the box handles by itself.

# Mask/measure at this width: page_mask's 25 px morphology is tuned for
# roughly this scale, and it keeps a 4K frame cheap (~60 ms).
TRACK_WORK_WIDTH = 1600
# Measurement band around a box edge, as a fraction of frame width. A
# boundary beyond this simply isn't measured (reported unreliable).
TRACK_BAND_FRAC = 0.04
# Watchdog alert threshold. Between adjacent keyframes of a static book the
# per-edge measurement wobbles with p90 ≈ 1.2% of the frame width (page
# curl, the fanned stack, a hand); a median-smoothed shift beyond ~2% is a
# real event worth an operator's glance.
WATCHDOG_ALERT_FRAC = 0.02
# Tracker dead-band: the box only moves when the estimated translation
# differs from the currently applied one by more than this fraction of the
# frame width. Below it is measurement wobble (pairing + median-of-3 puts
# noise around 0.5–0.7% p90) and a static box shouldn't jitter with it.
TRACK_DEADBAND_FRAC = 0.006
CONSENSUS_SAMPLES = 15


def _order_quad(pts):
    """Order 4 points as (tl, tr, br, bl). Same rule as p5's order_points."""
    pts = np.asarray(pts, dtype=float)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]])


def _quad_axes(quad):
    """Unit vectors along the quad's top edge (u) and down its left edge (v)."""
    tl, tr, br, bl = [np.asarray(p, dtype=float) for p in quad]
    u = (tr - tl) + (br - bl)
    v = (bl - tl) + (br - tr)
    return u / max(1e-9, np.linalg.norm(u)), v / max(1e-9, np.linalg.norm(v))


def edge_boundary_offsets(mask, quad, band, n_samples=25, step=2):
    """Where the page boundary sits relative to each edge of ``quad``.

    Casts rays along the outward normal from points spread over each edge's
    middle 80%; each ray reports the offset at which the page mask ends
    (negative = boundary inside the box, positive = outside). The per-edge
    result is the median over its rays, so a hand or a glare patch crossing
    part of an edge is outvoted. An edge is *unreliable* when most rays never
    cross the boundary within ±``band`` px — mask everywhere (a page stack
    running past the box and out of frame) or nowhere — or when the median
    sits at the band limit.

    Returns ``(offsets, reliable)``, each length 4, indexed top, right,
    bottom, left. All units are mask pixels.
    """
    m = mask > 0
    h, w = m.shape[:2]
    tl, tr, br, bl = [np.asarray(p, dtype=float) for p in quad]
    u, v = _quad_axes(quad)
    edges = ((tl, tr, -v), (tr, br, u), (bl, br, v), (tl, bl, -u))
    ts = np.arange(-band, band + 1e-9, step)
    offsets, reliable = [], []
    for a, b, n in edges:
        span = np.linspace(0.1, 0.9, n_samples)[:, None]
        base = a[None, :] + span * (b - a)[None, :]
        pts = base[:, None, :] + ts[None, :, None] * n[None, None, :]
        xi = np.rint(pts[..., 0]).astype(int)
        yi = np.rint(pts[..., 1]).astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        inside = np.zeros(ok.shape, dtype=bool)
        inside[ok] = m[yi[ok], xi[ok]]
        # The mask is a solid blob, so along a ray "inside length" locates the
        # step edge: boundary = -band + (pixels inside) — no explicit crossing
        # search, and small speckle just nudges the estimate.
        counts = inside.sum(axis=1)
        mixed = (counts > 0) & (counts < len(ts))
        if mixed.sum() >= n_samples * 0.5:
            off = float(np.median(-band + counts[mixed] * step))
            rel = abs(off) <= band * 0.95
        else:
            off, rel = 0.0, False
        offsets.append(off)
        reliable.append(bool(rel))
    return offsets, reliable


def measure_quad_offsets(img, quad_px, band_px=None):
    """Per-edge page-boundary offsets for a full-res frame, in full-res px.

    Downscales, runs ``page_mask``, and measures ``edge_boundary_offsets``
    around ``quad_px``. Anchor frames and tracked frames must both be measured
    through this same path so any systematic bias of the mask cancels in the
    difference the tracker uses. Returns ``(offsets, reliable)``.
    """
    h, w = img.shape[:2]
    if band_px is None:
        band_px = TRACK_BAND_FRAC * w
    s = min(1.0, TRACK_WORK_WIDTH / w)
    small = (
        cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if s < 1.0
        else img
    )
    mask = page_mask(small)
    off, rel = edge_boundary_offsets(
        mask, np.asarray(quad_px, dtype=float) * s, band=band_px * s
    )
    return [o / s for o in off], rel


def quad_edge_bases(quad):
    """Each edge's scalar position along its outward normal, plus the axes.

    Projects the midpoint of each edge (top, right, bottom, left) onto that
    edge's outward normal. Adding a measured boundary offset to the base
    turns it into an *absolute* boundary position in frame coordinates —
    comparable across measurements taken against differently-placed boxes,
    which is what lets the tracker measure around the box it has already
    moved. Returns ``(bases, (u, v))`` with bases in the quad's units.
    """
    tl, tr, br, bl = [np.asarray(p, dtype=float) for p in quad]
    u, v = _quad_axes(quad)
    mids = ((tl + tr) / 2, (tr + br) / 2, (bl + br) / 2, (tl + bl) / 2)
    normals = (-v, u, v, -u)
    return [float(m @ n) for m, n in zip(mids, normals)], (u, v)


def rigid_shift(anchor_s, anchor_rel, s_window, rel_window):
    """Split the boundary's movement since the anchor into rigid + residual.

    Inputs are per-edge *absolute* boundary positions (edge base + measured
    offset, see ``quad_edge_bases``), for the anchor and for a window of a
    few consecutive keyframes. Each edge is median-smoothed across the
    window exactly as the old watchdog was — a one-frame artifact (a hand, a
    lifted page fan) cannot move the box or alert, while a change that
    *persists* does; validated on a real session, single-frame deltas
    false-alarm as often as they detect.

    Opposing edges are then paired: their antisymmetric part is rigid
    translation (the book moved — the tracker follows it), their symmetric
    part is expansion (the paper outline grew — the fanned-stack artifact
    that made naive per-edge following worse than a static box, so it is
    only ever reported). An edge whose partner is unmeasurable can't be
    decomposed: that axis isn't tracked and the whole delta counts as
    residual.

    Returns ``(shift, residual, measured)``: ``shift`` is ``[t_u, t_v]``
    along the box axes (px, None = axis untracked), ``residual`` the worst
    non-rigid |delta| (px), ``measured`` whether any edge was measurable at
    all (an immeasurable frame deserves a glance too).
    """
    d = [None] * 4
    for k in range(4):
        if not anchor_rel[k]:
            continue
        vals = [s[k] for s, r in zip(s_window, rel_window) if r[k]]
        if len(vals) < (len(s_window) + 1) // 2:
            continue
        d[k] = float(np.median(vals)) - anchor_s[k]
    measured = any(x is not None for x in d)
    shift, residual = [None, None], 0.0
    # Axis pairs as (positive-normal edge, negative-normal edge):
    # u = (right, left), v = (bottom, top).
    for ax, (kp, km) in enumerate(((1, 3), (2, 0))):
        if d[kp] is not None and d[km] is not None:
            shift[ax] = (d[kp] - d[km]) / 2
            residual = max(residual, abs((d[kp] + d[km]) / 2))
        else:
            for k in (kp, km):
                if d[k] is not None:
                    residual = max(residual, abs(d[k]))
    return shift, residual, measured


def consensus_geometry(images_dir, keyframes, cache_path=None,
                       samples=CONSENSUS_SAMPLES, log_fn=None):
    """One crop box for the whole session, voted from a sample of frames.

    Computes the page mask on ``samples`` keyframes spread across the session,
    keeps the pixels that are page in at least half of them (a hand, glare, or
    a mid-turn page in any one frame is outvoted), and takes the largest
    blob's minimum-area rectangle — a snug rotated box with the session's
    angle built in. Also measures the box's per-edge baseline boundary
    offsets on the voted mask, which is what the boundary tracker needs to
    use the consensus as its anchor on frames with no manual correction yet.

    Returns ``{"quad": 4 fractional corners, "edge_ref": [4 offsets in
    full-res px], "edge_rel": [4 bools], "size": [W, H]}`` or None when there
    aren't enough readable same-sized frames or no page-sized region exists.
    Cached to ``cache_path`` keyed by the sampled files' identity, so the
    ~2 s vote runs once per project, not once per run."""
    cands = [kf["filename"] for kf in keyframes if not kf.get("is_cover")]
    if len(cands) < 3:
        return None
    picks = sorted(set(np.linspace(0, len(cands) - 1,
                                   min(samples, len(cands))).astype(int)))
    files, fp = [], []
    for i in picks:
        p = Path(images_dir) / cands[i]
        if p.exists():
            st = p.stat()
            files.append(p)
            fp.append([cands[i], st.st_size, st.st_mtime_ns])
    if len(files) < 3:
        return None

    if cache_path is not None and Path(cache_path).exists():
        try:
            data = json.loads(Path(cache_path).read_text())
            if data.get("fingerprint") == fp:
                return data
        except (json.JSONDecodeError, KeyError):
            pass

    if log_fn:
        log_fn(f"Voting consensus crop box from {len(files)} frames…")
    masks, size, scale = [], None, 1.0
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        if size is None:
            size = (w, h)
            scale = min(1.0, TRACK_WORK_WIDTH / w)
        elif (w, h) != size:
            # Mixed sizes mean images/ was already cropped in place (Phase 5
            # ran) — a vote over those is meaningless.
            continue
        small = (
            cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else img
        )
        masks.append(page_mask(small) > 0)
    if len(masks) < 3:
        return None

    maj = (np.mean(masks, axis=0) >= 0.5).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(maj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    big = max(cnts, key=cv2.contourArea)
    mh, mw = maj.shape[:2]
    if cv2.contourArea(big) < 0.05 * mw * mh:
        return None
    solid = np.zeros_like(maj)
    cv2.drawContours(solid, [big], -1, 255, -1)
    quad = _order_quad(cv2.boxPoints(cv2.minAreaRect(big)))
    off, rel = edge_boundary_offsets(
        solid, quad, band=TRACK_BAND_FRAC * size[0] * scale
    )

    W, H = size
    data = {
        "fingerprint": fp,
        "size": [W, H],
        "quad": [[round(float(x) / scale / W, 5), round(float(y) / scale / H, 5)]
                 for x, y in quad],
        "edge_ref": [round(o / scale, 2) for o in off],
        "edge_rel": rel,
    }
    if cache_path is not None:
        Path(cache_path).write_text(json.dumps(data))
    return data


def text_skew(page, max_deg: float = 3.0) -> float:
    """Residual skew (deg) of the text lines in a single page image.

    Projection-profile search: rotate the dark (text) pixels by candidate
    angles and keep the angle that concentrates them into the sharpest row
    profile — the squared-bin-count score peaks when lines are horizontal and
    the gaps between them empty. The result is the angle to pass to
    ``cv2.getRotationMatrix2D`` to level the text. Returns 0 when the page
    has too little text to measure."""
    g = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    s = min(1.0, 1000.0 / max(h, w, 1))
    if s < 1.0:
        g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    # Ignore the outer margins: page edges and gutter shadow are dark, slanted
    # structures that would otherwise dominate the profile.
    mh, mw = int(g.shape[0] * 0.07), int(g.shape[1] * 0.07)
    g = g[mh : g.shape[0] - mh, mw : g.shape[1] - mw]
    if g.size == 0:
        return 0.0
    bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ys, xs = np.nonzero(bw)
    if len(ys) < 500:
        return 0.0
    if len(ys) > 60000:
        sel = np.random.default_rng(0).choice(len(ys), 60000, replace=False)
        ys, xs = ys[sel], xs[sel]
    x = xs - xs.mean()
    y = ys - ys.mean()

    def score(a):
        t = np.radians(a)
        # y' row of cv2.getRotationMatrix2D, so the best `a` feeds it directly.
        yr = y * np.cos(t) - x * np.sin(t)
        hist = np.bincount(((yr - yr.min()) / 2.0).astype(np.int64))
        hist = hist.astype(np.float64)
        return float((hist * hist).sum())

    best = 0.0
    for step, span in ((0.25, max_deg), (0.05, 0.3)):
        cands = np.arange(best - span, best + span + 1e-9, step)
        best = float(cands[int(np.argmax([score(a) for a in cands]))])
    return best if abs(best) >= 0.05 else 0.0


def check_overwrite_dir(dir_path: Path) -> bool:
    """Prompt to confirm overwrite if directory has files."""
    if not dir_path.exists():
        return True
    files = list(dir_path.iterdir())
    if not files:
        return True
    while True:
        response = input(f"  '{dir_path}' already has {len(files)} files. Overwrite? [y/n]: ").strip().lower()
        if response in ("y", "yes"):
            return True
        elif response in ("n", "no"):
            return False
        print("  Please enter 'y' or 'n'.")
