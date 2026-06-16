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
