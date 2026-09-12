"""
Shared utilities for the ScanStudio pipeline.

Provides logging, overwrite prompts, and path helpers used by all scripts.

Directory structure per video:
  output/<video_name>/
  ├── images/    # keyframe images (4K), modified in-place
  ├── pages/     # split/cropped individual pages
  ├── pages_orig/ # pristine copies of pages P7 re-rendered (rotate/translate)
  ├── plots/     # all diagnostic plots
  ├── data/      # .npy signal data
  ├── json/      # all metadata, configs, logs
  ├── reports/   # .md and .txt reports
  └── pdf/       # final PDFs
"""

import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


# ── Platform portability ─────────────────────────────────────
#
# The pipeline itself is plain Python/OpenCV and runs anywhere; only three
# things actually differ between macOS and Linux — which capture backend
# reaches the camera's high-res modes, which key means "save", and how a
# machine plays a short sound. Each is resolved once here so the phase
# scripts stay free of platform branches.

IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

# How many camera indices a scan walks. On Linux a single physical camera
# usually claims two /dev/video nodes (capture + metadata), so the same
# number of indices covers half as many cameras.
CAMERA_SCAN_RANGE = 10 if IS_LINUX else 5


def camera_backend():
    """OpenCV capture backend that exposes a webcam's full mode list.

    macOS reaches a camera's high-res modes only through AVFoundation — the
    default backend silently tops out at 1080p on some cameras. On Linux the
    equivalent is V4L2, which is also the backend that honours a FOURCC
    request (see ``prepare_capture``). Anything else gets OpenCV's own pick.
    """
    if IS_MAC:
        return cv2.CAP_AVFOUNDATION
    if IS_LINUX:
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def backend_name(backend) -> str:
    """Human-readable name for an OpenCV capture backend constant."""
    return {cv2.CAP_AVFOUNDATION: "AVFOUNDATION", cv2.CAP_V4L2: "V4L2",
            cv2.CAP_ANY: "DEFAULT"}.get(backend, str(backend))


def prepare_capture(cap, want_w: int, want_h: int, fps: Optional[float] = None):
    """Ask an open capture for ``want_w``x``want_h`` (and optionally fps).

    Order matters on V4L2: most UVC cameras offer 4K only as MJPG and default
    to raw YUYV, which caps out far lower and at a few fps, so the format is
    requested *before* the resolution. AVFoundation picks the format itself
    and is left alone. The camera negotiates to its nearest supported mode
    either way, so callers must read back what they actually got.
    """
    if not IS_MAC:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(want_w))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(want_h))
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, float(fps))


def camera_label(idx: int) -> str:
    """How to name camera index ``idx`` in operator-facing output."""
    return f"index {idx} (/dev/video{idx})" if IS_LINUX else f"index {idx}"


# Recording codecs, in preference order per choice. "mp4v" writes MPEG-4
# Part 2, which no browser decodes — P4's web review can't scrub such a
# recording in its <video> and transcodes a proxy first. "avc1" is real
# H.264 and needs no proxy, but it is roughly half as fast: measured on
# 150 consecutive 4K frames of an actual page turn, mp4v sustained 21.5
# fps and avc1 10.2. Live capture at 4K is already encoder-bound, so
# real-time recording keeps mp4v and pays the one-time proxy instead;
# offline passes (normalize) prefer H.264, where throughput is free.
VIDEO_CODECS = {"auto": ("avc1", "mp4v"), "h264": ("avc1",), "mp4v": ("mp4v",)}


def open_video_writer(path, fps: float, size, codec: str = "auto"):
    """Open a VideoWriter for ``path``, preferring H.264.

    Returns ``(writer, fourcc)``; ``(None, "")`` if no codec opened. The
    fourcc that won is the caller's to report — it decides whether the
    recording is playable in the browser review.
    """
    for tag in VIDEO_CODECS[codec]:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*tag),
                                 fps, size)
        if writer.isOpened():
            return writer, tag
        writer.release()
    return None, ""


# Save accelerator: ⌘S is the macOS convention, Ctrl+S everywhere else.
SAVE_ACCEL = "<Command-s>" if IS_MAC else "<Control-s>"
SAVE_LABEL = "⌘S" if IS_MAC else "Ctrl+S"


def bind_save(root, callback):
    """Bind the platform's save accelerator to ``callback``.

    Ctrl+S is bound on macOS too — it costs nothing and spares anyone whose
    muscle memory came from the other platform.
    """
    for accel in {SAVE_ACCEL, "<Control-s>"}:
        root.bind(accel, lambda e: callback())


_MONO_FAMILY = None
# In preference order; the first one the system actually has wins. Menlo is
# macOS, the next three ship with common Linux desktops.
_MONO_CANDIDATES = ("Menlo", "DejaVu Sans Mono", "Liberation Mono",
                    "Ubuntu Mono", "Noto Sans Mono", "Consolas", "Courier New")


def mono(size: int, *style: str):
    """A Tk font tuple in the best monospace family this system has.

    The review GUIs align their labels by character width, and Tk silently
    substitutes a *proportional* default for a family the system doesn't
    have — a hard-coded "Menlo" reads fine on macOS and ragged on Linux.
    Needs an existing Tk root, so call it from UI construction rather than
    at import time.
    """
    global _MONO_FAMILY
    if _MONO_FAMILY is None:
        from tkinter import font as tkfont

        have = {f.lower() for f in tkfont.families()}
        _MONO_FAMILY = next(
            (f for f in _MONO_CANDIDATES if f.lower() in have),
            "Courier",   # Tk resolves this to a fixed font on every platform
        )
    return (_MONO_FAMILY, size, *style)


# Short chime for a capture, as (executable, sound file) pairs in preference
# order. macOS ships afplay and a stock alert; Linux desktops ship one of the
# PulseAudio/ALSA players plus the freedesktop sound theme.
_SOUND_CANDIDATES = (
    [("afplay", "/System/Library/Sounds/Glass.aiff")]
    if IS_MAC
    else [
        ("paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"),
        ("paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"),
        ("pw-play", "/usr/share/sounds/freedesktop/stereo/complete.oga"),
        ("aplay", "/usr/share/sounds/alsa/Front_Center.wav"),
    ]
)


def sound_player():
    """Resolve a zero-argument callable that plays a short capture chime.

    Resolved once, at startup: the player and the file it can play differ per
    platform, and a machine with neither should fall back to the terminal
    bell rather than fork a doomed process on every capture.
    """
    for exe, snd in _SOUND_CANDIDATES:
        if shutil.which(exe) and Path(snd).exists():
            cmd = [exe, snd]

            def play(cmd=cmd):
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)

            return play

    if shutil.which("canberra-gtk-play"):   # theme sound, no file path needed
        def play_theme():
            subprocess.Popen(["canberra-gtk-play", "-i", "complete"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return play_theme

    def bell():
        sys.stdout.write("\a")
        sys.stdout.flush()

    return bell


def log(msg: str):
    """Print a timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def derive_output_dir(video_path: str, output_dir_override: Optional[str] = None) -> Path:
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

    def __init__(self, output_dir: Union[str, Path]):
        self.base = Path(output_dir)
        self.images = self.base / "images"
        self.pages = self.base / "pages"
        # Pristine copies of the pages P7 re-rendered, so its rotate/translate
        # stays adjustable and reversible instead of compounding on itself.
        self.pages_orig = self.base / "pages_orig"
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

    On macOS, Tk windows open behind the active app and never steal focus; many
    Linux window managers apply the same focus-stealing prevention. Either way a
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


def _mask_plausible(mask) -> bool:
    """Whether a mask looks like a book on a table rather than a failure.

    The HSV mask's failures are not subtle: on a pale table it merges page
    and wood into a blob spanning the whole frame, and under bad lighting it
    keeps only a glare patch. A real spread covers a substantial minority of
    the frame and, while it often runs off the top and bottom, never spans
    essentially the full frame in *both* axes.
    """
    h, w = mask.shape[:2]
    area = cv2.countNonZero(mask) / (w * h)
    if not (0.10 <= area <= 0.90):
        return False
    _, _, bw, bh = cv2.boundingRect(mask)
    return not (bw >= 0.97 * w and bh >= 0.97 * h)


# rembg (U^2-Net) is an optional dependency: the session is created on first
# use and the import failure is remembered, so a machine without it pays one
# try and then behaves exactly as before the backstop existed.
_u2net = {"session": None, "remove": None, "state": "untried"}
# The web review resolves geometry on the request thread while the consensus
# vote runs in its own; without a lock a cold cache means two concurrent
# new_session() calls, each downloading the model.
_u2net_lock = threading.Lock()


def u2net_page_mask(img):
    """Page mask from U^2-Net salient-object segmentation, or None.

    The learned counterpart to ``page_mask``: rembg's U^2-Net segments the
    book as an *object*, so it never merges it with a same-colored table —
    the HSV mask's catastrophic failure mode. It is coarser along the
    boundary and ~0.5 s/frame on CPU at the tracker's working width, which
    is why it backstops the HSV mask rather than replacing it. Returns the
    same contract as ``page_mask`` (uint8 0/255, largest blob filled solid),
    or None when rembg isn't installed or finds nothing.
    """
    with _u2net_lock:
        if _u2net["state"] == "unavailable":
            return None
        if _u2net["session"] is None:
            try:
                from rembg import new_session, remove

                _u2net["session"], _u2net["remove"] = new_session("u2net"), remove
                _u2net["state"] = "ready"
            except ImportError:
                _u2net["state"] = "unavailable"
                return None
    m = _u2net["remove"](
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        session=_u2net["session"],
        only_mask=True,
        post_process_mask=True,
    )
    m = (m > 127).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    clean = np.zeros_like(m)
    cv2.drawContours(clean, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return clean


def page_mask_robust(img):
    """``page_mask`` with a learned backstop for its catastrophic failures.

    Benchmarked against 120 operator-drawn boxes: the HSV mask tracks the
    book to ~1% of the frame width in the typical case but its worst frames
    are 37%-of-width disasters (page merged with a pale table), while U^2-Net
    never produced one. So the cheap mask runs first and the model only runs
    on frames where the HSV blob fails the plausibility check. When rembg is
    missing or U^2-Net's blob is implausible too, the HSV mask is returned
    unchanged — identical behavior to the pre-backstop pipeline.
    """
    mask = page_mask(img)
    if _mask_plausible(mask):
        return mask
    m2 = u2net_page_mask(img)
    return m2 if m2 is not None and _mask_plausible(m2) else mask


def book_center_x(mask) -> Optional[float]:
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


def detect_gutter(spread, mask=None, prior: Optional[float] = None) -> int:
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
        mask = page_mask_robust(spread)

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
#
# Also tried and rejected (2026-08, against 117 operator boxes across three
# sessions): snapping the published box's edges to the strongest nearby
# image gradient, phone-scanner style. At the production horizon (anchor a
# few keyframes back) it made every session *worse* — next-sample median
# error 0.58% → 0.95% of the width — because the operator's box encodes
# intent, not the physical outline: it often sits deliberately off the
# gradient line (margin, excluded covers), and an absolute snap drags it
# back. The same benchmark is what put the U^2-Net backstop into
# page_mask_robust: rigid tracking with the plain HSV mask had 9/117
# catastrophic frames (>5% error, worst 6.7%) at long horizons, the
# backstop zero (worst 4.2%), with no change at short horizons.
#
# Also tried and rejected (2026-08, 1159 operator boxes across 8 books):
# replacing the mask as the tracker's boundary measurement with a *learned*
# corner regressor trained on this rig's own geometry — the obvious answer
# to "the operator's box is a convention no generic segmenter targets".
# Two architectures, both differencing against the anchor frame so their
# per-session bias cancels: a full-frame MobileNetV3-Small regressing 4
# corners + gutter, and a per-corner patch refiner (96 px patches, jittered,
# flip-canonicalized). Emulating this watchdog and scoring at the frames the
# operator actually corrected, both lost to the mask: median next-edit error
# 0.80% of width for the mask, 0.83% for the refiner, 0.92% for the
# full-frame net. The patch refiner could not even improve a deliberately
# jittered box (2.10% median in, 2.09–5.04% out). The ceiling is not
# perception: the page-block boundary is partly *invisible*, inside the
# fanned stack, so there is nothing local to regress onto. This closes the
# "anchor-relative snap" variant the edge-snap note above left open.
#
# What that benchmark did establish is that the corrections themselves are
# largely cosmetic: across 347 operator edits the propagated box severs a
# median of 11 glyphs per frame and the operator's own replacement 8, out of
# ~2581 — 40% of edits reduce severed glyphs, 31% increase them. The box was
# already good; the win is in *asking for fewer corrections*, not in
# measuring the boundary more precisely.

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
# Drift budget: keyframes a box may ride its anchor before the operator is
# asked to re-anchor it. Tracking error grows monotonically with this
# horizon and never self-corrects — measured over three books tracked from a
# single anchor with no corrections, median max-corner error runs 0.59% of
# the frame width at horizon 1-15, 1.48% at 16-40, 2.22% at 41-80, 4.35% at
# 81-150. Frames where the box mangles text (severing >2x the glyph count
# the operator's own box does, and >25 glyphs) follow the same curve: 0% up
# to horizon 15, 4% at 16-40, 22% at 41-80, 44% at 81-150. Forty is where
# that damage starts being common, and it is the strongest predictor
# available — horizon scores AUC 0.905 for "error >3%", against 0.708 for
# accumulated shift, 0.600 for the worst per-edge offset, and 0.583 for the
# non-rigid residual the watchdog already flags on. So the residual alert
# and this budget are complementary: the residual catches a *sudden* event
# (a hand, a bumped rig), the budget catches slow accumulation the residual
# is nearly blind to.
DRIFT_HORIZON_LIMIT = 40
# The other half of the budget. A keyframe count only measures drift if every
# rig drifts at the same rate, and they don't: on the corpus above the box
# needed 41-80 keyframes to accumulate 2.2% of the frame width, while a
# session shot on a second rig moved 1-4% within 1-9 keyframes and swung
# 12.6% across the book. Forty frames would not have fired once before that
# session was already past 3%. Accumulated displacement transfers where a
# frame count doesn't, so whichever budget is spent first raises the
# reminder. (Measured alone on the single-rig corpus, displacement is the
# weaker predictor — AUC 0.708 against 0.905 — but that corpus holds drift
# rate constant, which is exactly the condition that flatters frame count.)
DRIFT_SHIFT_FRAC = 0.015


def drift_due(idx, anchor_idx, limit=DRIFT_HORIZON_LIMIT) -> bool:
    """Whether frame ``idx`` is a re-anchor reminder for its anchor.

    True every ``limit`` keyframes of horizon rather than for every frame
    past it: once the budget is spent, flagging all of the rest would bury
    the residual alerts in a wall of duplicates, while one reminder per
    budget-length reads as the cadence it is. ``anchor_idx`` is the index
    that owns the box (``resolve_crop_anchor``), or None when the box comes
    from the session consensus — that box was voted from the whole session
    rather than placed on a frame, so its horizon is counted from the start.
    """
    horizon = idx - (0 if anchor_idx is None else anchor_idx)
    return horizon > 0 and horizon % limit == 0


def shift_due(shift_frac, mark, shift_limit=DRIFT_SHIFT_FRAC) -> bool:
    """Whether the box has travelled another budget's worth since ``mark``.

    ``shift_frac`` is the tracker's accumulated translation from the anchor
    as a fraction of frame width; ``mark`` is the value at the last
    reminder. Comparing against a mark rather than an absolute threshold
    keeps the cadence one reminder per budget spent — the same discipline
    ``drift_due`` applies to the horizon — instead of firing on every frame
    once a static threshold is crossed. The caller advances the mark to
    ``shift_frac`` when this returns True, and resets it to 0.0 whenever the
    anchor changes.
    """
    return shift_frac >= mark + shift_limit


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

    Downscales, runs ``page_mask_robust``, and measures
    ``edge_boundary_offsets`` around ``quad_px``. Anchor frames and tracked
    frames must both be measured through this same path so any systematic
    bias of the mask cancels in the difference the tracker uses. (The U^2-Net
    backstop can break that cancellation on the rare frame where only one
    side of the comparison fell back — but the alternative on those frames
    was a whole-frame HSV blob, a far larger error than the masks' boundary
    disagreement.) Returns ``(offsets, reliable)``.
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
    mask = page_mask_robust(small)
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
        masks.append(page_mask_robust(small) > 0)
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


# ── Documents: one scan → many PDFs ──────────────────────────
#
# A single recording is usually one physical book, but a book is often several
# documents (chapters, articles, a run of receipts). Rather than splitting the
# finished PDF afterwards, the boundaries are marked during review: P4's "Doc
# Start" on a spread, refined to the exact page with P7's "First Page". Both
# land on the page entry as ``is_doc_start``, so P9 needs nothing but
# pages.json to know where each document begins.


def slugify(text: Optional[str], max_len: int = 40) -> str:
    """Filename-safe slug from a free-text title ('' when there's nothing)."""
    first = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return re.sub(r"[^a-z0-9]+", "-", first.lower()).strip("-")[:max_len].strip("-")


def segment_documents(pages):
    """Split ``pages`` into documents at every page tagged ``is_doc_start``.

    Returns a list of ``{"title", "slug", "pages"}``. Pages before the first
    tag (or all of them, when nothing is tagged) form one leading document, so
    an untagged project yields exactly one segment covering the whole scan —
    which is what keeps the single-PDF path identical to its old behaviour.
    The title is the ``doc_title`` recorded on the starting page, if any. The
    leading document reads its title the same way: the first page carries no
    ``is_doc_start`` (it opens a document implicitly) but may still be named."""
    docs = []
    for pg in pages:
        if not docs or pg.get("is_doc_start"):
            title = pg.get("doc_title")
            docs.append({"title": title, "slug": slugify(title), "pages": []})
        docs[-1]["pages"].append(pg)
    return docs


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
