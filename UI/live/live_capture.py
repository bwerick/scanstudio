#!/usr/bin/env python3
"""
ScanStudio Live Capture — standalone app launched by the web UI.

Two phases:
  SETUP      live view with alignment aids; nothing is recorded or captured.
             Frame the book, square up the camera, then press Enter.
  CAPTURING  records the camera and auto-captures the sharpest frame each
             time the book settles after a page turn.

Emits the same artifacts the offline motion → peaks → keyframes phases
produce, so review, crop, split, and export work unchanged. Runs as its own
process so the capture window owns a main thread (macOS requires that of
OpenCV windows). Progress is published to json/live_status.json.

Keys
  setup      Enter / Space  start capturing      K  keystone check
  capturing  Space  pause    C  capture now      U  undo last
  any time   G  center line  H  keys             M  mute     Q / Esc  finish
"""

import argparse
import json
import platform
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from engine import ProjectPaths          # noqa: E402
from sounds import Player                # noqa: E402


# ── Look ─────────────────────────────────────────────────────
# Soft tones that still read instantly: coral moving, gold settling,
# teal still. Panels are translucent so the page always shows through.

PANEL_RGBA = (15, 15, 20, 168)
TEXT = (232, 236, 242)
MUTED = (150, 158, 176)
STATE = {
    "SETUP":    ((148, 160, 184), "SETUP"),
    "TURNING":  ((232, 138, 120), "MOVING"),
    "SETTLING": ((230, 196, 120), "SETTLING"),
    "STABLE":   ((94, 196, 170), "STILL"),
    "PAUSED":   ((166, 164, 210), "PAUSED"),
}
REC_RED = (232, 110, 100)

_FONT_CACHE = {}


def font(size, bold=False):
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    f = None
    cands = [("/System/Library/Fonts/SFNS.ttf", None),
             ("/System/Library/Fonts/HelveticaNeue.ttc", 1 if bold else 0),
             ("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0),
             ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
              else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", None),
             ("C:/Windows/Fonts/segoeuib.ttf" if bold
              else "C:/Windows/Fonts/segoeui.ttf", None)]
    for path, idx in cands:
        try:
            f = ImageFont.truetype(path, size, index=idx or 0)
            if path.endswith("SFNS.ttf"):
                try:
                    f.set_variation_by_name("Semibold" if bold else "Regular")
                except Exception:
                    pass
            break
        except Exception:
            continue
    _FONT_CACHE[key] = f or ImageFont.load_default()
    return _FONT_CACHE[key]


def blend(dst, rgba, x, y):
    """Alpha-composite an RGBA sprite onto a BGR frame, clipped to bounds."""
    h, w = rgba.shape[:2]
    H, W = dst.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sp = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = sp[..., 3:4].astype(np.float32) / 255.0
    roi = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (sp[..., 2::-1].astype(np.float32) * a
                         + roi * (1 - a)).astype(np.uint8)


def panel(lines, u, align="left", dot=None):
    """
    A translucent rounded panel holding [(text, size, color, bold), ...].
    `dot` draws a small filled circle ahead of the first line — the record
    light — so it is part of the layout rather than painted on afterwards.
    Rendered at 2× and downsampled so type and corners are smooth.
    """
    ss = 2
    pad = int(12 * u) * ss
    gap = int(5 * u) * ss
    fs = [font(int(sz * u) * ss, b) for _, sz, _, b in lines]
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    sizes = [probe.textbbox((0, 0), t, font=f) for (t, *_), f in zip(lines, fs)]
    dot_w = int(16 * u) * ss if dot else 0
    tw = max((b[2] - b[0]) + (dot_w if i == 0 else 0) for i, b in enumerate(sizes))
    th = sum(b[3] - b[1] for b in sizes) + gap * (len(lines) - 1)
    W, H = tw + 2 * pad, th + 2 * pad
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, W - 1, H - 1), radius=int(10 * u) * ss,
                        fill=PANEL_RGBA)
    y = pad
    for i, ((t, _, col, _), f, b) in enumerate(zip(lines, fs, sizes)):
        x = pad if align == "left" else (W - (b[2] - b[0])) // 2
        if i == 0 and dot:
            r = int(4.5 * u) * ss
            cy = y + (b[3] - b[1]) // 2
            d.ellipse((x, cy - r, x + 2 * r, cy + r), fill=dot)
            x += dot_w
        d.text((x - b[0], y - b[1]), t, font=f, fill=col)
        y += (b[3] - b[1]) + gap
    img = img.resize((W // ss, H // ss), Image.LANCZOS)
    return np.array(img)


class SpriteCache:
    """Panels only re-render when their text changes."""

    def __init__(self):
        self._c = {}

    def get(self, key, make):
        if key not in self._c:
            if len(self._c) > 64:
                self._c.clear()
            self._c[key] = make()
        return self._c[key]


def meter(frac, color, settle_frac, arc, pulse, u, ss=3):
    """
    The motion circle. The filled *area* tracks motion (radius ∝ √motion),
    a faint inner ring marks the settle threshold, an arc around the rim
    counts down to a capture, and a ring pulses outward when one fires.
    """
    R = int(34 * u)
    C = int(R * 1.75)                     # canvas half-size, room for the pulse
    S = 2 * C * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = S // 2
    r = R * ss

    def circ(rad, **kw):
        d.ellipse((c - rad, c - rad, c + rad, c + rad), **kw)

    circ(r, fill=PANEL_RGBA)
    circ(r - 1 * ss, outline=(255, 255, 255, 38), width=max(1, ss))

    inner = r - int(5 * u) * ss
    if frac > 0.004:
        circ(max(1, int(inner * frac)), fill=(*color, 225))

    sr = int(inner * settle_frac)
    circ(sr, outline=(255, 255, 255, 70), width=max(1, ss))

    if arc > 0:
        w = max(2, int(2.6 * u * ss))
        d.arc((c - r + w, c - r + w, c + r - w, c + r - w),
              start=-90, end=-90 + 360 * arc, fill=(*color, 240), width=w)

    if pulse is not None:                  # 0 → 1 over the pulse lifetime
        pr = int(r * (1.0 + 0.7 * pulse))
        alpha = int(170 * (1 - pulse))
        circ(pr, outline=(*STATE["STABLE"][0], alpha),
             width=max(1, int(2.2 * u * ss)))

    img = img.resize((2 * C, 2 * C), Image.LANCZOS)
    return np.array(img)


def center_line(frame, alpha_core=0.62, alpha_edge=0.30):
    """
    Thin guide down the middle: a dark core with faint light edges, so it
    reads as a quiet line on white pages and still shows on a black cover.
    Blends only a 5-px strip, so it costs almost nothing per frame.
    """
    h, w = frame.shape[:2]
    x = w // 2
    y0, y1 = int(h * 0.03), int(h * 0.97)
    strip = frame[y0:y1, x - 2:x + 3].astype(np.float32)
    col = np.array([[235, 235, 235], [28, 24, 22], [28, 24, 22],
                    [28, 24, 22], [235, 235, 235]], np.float32)
    a = np.array([alpha_edge, alpha_core, alpha_core, alpha_core,
                  alpha_edge], np.float32)[None, :, None]
    frame[y0:y1, x - 2:x + 3] = (col[None] * a + strip * (1 - a)).astype(np.uint8)


def res_label(w, h, fps):
    name = {2160: "4K", 1440: "1440p", 1080: "1080p", 720: "720p"}.get(h, f"{h}p")
    return f"{name}  ·  {w}×{h}  ·  {fps:g} fps"


# ── Detector ─────────────────────────────────────────────────

class LiveDetector:
    """
    TURNING → SETTLING → STABLE on the same signal the offline pipeline
    uses: mean absolute difference between consecutive analysis frames,
    then a rolling mean. A capture fires once motion stays under
    `settle_threshold` for `settle_time`, and the next is armed only after
    motion exceeds `turn_threshold` — one settle can never capture twice.
    Within a settle the sharpest frame wins, mirroring the offline rule.
    """

    def __init__(self, fps, settle_threshold, turn_threshold,
                 settle_time, smoothing_window):
        self.fps = fps
        self.settle_thr, self.turn_thr = settle_threshold, turn_threshold
        self.settle_frames = max(2, int(round(settle_time * fps)))
        self.window = deque(maxlen=max(1, smoothing_window))
        self.state, self.armed, self.paused = "TURNING", True, False
        self.settle_run, self.prev_gray = 0, None
        self.raw, self.smooth, self.candidates = [], [], []
        self.turn_peak, self.peak_frames = (0.0, 0), []

    def motion(self, gray):
        m = 0.0 if self.prev_gray is None else float(
            np.mean(cv2.absdiff(self.prev_gray, gray)))
        self.prev_gray = gray
        self.window.append(m)
        return m, float(np.mean(self.window))

    def update(self, frame_idx, gray):
        raw, m = self.motion(gray)
        self.raw.append(raw)
        self.smooth.append(m)
        if m > self.turn_peak[0]:
            self.turn_peak = (m, frame_idx)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        capture = None
        if m > self.turn_thr:
            self.state, self.settle_run, self.candidates = "TURNING", 0, []
            self.armed = True
        elif m < self.settle_thr:
            self.settle_run += 1
            self.state = "SETTLING" if self.settle_run < self.settle_frames else "STABLE"
            self.candidates.append((frame_idx, sharp))
            if self.settle_run == self.settle_frames and self.armed and not self.paused:
                capture = self._take()
        else:
            self.settle_run, self.candidates = 0, []
        return capture, m

    def settle_progress(self):
        if self.state != "SETTLING" or not self.armed or self.paused:
            return 0.0
        return min(1.0, self.settle_run / self.settle_frames)

    def _take(self):
        best = max(self.candidates, key=lambda c: c[1])[0]
        self.armed = False
        if self.turn_peak[0] > 0:
            self.peak_frames.append(self.turn_peak[1])
        self.turn_peak = (0.0, 0)
        return best

    def force_capture(self, frame_idx):
        return max(self.candidates, key=lambda c: c[1])[0] if self.candidates else frame_idx

    def spreads(self, total, keyframes):
        b = [0] + [k["frame_index"] for k in keyframes] + [total]
        return [{"spread_index": i + 1, "start_frame": s, "end_frame": e,
                 "frame_count": e - s, "duration_sec": round((e - s) / self.fps, 3)}
                for i, (s, e) in enumerate(zip(b[:-1], b[1:]))]


# ── Keystone check (setup only) ──────────────────────────────

class KeystoneWatcher:
    """
    Squareness check for the camera. Runs the page detector off the UI
    thread every ~0.6 s and compares opposite edges of the page it finds:
    a camera tilted toward the table makes the near edge read longer.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.latest = None
        self.result = None       # (quad_norm, tb, lr)
        self.running = True
        self._ema = None
        threading.Thread(target=self._loop, daemon=True).start()

    def feed(self, preview):
        with self.lock:
            self.latest = preview

    def _loop(self):
        try:
            from engine.crop import detect_quad
        except Exception:
            return
        while self.running:
            time.sleep(0.6)
            with self.lock:
                img = None if self.latest is None else self.latest.copy()
            if img is None:
                continue
            try:
                quad, dbg = detect_quad(img)
            except Exception:
                quad = None
            if quad is None or dbg.get("lines_used", 0) < 3:
                with self.lock:
                    self.result = None
                continue
            h, w = img.shape[:2]
            q = np.asarray(quad, np.float64)
            tl, tr, br, bl = q
            top, bot = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
            lft, rgt = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
            m = np.array([(top - bot) / ((top + bot) / 2),
                          (lft - rgt) / ((lft + rgt) / 2)])
            self._ema = m if self._ema is None else 0.6 * self._ema + 0.4 * m
            with self.lock:
                self.result = ((q / [w, h]).tolist(), *self._ema)

    def read(self):
        with self.lock:
            return self.result

    def stop(self):
        self.running = False


# ── Camera ───────────────────────────────────────────────────

def camera_backend():
    return {"Darwin": cv2.CAP_AVFOUNDATION,
            "Windows": cv2.CAP_DSHOW}.get(platform.system(), cv2.CAP_ANY)


def open_source(source, want_w, want_h, fps):
    """Camera index, 'auto', or a video file to replay. → (cap, w, h, is_file)."""
    s = str(source)
    if s != "auto" and not s.isdigit():
        cap = cv2.VideoCapture(s)
        ok, f = cap.read()
        if not ok:
            return None, 0, 0, True
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return cap, f.shape[1], f.shape[0], True

    def try_open(i):
        cap = cv2.VideoCapture(i, camera_backend())
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, f = cap.read()
        if not ok or f is None:
            cap.release()
            return None
        return cap, f.shape[1], f.shape[0]

    if s != "auto":
        r = try_open(int(s))
        return (*r, False) if r else (None, 0, 0, False)

    best = None      # USB indices shuffle on reconnect: probe, prefer the requested mode
    for i in range(6):
        r = try_open(i)
        if r is None:
            continue
        cap, w, h = r
        if w >= want_w and h >= want_h:
            if best:
                best[0].release()
            return cap, w, h, False
        if best is None or w * h > best[1] * best[2]:
            if best:
                best[0].release()
            best = (cap, w, h)
        else:
            cap.release()
    return (*best, False) if best else (None, 0, 0, False)


# ── Composition ──────────────────────────────────────────────

class Hud:
    """Everything drawn on the frame. Kept separate from capture so it can be
    rendered headless for previews and tests."""

    HELP = [("Keys", 15, TEXT, True),
            ("Enter · Space      start capturing  (setup)", 12.5, MUTED, False),
            ("Space              pause / resume", 12.5, MUTED, False),
            ("C                  capture now", 12.5, MUTED, False),
            ("U                  undo last capture", 12.5, MUTED, False),
            ("G                  center line", 12.5, MUTED, False),
            ("K                  keystone check  (setup)", 12.5, MUTED, False),
            ("M                  mute", 12.5, MUTED, False),
            ("Q · Esc            finish", 12.5, MUTED, False)]

    def __init__(self, ph, settle_thr, turn_thr):
        self.u = ph / 720.0
        self.settle_thr, self.turn_thr = settle_thr, turn_thr
        self.full = max(turn_thr * 1.6, 1e-6)
        self.cache = SpriteCache()
        self.fill = 0.0
        self.col = np.array(STATE["SETUP"][0], np.float32)

    def compose(self, frame, *, phase, state, motion, arc, captures, elapsed,
                paused, muted, guide, show_help, res, pulse, toast, keystone):
        u = self.u
        H, W = frame.shape[:2]
        m = int(18 * u)

        if guide:
            center_line(frame)

        if keystone is not None:
            q, tb, lr = keystone
            pts = (np.array(q) * [W, H]).astype(np.int32)
            ok = max(abs(tb), abs(lr)) < 0.015
            col = STATE["STABLE"][0] if ok else STATE["SETTLING"][0]
            cv2.polylines(frame, [pts], True, col[::-1], max(1, int(2 * u)),
                          cv2.LINE_AA)

        # ── motion circle, top right, state below ──
        key = "PAUSED" if paused else ("SETUP" if phase == "setup" else state)
        target_col = np.array(STATE[key][0], np.float32)
        self.col += (target_col - self.col) * 0.25          # colors ease, never snap
        target = float(np.sqrt(np.clip(motion / self.full, 0, 1)))
        self.fill += (target - self.fill) * 0.35            # fill breathes, no jitter
        col = tuple(int(c) for c in self.col)
        spr = meter(self.fill, col, float(np.sqrt(self.settle_thr / self.full)),
                    arc, pulse, u)
        sx = W - m - spr.shape[1] + int(spr.shape[1] * 0.2)
        sy = m - int(spr.shape[0] * 0.2)
        blend(frame, spr, sx, sy)
        lbl = self.cache.get(("state", key), lambda: panel(
            [(STATE[key][1], 11.5, STATE[key][0], True)], u, "center"))
        cx = sx + spr.shape[1] // 2
        blend(frame, lbl, cx - lbl.shape[1] // 2,
              sy + int(spr.shape[0] * 0.8) + int(4 * u))

        # ── status, top left ──
        if phase == "setup":
            lines, dot = [("Setup", 15, TEXT, True), (res, 11.5, MUTED, False)], None
        else:
            mm, ss_ = divmod(int(elapsed), 60)
            lines = [(f"{mm:02d}:{ss_:02d}     {captures} captured", 15, TEXT, True),
                     (res + ("   ·   muted" if muted else ""), 11.5, MUTED, False)]
            if paused:
                dot = STATE["PAUSED"][0]
            else:   # slow blink: lit, then a dim ember — never fully off
                dot = REC_RED if int(elapsed * 1.6) % 2 == 0 else (110, 60, 58)
        tag = (tuple(t for t, *_ in lines), dot)
        p = self.cache.get(("status", tag), lambda: panel(lines, u, dot=dot))
        blend(frame, p, m, m)

        # ── bottom center: hint in setup, toasts while capturing ──
        msg = None
        if phase == "setup":
            if keystone is not None:
                q, tb, lr = keystone
                big = tb if abs(tb) >= abs(lr) else lr
                if max(abs(tb), abs(lr)) < 0.015:
                    ks = "camera looks square"
                elif abs(tb) >= abs(lr):
                    ks = f"{'top' if tb > 0 else 'bottom'} edge {abs(tb)*100:.1f}% longer"
                else:
                    ks = f"{'left' if lr > 0 else 'right'} edge {abs(lr)*100:.1f}% longer"
                msg = ("Frame the book, then press Enter to start   ·   " + ks)
            else:
                msg = "Frame the book and square up the camera, then press Enter to start"
        elif toast:
            msg = toast
        if msg:
            t = self.cache.get(("msg", msg), lambda: panel(
                [(msg, 13, TEXT, False)], u, "center"))
            blend(frame, t, (W - t.shape[1]) // 2, H - m - t.shape[0])

        hint = self.cache.get("h", lambda: panel([("H  keys", 11, MUTED, False)], u))
        blend(frame, hint, m, H - m - hint.shape[0])

        if show_help:
            hp = self.cache.get("help", lambda: panel(self.HELP, u))
            blend(frame, hp, (W - hp.shape[1]) // 2, (H - hp.shape[0]) // 2)
        return frame


# ── Main ─────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="ScanStudio live capture")
    p.add_argument("output_dir")
    p.add_argument("video_out")
    p.add_argument("--camera", default="auto",
                   help="camera index, 'auto', or a video file to replay")
    p.add_argument("--capture-width", type=int, default=3840)
    p.add_argument("--capture-height", type=int, default=2160)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--codec", default="mp4v", choices=["mp4v", "avc1"])
    p.add_argument("--analysis-height", type=int, default=360)
    p.add_argument("--preview-height", type=int, default=720)
    p.add_argument("--smoothing-window", type=int, default=15)
    p.add_argument("--settle-threshold", type=float, default=2.0)
    p.add_argument("--turn-threshold", type=float, default=5.0)
    p.add_argument("--settle-time", type=float, default=0.4)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--sound-set", default="pluck",
                   choices=["pluck", "chime", "tick", "off"])
    p.add_argument("--no-guide", action="store_true")
    args = p.parse_args()

    paths = ProjectPaths(args.output_dir)
    paths.ensure("images", "json", "data", "plots")
    status_path = paths.json / "live_status.json"

    def publish(**kw):
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(kw))
        tmp.replace(status_path)        # atomic: the UI never reads half a file

    publish(running=True, phase="opening", captures=0)
    cap, W, H, is_file = open_source(args.camera, args.capture_width,
                                     args.capture_height, args.fps)
    if cap is None:
        publish(running=False, phase="error", captures=0,
                error=f"Could not open camera {args.camera}")
        sys.exit(1)

    aw = int(W * args.analysis_height / H)
    ph = args.preview_height
    pw = int(W * ph / H)
    res = res_label(W, H, args.fps)
    hud = Hud(ph, args.settle_threshold, args.turn_threshold)
    snd = Player(args.sound_set, HERE / "sounds")
    muted = args.sound_set == "off"
    guide, show_help = not args.no_guide, False
    ks = KeystoneWatcher()
    ks_on = True

    win = "ScanStudio — Live Capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(win, pw, ph)
    delay = max(1, int(1000 / args.fps)) if is_file else 1

    # ── setup: live view only, nothing recorded ──
    setup_det = LiveDetector(args.fps, args.settle_threshold,
                             args.turn_threshold, args.settle_time,
                             args.smoothing_window)
    started, last_pub = False, 0.0
    frame = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        preview = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(cv2.resize(preview, (aw, args.analysis_height),
                                       interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        _, mot = setup_det.motion(gray)
        if ks_on:
            ks.feed(preview)
        out = hud.compose(preview, phase="setup", state="SETUP", motion=mot,
                          arc=0.0, captures=0, elapsed=0, paused=False,
                          muted=muted, guide=guide, show_help=show_help,
                          res=res, pulse=None, toast=None,
                          keystone=ks.read() if ks_on else None)
        cv2.imshow(win, out)
        now = time.time()
        if now - last_pub > 0.5:
            publish(running=True, phase="setup", captures=0)
            last_pub = now
        k = cv2.waitKey(delay) & 0xFF
        if k in (13, 10, 32):
            started = True
            break
        if k in (ord("q"), 27):
            break
        if k == ord("g"):
            guide = not guide
        elif k == ord("h"):
            show_help = not show_help
        elif k == ord("k"):
            ks_on = not ks_on
        elif k == ord("m"):
            muted = not muted
            snd.muted = muted
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    ks.stop()

    if not started:
        cap.release()
        cv2.destroyAllWindows()
        publish(running=False, phase="cancelled", captures=0)
        return

    # ── capturing ──
    Path(args.video_out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc(*args.codec),
                             args.fps, (W, H))
    if not writer.isOpened():
        cap.release()
        cv2.destroyAllWindows()
        publish(running=False, phase="error", captures=0,
                error="Could not open the video writer")
        sys.exit(1)

    # Encode on a background thread so the 4K encode never stalls the preview.
    # Bounded queue: a slow encoder applies backpressure instead of eating RAM.
    fq, STOP = queue.Queue(maxsize=64), object()

    def encode():
        while True:
            item = fq.get()
            if item is STOP:
                break
            writer.write(item)

    wt = threading.Thread(target=encode, daemon=True)
    wt.start()

    det = LiveDetector(args.fps, args.settle_threshold, args.turn_threshold,
                       args.settle_time, args.smoothing_window)
    ring = deque(maxlen=det.settle_frames + 4)
    keyframes = []
    toast, toast_until, pulse_t0 = None, 0.0, None
    snd.play("start")

    def say(text, secs=1.2):
        nonlocal toast, toast_until
        toast, toast_until = text, time.time() + secs

    def commit(fi, manual):
        nonlocal pulse_t0
        img = next((f for i, f in ring if i == fi), None)
        if img is None:
            return
        fn = f"frame{fi:06d}.jpg"
        cv2.imwrite(str(paths.images / fn), img,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        start = keyframes[-1]["frame_index"] if keyframes else 0
        sm = det.smooth[fi] if fi < len(det.smooth) else 0.0
        keyframes.append({
            "frame_index": fi, "time_sec": round(fi / args.fps, 2),
            "motion_value": round(float(sm), 4),
            "sharpness": round(float(cv2.Laplacian(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()), 1),
            "filename": fn, "spread_start": start, "spread_end": fi,
            "spread_duration": round((fi - start) / args.fps, 3),
            "source": "live_manual" if manual else "live",
        })
        pulse_t0 = time.time()
        say(f"Captured #{len(keyframes)}" + ("  ·  manual" if manual else ""))
        snd.play("manual" if manual else "capture")

    # the first frame of the take is the frame that ended setup
    t0, fi, last_pub = time.time(), -1, 0.0
    pending = frame
    while True:
        if pending is not None:
            frame, pending, ok = pending, None, True
        else:
            ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        fq.put(frame)
        ring.append((fi, frame))
        preview = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(cv2.resize(preview, (aw, args.analysis_height),
                                       interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        got, mot = det.update(fi, gray)
        if got is not None:
            commit(got, manual=False)

        now = time.time()
        pulse = None
        if pulse_t0 is not None:
            pulse = (now - pulse_t0) / 0.55
            if pulse >= 1:
                pulse, pulse_t0 = None, None
        out = hud.compose(preview, phase="capturing", state=det.state,
                          motion=mot, arc=det.settle_progress(),
                          captures=len(keyframes), elapsed=now - t0,
                          paused=det.paused, muted=snd.muted, guide=guide,
                          show_help=show_help, res=res, pulse=pulse,
                          toast=toast if now < toast_until else None,
                          keystone=None)
        cv2.imshow(win, out)

        if now - last_pub > 0.4:
            publish(running=True, phase="recording", captures=len(keyframes),
                    state=det.state, motion=round(mot, 2), paused=det.paused,
                    elapsed=round(now - t0, 1), frames=fi + 1)
            last_pub = now

        k = cv2.waitKey(delay) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            det.paused = not det.paused
            say("Paused" if det.paused else "Resumed")
        elif k == ord("c"):
            commit(det.force_capture(fi), manual=True)
        elif k == ord("u") and keyframes:
            kf = keyframes.pop()
            (paths.images / kf["filename"]).unlink(missing_ok=True)
            say(f"Undone  ·  {len(keyframes)} captured")
            snd.play("undo")
        elif k == ord("g"):
            guide = not guide
        elif k == ord("h"):
            show_help = not show_help
        elif k == ord("m"):
            snd.muted = not snd.muted
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    snd.play("stop")
    publish(running=True, phase="saving", captures=len(keyframes))
    cap.release()
    fq.put(STOP)
    wt.join()
    writer.release()
    cv2.destroyAllWindows()

    total = fi + 1
    raw = np.array(det.raw, np.float64)
    smooth = np.array(det.smooth, np.float64)
    np.save(str(paths.data / "motion_signal.npy"), raw[1:] if len(raw) > 1 else raw)
    np.save(str(paths.data / "smoothed_signal.npy"),
            smooth[1:] if len(smooth) > 1 else smooth)
    np.save(str(paths.data / "peaks.npy"), np.array(det.peak_frames, np.int64))
    (paths.json / "spreads.json").write_text(
        json.dumps(det.spreads(total, keyframes), indent=2))
    (paths.json / "metadata.json").write_text(json.dumps({
        "video_path": str(args.video_out), "fps": args.fps,
        "total_frames": total, "duration_sec": total / args.fps,
        "original_width": W, "original_height": H,
        "analysis_width": aw, "analysis_height": args.analysis_height,
        "frames_processed": total, "smoothing_window": args.smoothing_window,
        "capture_source": "live",
        "live_params": {k: getattr(args, k) for k in (
            "settle_threshold", "turn_threshold", "settle_time",
            "codec", "sound_set")},
    }, indent=2))
    (paths.json / "keyframes.json").write_text(json.dumps(keyframes, indent=2))

    try:        # plots for the analysis view — best effort, never fails a take
        from engine.motion import plot_motion_signal
        from engine.peaks import plot_peaks, build_spreads
        if len(smooth) > 30:
            plot_motion_signal(raw[1:], smooth[1:], args.fps,
                               paths.plots / "motion_plot.png")
            pk = np.array(det.peak_frames, np.int64)
            if len(pk):
                plot_peaks(smooth[1:], pk,
                           build_spreads(pk, len(smooth) - 1, args.fps),
                           args.fps, paths.plots / "peaks_plot.png")
    except Exception as e:
        print(f"plot generation skipped: {e}")

    publish(running=False, phase="done", captures=len(keyframes),
            frames=total, elapsed=round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
