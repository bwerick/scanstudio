#!/usr/bin/env python3
"""
Phase 0: Live Capture (browser camera)

The same live capture as p0_live_capture, with the browser as the camera:
Chrome owns the sensor via getUserMedia and records via MediaRecorder, while
this server runs the *same* LiveDetector the local front end runs — two front
ends, one brain. Exists because ChromeOS's Linux container has no uvcvideo
and therefore no /dev/video* no matter what the USB-sharing toggles say; the
browser is the only road to the camera there. Emits the same artifacts as
p0_live_capture, so `make finish` works unchanged.

The tab streams small grayscale analysis frames (~360p, ~7 MB/s) over one
WebSocket; the detector picks keyframes by *timestamp*, and only those frames
cross the wire at full resolution, as JPEGs. The recording never crosses raw:
MediaRecorder chunks (hardware H.264/VP9) are appended to a raw file, then
normalized to constant frame rate at --fps when the session ends — P4 scrubs
the recording by frame index, so a variable-rate file would silently show the
wrong moment. Normalization uses ffmpeg when available, else OpenCV.

Timeline contract: the detector requires contiguous frame indices at --fps.
Browser frames are variable-rate, so this server maps each frame's timestamp
onto that timeline; a gap of dropped frames is credited with the difference
measured across it (a page turn that fell into a drop still changed the page,
so it still reads as motion), duplicates are dropped, and both counts land in
metadata.json.

Usage:
  python scripts/p0_web_capture.py output/mybook recordings/mybook.mp4
  ... then open http://localhost:8412 in Chrome.

  ChromeOS: Chrome runs outside the container, so forward the port first —
  Settings > Linux > Port forwarding > Add (8412). `localhost` is a secure
  context, so getUserMedia works; `penguin.linux.test` is not, and won't.

Keys (in the browser tab):
  Q       Finish and save        C       Force-capture now
  U       Undo last capture      Space   Pause / resume auto-capture
  M       Toggle capture sound
"""

import argparse
import json
import shutil
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from live_state import LiveDetector, keyframe_record
from miniws import CLOSE, TEXT, WebSocket, accept_key
from utils import ProjectPaths, log

# Binary message tags (first byte of every binary WebSocket message).
MSG_FRAME = 1    # [u8 1][f64 ts_us][u16 w][u16 h][w*h gray]      browser -> server
MSG_JPEG = 2     # [u8 2][u32 frame_index][jpeg bytes]            browser -> server
MSG_CHUNK = 3    # [u8 3][u32 seq][MediaRecorder chunk bytes]     browser -> server

_FRAME_HDR = struct.Struct("<dHH")

TIMESLICE_MS = 1000       # MediaRecorder chunk interval
# Extra full-res frames the browser rings beyond settle_frames, covering the
# round trip between an analysis frame landing here and a capture request
# arriving there. The ring holds *copies* (~12 MB each at 4K), so this is a
# memory knob, not a camera-buffer-pool one; localhost RTT is ~1 frame.
RING_SLACK = 4


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Phase 0: Live capture via browser camera")
    p.add_argument("output_dir", help="Base output directory (e.g. output/mybook)")
    p.add_argument("video_out", help="Path to write the recording (e.g. recordings/mybook.mp4)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8412)
    p.add_argument("--capture-width", type=int, default=3840,
                   help="Requested camera width (default 3840 = 4K UHD)")
    p.add_argument("--capture-height", type=int, default=2160,
                   help="Requested camera height (default 2160 = 4K UHD)")
    p.add_argument("--fps", type=float, default=30.0, help="Timeline / recording fps")
    p.add_argument("--analysis-height", type=int, default=360)
    p.add_argument("--smoothing-window", type=int, default=15)
    p.add_argument("--settle-threshold", type=float, default=2.0,
                   help="Motion below this counts as 'still'")
    p.add_argument("--turn-threshold", type=float, default=5.0,
                   help="Motion above this counts as a page turn")
    p.add_argument("--settle-time", type=float, default=0.4,
                   help="Seconds of stillness required before capturing")
    p.add_argument("--jpeg-quality", type=int, default=95,
                   help="JPEG quality for captured keyframes")
    p.add_argument("--keep-raw", action="store_true",
                   help="Keep the raw MediaRecorder file next to the normalized one")
    return p.parse_args(argv)


def normalize_recording(raw_path, out_path, fps, use_ffmpeg=None, progress=None):
    """Re-encode the variable-rate MediaRecorder file to CFR at ``fps``.

    P4 seeks the recording with CAP_PROP_POS_FRAMES, so downstream the file
    must be constant-rate on the same timeline the artifacts use. Returns a
    status string starting with "ok" on success. ``progress``, if given, is
    called with seconds of output video written so far.
    """
    if use_ffmpeg is None:
        use_ffmpeg = shutil.which("ffmpeg") is not None

    if use_ffmpeg:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-progress", "pipe:1",
               "-nostats", "-i", str(raw_path),
               "-vf", f"fps={fps:g}", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(out_path)]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
        except OSError as e:
            return f"failed: ffmpeg: {e}"
        # Drain stderr on the side so an error-spewing encode can't deadlock
        # against the stdout progress reads.
        err_chunks = []
        drain = threading.Thread(target=lambda: err_chunks.append(proc.stderr.read()),
                                 daemon=True)
        drain.start()
        # -progress emits key=value blocks every ~0.5s; out_time_us is the
        # encode position on the output timeline (older ffmpeg spells it
        # out_time_ms — also microseconds).
        for line in proc.stdout:
            key, _, val = line.strip().partition("=")
            if progress is not None and key in ("out_time_us", "out_time_ms"):
                try:
                    progress(int(val) / 1e6)
                except ValueError:
                    pass                      # "N/A" before the first frame
        proc.wait()
        drain.join(timeout=5)
        if proc.returncode != 0:
            err = (err_chunks[0] if err_chunks else "").strip()
            return f"failed: ffmpeg: {err[:400]}"
        return "ok (ffmpeg)"

    # OpenCV fallback: sequential decode, duplicating each frame until the
    # constant-rate timeline catches up with the next frame's timestamp. If the
    # backend reports no timestamps (POS_MSEC stuck at 0) this degrades to a
    # straight copy, which is still better than nothing.
    log("  (no ffmpeg — using OpenCV, which software-encodes and is slow at 4K;"
        " `make install` sets up ffmpeg)")
    cap = cv2.VideoCapture(str(raw_path))
    if not cap.isOpened():
        return "failed: OpenCV cannot read the raw recording"
    writer, prev, out_n, next_report = None, None, 0, 900
    while True:
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)   # time of the frame about to decode
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (w, h))
            if not writer.isOpened():
                cap.release()
                return "failed: OpenCV cannot open the output writer"
        while prev is not None and out_n * 1000.0 / fps < t_ms - 500.0 / fps:
            writer.write(prev)
            out_n += 1
        writer.write(frame)
        out_n += 1
        prev = frame
        if progress is not None:
            # out_n/fps covers backends whose POS_MSEC is stuck at 0.
            progress(max(t_ms / 1000.0, out_n / fps))
        elif out_n >= next_report:
            log(f"    ...{out_n} frames ({out_n / fps:.0f}s of video)")
            next_report += 900
    cap.release()
    if writer is None:
        return "failed: raw recording has no frames"
    writer.release()
    return f"ok (opencv, {out_n} frames)"


def _fmt_secs(s):
    s = int(round(s))
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


class NormalizeProgress:
    """Progress reporter for normalize_recording: an in-place terminal bar
    (periodic log lines when stdout is not a tty), mirrored to the browser
    tab as throttled "normalizing" messages carrying a fraction and an ETA."""

    def __init__(self, duration_s, send):
        self.duration_s = duration_s
        self.send = send
        self._t0 = time.monotonic()
        self._next_draw = 0.0         # wall clock of the next terminal update
        self._next_send = 0.0         # ... and the next browser update
        self._tty = sys.stdout.isatty()
        self._drawn = False

    def __call__(self, done_s):
        now = time.monotonic()
        frac = min(done_s / self.duration_s, 1.0) if self.duration_s > 0 else 0.0
        elapsed = now - self._t0
        eta = elapsed * (1.0 - frac) / frac if frac > 0.02 else None
        eta_txt = f" · ~{_fmt_secs(eta)} left" if eta is not None else ""
        if now >= self._next_draw:
            self._next_draw = now + (0.25 if self._tty else 10.0)
            if self._tty:
                n = int(frac * 30)
                print(f"\r    [{'#' * n}{'.' * (30 - n)}] {frac * 100:3.0f}% · "
                      f"{done_s:.0f}s / {self.duration_s:.0f}s of video{eta_txt}   ",
                      end="", flush=True)
                self._drawn = True
            else:
                log(f"    ...normalizing: {frac * 100:.0f}% "
                    f"({done_s:.0f}s of {self.duration_s:.0f}s){eta_txt}")
        if now >= self._next_send:
            self._next_send = now + 1.0
            self.send({"type": "normalizing", "progress": round(frac, 4),
                       "eta_s": None if eta is None else round(eta)})

    def finish_line(self):
        """End the \\r bar with a newline so the next log starts clean."""
        if self._drawn:
            print(flush=True)


class WebSession:
    """One capture session driven by a browser tab.

    Speaks the message protocol; holds the detector, the keyframes and the raw
    recording file. Deliberately socket-free — ``send`` is any callable taking
    a dict — so the whole session is testable by feeding handle_text /
    handle_binary and reading what it tried to send.
    """

    def __init__(self, cfg, paths, send):
        self.cfg = cfg
        self.paths = paths
        self._send = send
        self.det = LiveDetector(fps=cfg.fps,
                                settle_threshold=cfg.settle_threshold,
                                turn_threshold=cfg.turn_threshold,
                                settle_time=cfg.settle_time,
                                smoothing_window=cfg.smoothing_window)
        self.keyframes = []
        self.pending = {}            # frame_index -> Capture awaiting its JPEG
        self.camera = {}             # actuals from the browser's hello
        self.finished = False

        self.frames_seen = 0         # real analysis frames received
        self.gap_filled = 0
        self.dup_dropped = 0
        self.last_stats = None       # most recent browser "stats" message
        self.analysis_w = None
        self.analysis_h = None

        self._t0_us = None           # timestamp anchoring index 0
        self._last_index = -1
        self._ts_by_index = {}       # index -> browser timestamp (for captures)

        self.raw_path = None
        self._raw_file = None
        self.raw_bytes = 0
        self._chunks = 0

    # ── Outbound ─────────────────────────────────────────────

    def send(self, msg):
        """Send if the socket is still there; a dead socket must not stop
        artifact writing (finish() runs on disconnect too)."""
        try:
            self._send(msg)
        except (OSError, ConnectionError):
            pass

    def send_config(self):
        cfg = self.cfg
        self.send({
            "type": "config",
            "fps": cfg.fps,
            "capture_width": cfg.capture_width,
            "capture_height": cfg.capture_height,
            "analysis_height": cfg.analysis_height,
            "settle_threshold": cfg.settle_threshold,
            "turn_threshold": cfg.turn_threshold,
            "settle_frames": self.det.settle_frames,
            "jpeg_quality": cfg.jpeg_quality,
            "ring": self.det.settle_frames + RING_SLACK,
            "timeslice_ms": TIMESLICE_MS,
        })

    # ── Inbound ──────────────────────────────────────────────

    def handle_text(self, text):
        m = json.loads(text)
        t = m.get("type")
        if t == "hello":
            self.camera = {k: m.get(k) for k in ("width", "height", "frame_rate", "mime")}
            log(f"  Browser camera: {m.get('width')}x{m.get('height')} "
                f"@ {m.get('frame_rate')} fps, recording as {m.get('mime')}")
        elif t == "force":
            self._request_pixels(self.det.force_capture())
        elif t == "undo":
            self._undo()
        elif t == "pause":
            self.det.set_paused(bool(m.get("value")))
            log(f"  Auto-capture {'paused' if self.det.paused else 'resumed'}")
        elif t == "hidden":
            log("  WARNING: tab hidden — frame delivery may stall; "
                "keep the capture tab visible while scanning")
        elif t == "capture_note":
            log(f"  NOTE from browser: {m.get('text')}")
        elif t == "stats":
            # Browser and server share a clock (same machine / same host for
            # the ChromeOS port forward), so epoch_ms measures how long this
            # message sat behind the socket's frame/chunk backlog.
            lag_s = max(0.0, time.time() - m.get("epoch_ms", 0) / 1000.0)
            self.last_stats = m
            log(f"  [stats] cam {m.get('cam_fps')} fps · "
                f"proc {m.get('proc_ms')} ms/frame · "
                f"frame-ts step {m.get('ts_delta_ms')} ms · "
                f"skipped {m.get('skipped')} · "
                f"ws buffered {m.get('buffered', 0) / 1e6:.1f} MB · "
                f"ring {m.get('ring_mode')} · socket lag {lag_s:.2f} s")
        elif t == "capture_missing":
            fi = m.get("frame_index")
            self.pending.pop(fi, None)
            log(f"  WARNING: browser no longer had frame {fi}; capture dropped")
            self.send({"type": "notice",
                       "text": f"Capture at frame {fi} was lost — "
                               "press C to retake once the page is steady"})
        elif t == "finish":
            self.finish()

    def handle_binary(self, data):
        tag = data[0]
        if tag == MSG_FRAME:
            ts_us, w, h = _FRAME_HDR.unpack_from(data, 1)
            if len(data) < 13 + w * h:
                return
            gray = np.frombuffer(data, np.uint8, count=w * h, offset=13).reshape(h, w)
            self._on_frame(ts_us, gray)
        elif tag == MSG_JPEG:
            (fi,) = struct.unpack_from("<I", data, 1)
            self._commit_jpeg(fi, data[5:])
        elif tag == MSG_CHUNK:
            self._append_chunk(data[5:])

    # ── The analysis timeline ────────────────────────────────

    def _on_frame(self, ts_us, gray):
        """Map a browser frame onto the contiguous constant-rate timeline."""
        if self._t0_us is None:
            self._t0_us = ts_us
        index = int(round((ts_us - self._t0_us) * self.cfg.fps / 1e6))
        if index <= self._last_index:
            self.dup_dropped += 1
            if self.dup_dropped == 30:
                log("  WARNING: many frames collapsing onto the same timeline "
                    "index — the browser's frame timestamps may not be "
                    "microseconds, or the camera runs far below "
                    f"{self.cfg.fps:g} fps")
            return
        if self.analysis_w is None:
            self.analysis_h, self.analysis_w = gray.shape
        # This frame stands for every timeline slot since the last one: the
        # detector credits the whole gap with the difference measured across
        # it (see LiveDetector.update), so dropped frames never read as
        # fabricated stillness. Every slot maps to this frame's timestamp —
        # any capture landing in the gap resolves to a frame that exists.
        span = index - self._last_index
        if span - 1 > self.cfg.fps:
            log(f"  WARNING: {span - 1} frames missing before index {index} "
                f"(~{(span - 1) / self.cfg.fps:.1f}s)")
        self.gap_filled += span - 1
        self.frames_seen += 1
        for idx in range(max(index - span + 1, index - 95), index + 1):
            self._ts_by_index[idx] = ts_us
        if len(self._ts_by_index) > 192:
            for k in [k for k in self._ts_by_index if k < index - 96]:
                del self._ts_by_index[k]
        self._last_index = index

        tick = self.det.update(index, gray, span=span)
        self.send({"type": "tick", "frame_index": index, "state": tick.state,
                   "motion": round(tick.motion, 2), "smooth": round(tick.smooth, 2),
                   "brightness": round(tick.brightness, 1),
                   "captured": len(self.keyframes), "paused": self.det.paused})
        if tick.capture is not None:
            self._request_pixels(tick.capture)

    # ── Captures ─────────────────────────────────────────────

    def _request_pixels(self, capture):
        """The detector named a frame; ask the browser for its pixels."""
        if capture is None:
            return
        fi = capture.frame_index
        ts = self._ts_by_index.get(fi)
        if ts is None:
            log(f"  WARNING: no timestamp for capture at frame {fi}; dropped")
            return
        self.pending[fi] = capture
        self.send({"type": "capture", "frame_index": fi, "ts_us": ts,
                   "reason": capture.reason})

    def _commit_jpeg(self, fi, jpeg):
        capture = self.pending.pop(fi, None)
        if capture is None or not jpeg:
            return
        spread_start = self.keyframes[-1]["frame_index"] if self.keyframes else 0
        filename = f"frame{fi:06d}.jpg"
        (self.paths.images / filename).write_bytes(jpeg)
        self.keyframes.append(
            keyframe_record(capture, self.cfg.fps, spread_start, filename,
                            source="live"))
        self.send({"type": "captured", "count": len(self.keyframes),
                   "frame_index": fi, "reason": capture.reason})
        log(f"  Captured #{len(self.keyframes)}: frame {fi} "
            f"(sharp={capture.sharpness:.0f}, {capture.reason})")

    def _undo(self):
        if not self.keyframes:
            return
        kf = self.keyframes.pop()
        (self.paths.images / kf["filename"]).unlink(missing_ok=True)
        self.send({"type": "undone", "count": len(self.keyframes)})
        log(f"  Undid capture {kf['filename']}")

    # ── The recording ────────────────────────────────────────

    def _append_chunk(self, chunk):
        if self._raw_file is None:
            mime = self.camera.get("mime") or "video/webm"
            ext = "mp4" if "mp4" in mime else "webm"
            out = Path(self.cfg.video_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            self.raw_path = out.parent / f"{out.stem}.raw.{ext}"
            self._raw_file = open(self.raw_path, "wb")
        self._raw_file.write(chunk)
        self.raw_bytes += len(chunk)
        self._chunks += 1

    # ── End of session ───────────────────────────────────────

    def finish(self):
        if self.finished:
            return
        self.finished = True
        if self._raw_file is not None:
            self._raw_file.close()

        total_frames = self._last_index + 1
        det, cfg, paths = self.det, self.cfg, self.paths
        log(f"  Session over: {total_frames} timeline frames "
            f"({self.frames_seen} received, {self.gap_filled} gap-filled, "
            f"{self.dup_dropped} duplicates), {len(self.keyframes)} keyframes, "
            f"{self.raw_bytes / 1e6:.1f} MB recorded")

        np.save(str(paths.data / "motion_signal.npy"), det.motion_signal())
        np.save(str(paths.data / "smoothed_signal.npy"), det.smoothed_signal())
        # P2/P3 markers too, so `make finish` sees the front half as satisfied.
        np.save(str(paths.data / "peaks.npy"), det.peaks())
        (paths.json / "spreads.json").write_text(
            json.dumps(det.spreads(total_frames), indent=2))
        (paths.json / "keyframes.json").write_text(json.dumps(self.keyframes, indent=2))

        status = "in progress" if self.raw_path is not None else "no recording received"
        metadata = {
            "video_path": str(cfg.video_out),
            "fps": cfg.fps,
            "total_frames": total_frames,
            "duration_sec": total_frames / cfg.fps,
            "original_width": self.camera.get("width"),
            "original_height": self.camera.get("height"),
            "analysis_width": self.analysis_w,
            "analysis_height": self.analysis_h,
            "frames_processed": total_frames,
            "smoothing_window": cfg.smoothing_window,
            "capture_source": "live-web",
            "web": {
                "frames_received": self.frames_seen,
                "gap_filled": self.gap_filled,
                "duplicates_dropped": self.dup_dropped,
                "recorder_mime": self.camera.get("mime"),
                "recorder_chunks": self._chunks,
                "normalize": status,
                "raw_recording": str(self.raw_path) if self.raw_path else None,
                "last_stats": self.last_stats,
            },
        }
        meta_path = paths.json / "metadata.json"
        # Written before the normalize, which can run for minutes on a 4K
        # session — a Ctrl-C mid-encode must not cost the scan its metadata.
        meta_path.write_text(json.dumps(metadata, indent=2))

        if self.raw_path is not None:
            self.send({"type": "normalizing"})
            log(f"  Normalizing recording to CFR {cfg.fps:g} fps -> {cfg.video_out}")
            reporter = NormalizeProgress(total_frames / cfg.fps, self.send)
            status = normalize_recording(self.raw_path, cfg.video_out, cfg.fps,
                                         progress=reporter)
            reporter.finish_line()
            log(f"  Normalize: {status}")
            if status.startswith("ok") and not cfg.keep_raw:
                self.raw_path.unlink(missing_ok=True)
                raw_note = "removed after normalize"
            else:
                raw_note = str(self.raw_path)
            metadata["web"]["normalize"] = status
            metadata["web"]["raw_recording"] = raw_note
            meta_path.write_text(json.dumps(metadata, indent=2))

        self.send({"type": "finished", "keyframes": len(self.keyframes),
                   "total_frames": total_frames, "video": str(cfg.video_out),
                   "normalize": status})
        log(f"  Wrote {len(self.keyframes)} keyframes -> {paths.json / 'keyframes.json'}")


class CaptureServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, cfg, paths, html_path):
        super().__init__(addr, CaptureHandler)
        self.cfg = cfg
        self.paths = paths
        self.html_path = html_path
        self.session_lock = threading.Lock()
        self.session = None
        self.done = threading.Event()


class CaptureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):     # keep utils.log the only voice
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = self.server.html_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/ws":
            self._websocket()
        else:
            self.send_error(404)

    def _websocket(self):
        key = self.headers.get("Sec-WebSocket-Key")
        if key is None or "websocket" not in self.headers.get("Upgrade", "").lower():
            self.send_error(400, "WebSocket upgrade expected")
            return
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept_key(key))
        self.end_headers()
        self.wfile.flush()
        self.close_connection = True

        ws = WebSocket(self.connection)
        if not self.server.session_lock.acquire(blocking=False):
            ws.send_text(json.dumps({"type": "error",
                                     "message": "another capture session is already connected"}))
            ws.close()
            return
        try:
            self._run_session(ws)
        finally:
            self.server.session_lock.release()

    def _run_session(self, ws):
        srv = self.server
        session = WebSession(srv.cfg, srv.paths,
                             send=lambda m: ws.send_text(json.dumps(m)))
        srv.session = session
        log("  Browser connected")
        session.send_config()
        try:
            while not session.finished:
                op, payload = ws.recv()
                if op == CLOSE:
                    break
                if op == TEXT:
                    session.handle_text(payload.decode())
                else:
                    session.handle_binary(payload)
        except (ConnectionError, OSError) as e:
            log(f"  Connection lost: {e}")
        except Exception as e:      # a protocol bug must still salvage the scan
            log(f"  ERROR in session: {e!r}")
        if not session.finished:
            if session.frames_seen > 0 or session.raw_bytes > 0:
                log("  Browser disconnected mid-session — salvaging what arrived.")
                session.finish()
            else:
                # Nothing happened (a reload before starting); allow a retry.
                srv.session = None
                log("  Browser disconnected before capturing; waiting for a new connection")
                return
        ws.close()
        srv.done.set()


def build_server(args):
    """Construct the ready-to-serve HTTP server (separate from main for tests)."""
    paths = ProjectPaths(args.output_dir)
    paths.ensure("images", "json", "data", "plots")
    html_path = Path(__file__).resolve().parent.parent / "web" / "capture.html"
    if not html_path.exists():
        raise FileNotFoundError(f"capture page not found: {html_path}")
    return CaptureServer((args.host, args.port), args, paths, html_path)


def main():
    args = parse_args()
    log("=" * 60)
    log("PHASE 0: Live Capture (browser)")
    log("=" * 60)
    try:
        server = build_server(args)
    except (FileNotFoundError, OSError) as e:
        log(f"ERROR: {e}")
        sys.exit(1)
    port = server.server_address[1]
    log(f"Recording to {args.video_out}; artifacts under {args.output_dir}")
    log(f"Thresholds: settle<{args.settle_threshold} turn>{args.turn_threshold}, "
        f"settle_time={args.settle_time}s")
    log("")
    log(f"  Open   http://localhost:{port}   in Chrome and click Start.")
    log("")
    log("  ChromeOS: Chrome runs outside the Linux container — add port "
        f"{port} under Settings > Linux > Port forwarding first.")
    log("  (localhost is a secure context; penguin.linux.test is not.)")

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        server.done.wait()
    except KeyboardInterrupt:
        log("\nInterrupted.")
        session = server.session
        if session is not None and not session.finished and \
                (session.frames_seen or session.raw_bytes):
            session.finish()
    server.shutdown()

    session = server.session
    if session is not None and session.finished:
        log("")
        log("PHASE 0 COMPLETE")
        # finish-web keeps the reviews in the same browser (and the same
        # forwarded port) this capture just used — the ChromeOS-native path.
        log(f"  Next: make finish-web VIDEO={args.video_out}")


if __name__ == "__main__":
    main()
