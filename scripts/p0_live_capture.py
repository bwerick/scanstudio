#!/usr/bin/env python3
"""
Phase 0: Live Capture (webcam)

Real-time alternative to P1-P3. Records the webcam to a video file while an
online state machine watches the motion signal and auto-captures a keyframe
each time the book settles after a page turn. Emits the same artifacts the
offline front half produces, so P4-P9 work unchanged:

  recordings/<name>.mp4        the principal recording (byproduct)
  output/<name>/images/        full-res keyframe per spread
  output/<name>/json/keyframes.json
  output/<name>/json/metadata.json
  output/<name>/data/{motion_signal,smoothed_signal}.npy

Usage:
  python scripts/p0_live_capture.py output/mybook recordings/mybook.mp4
  python scripts/p0_live_capture.py output/mybook recordings/mybook.mp4 --camera 1

Keys (in the live window):
  Q / Esc   Quit and save
  U         Undo last capture
  C         Force-capture the current frame now
  Space     Pause / resume auto-capture
  M         Toggle capture sound mute
"""

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

from utils import log, ProjectPaths


def laplacian_sharpness(gray):
    h, w = gray.shape
    center = gray[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
    return float(cv2.Laplacian(center, cv2.CV_64F).var())


def build_spreads(peaks, total_len, fps):
    """Spread list in the same shape P2 emits (boundaries between page turns)."""
    if len(peaks) == 0:
        bounds = [(0, total_len)]
    else:
        bounds = [(0, int(peaks[0]))]
        bounds += [(int(peaks[i]), int(peaks[i + 1])) for i in range(len(peaks) - 1)]
        bounds.append((int(peaks[-1]), total_len))
    return [{"spread_index": i + 1, "start_frame": s, "end_frame": e,
             "frame_count": e - s, "duration_sec": round((e - s) / fps, 3),
             "start_time": round(s / fps, 2), "end_time": round(e / fps, 2)}
            for i, (s, e) in enumerate(bounds)]


def resolution_label(w, h):
    """Short, friendly name for a capture resolution (e.g. '4K', '1080p')."""
    for std_h, name in ((2160, "4K"), (1440, "1440p"), (1080, "1080p"),
                        (720, "720p"), (480, "480p")):
        if abs(h - std_h) <= 16:
            return name
    return f"{h}p"


def draw_overlay(disp, state, motion, smooth, settle_thr, turn_thr,
                 count, paused, flash_text, flash_until, muted=False, res_text=""):
    """Draw the live HUD onto the (downscaled) preview frame, in place.

    Layout is authored for a 540px-tall preview and scaled by ``fs`` so the HUD
    stays proportional at any --preview-height. ``res_text`` is the capture
    resolution string (built from the recording size, not ``disp``'s size).
    """
    h, w = disp.shape[:2]
    fs = h / 540.0

    def sc(v):
        return int(round(v * fs))

    th = max(1, sc(2))          # default stroke for headline text
    thin = max(1, sc(1))        # stroke for small text / lines

    # Top status bar
    bar_h = sc(64)
    cv2.rectangle(disp, (0, 0), (w, bar_h), (0, 0, 0), -1)
    state_color = {"WAITING": (0, 200, 255), "SETTLED": (0, 220, 0),
                   "TURNING": (0, 140, 255)}.get(state, (200, 200, 200))
    cv2.putText(disp, f"{state}", (sc(10), sc(24)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * fs, state_color, th)
    cv2.putText(disp, f"captured: {count}", (sc(10), sc(46)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * fs, (230, 230, 230), thin)
    if paused:
        cv2.putText(disp, "PAUSED", (w - sc(95), sc(24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * fs, (0, 0, 255), th)
    # Recording resolution — persistent, right-aligned so it reads at a glance
    if res_text:
        (tw, _), _ = cv2.getTextSize(res_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * fs, th)
        cv2.putText(disp, res_text, (w - tw - sc(10), sc(46)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * fs, (255, 255, 0), th)

    # Motion bar (maps motion onto a fixed scale relative to turn threshold)
    bx, by, bw, bh = sc(150), sc(14), w - sc(300), sc(16)
    scale = max(turn_thr * 1.6, motion, 1e-3)
    cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), (60, 60, 60), thin)
    fill = int(bw * min(smooth / scale, 1.0))
    cv2.rectangle(disp, (bx, by), (bx + fill, by + bh), state_color, -1)
    for thr, col in ((settle_thr, (0, 220, 0)), (turn_thr, (0, 140, 255))):
        x = bx + int(bw * min(thr / scale, 1.0))
        cv2.line(disp, (x, by - sc(3)), (x, by + bh + sc(3)), col, thin)
    cv2.putText(disp, f"motion {smooth:4.1f}", (bx, by + bh + sc(14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4 * fs, (200, 200, 200), thin)

    # Help line — in the status bar so it's always visible
    mute_label = "M unmute" if muted else "M mute"
    cv2.putText(disp, f"Q quit  |  U undo  |  C capture  |  Space pause  |  {mute_label}",
                (sc(10), sc(60)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * fs, (160, 160, 160), thin)

    # Capture flash
    if time.time() < flash_until and flash_text:
        (tw, _), _ = cv2.getTextSize(flash_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0 * fs, th)
        cv2.putText(disp, flash_text, (w // 2 - tw // 2, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0 * fs, (0, 255, 0), th)
    return disp


def open_camera(requested, want_w, want_h, fps):
    """Open the capture camera at ``want_w``x``want_h``; return (cap, w, h).

    USB camera indices shuffle as devices connect/disconnect (Continuity Camera,
    the built-in FaceTime cam, external webcams), so a fixed index is unreliable.
    ``requested="auto"`` scans indices and takes the first that actually delivers
    the requested resolution, falling back to the highest-resolution camera
    found. Pass an integer to force a specific index. Returns ``(None, 0, 0)`` if
    nothing opens.
    """
    # macOS exposes a webcam's high-res modes only through AVFoundation; the
    # default backend silently tops out at 1080p on some cameras.
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY

    def try_open(idx):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            return None
        # Request the capture mode *before* the first read; a UVC camera
        # negotiates to its nearest supported mode, so read back what we got.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(want_w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(want_h))
        cap.set(cv2.CAP_PROP_FPS, float(fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return None
        return cap, frame.shape[1], frame.shape[0]

    if str(requested) != "auto":
        return try_open(int(requested)) or (None, 0, 0)

    best = None  # (area, cap, w, h, idx)
    for idx in range(5):
        opened = try_open(idx)
        if opened is None:
            continue
        cap, w, h = opened
        if w >= want_w and h >= want_h:   # meets the request — take it, stop scanning
            if best is not None:
                best[1].release()
            log(f"  Auto-selected camera {idx}: {w}x{h}")
            return cap, w, h
        if best is None or w * h > best[0]:
            if best is not None:
                best[1].release()
            best = (w * h, cap, w, h, idx)
        else:
            cap.release()
    if best is None:
        return None, 0, 0
    log(f"  Auto-selected camera {best[4]}: {best[2]}x{best[3]} "
        f"(none met {want_w}x{want_h})")
    return best[1], best[2], best[3]


def main():
    p = argparse.ArgumentParser(description="Phase 0: Live webcam capture")
    p.add_argument("output_dir", help="Base output directory (e.g. output/mybook)")
    p.add_argument("video_out", help="Path to write the recording (e.g. recordings/mybook.mp4)")
    p.add_argument("--camera", default="auto",
                   help="Camera index, or 'auto' to pick whichever delivers the "
                        "requested resolution (USB indices shuffle on reconnect)")
    p.add_argument("--capture-width", type=int, default=3840,
                   help="Requested capture width (default 3840 = 4K UHD)")
    p.add_argument("--capture-height", type=int, default=2160,
                   help="Requested capture height (default 2160 = 4K UHD)")
    p.add_argument("--fps", type=float, default=30.0, help="Recording / timing fps")
    p.add_argument("--analysis-height", type=int, default=360)
    p.add_argument("--preview-height", type=int, default=1080,
                   help="Height of the on-screen preview. The full-res frame is "
                        "still recorded and captured; this only shrinks what gets "
                        "drawn and blitted to the window.")
    p.add_argument("--smoothing-window", type=int, default=15)
    p.add_argument("--settle-threshold", type=float, default=2.0,
                   help="Motion below this counts as 'still'")
    p.add_argument("--turn-threshold", type=float, default=5.0,
                   help="Motion above this counts as a page turn")
    p.add_argument("--settle-time", type=float, default=0.4,
                   help="Seconds of stillness required before capturing")
    p.add_argument("--jpeg-quality", type=int, default=95,
                   help="JPEG quality for captured keyframes (one near-lossless generation)")
    args = p.parse_args()

    log("=" * 60)
    log("PHASE 0: Live Capture")
    log("=" * 60)

    paths = ProjectPaths(args.output_dir)
    paths.ensure("images", "json", "data", "plots")

    cap, orig_w, orig_h = open_camera(args.camera, args.capture_width,
                                      args.capture_height, args.fps)
    if cap is None:
        log(f"ERROR: Cannot open camera {args.camera}")
        sys.exit(1)
    if (orig_w, orig_h) != (args.capture_width, args.capture_height):
        log(f"WARNING: got {orig_w}x{orig_h}, not the requested "
            f"{args.capture_width}x{args.capture_height}. No connected camera offers "
            f"that mode — run `make probe-camera` to see what each one supports.")
    scale = args.analysis_height / orig_h
    aw = int(orig_w * scale)
    ph = args.preview_height
    pw = int(orig_w * ph / orig_h)
    res_label = resolution_label(orig_w, orig_h)
    res_text = f"{res_label}  {orig_w}x{orig_h}"
    log(f"Camera {args.camera}: {orig_w}x{orig_h} ({res_label})  "
        f"analysis {aw}x{args.analysis_height}  preview {pw}x{ph}")
    log(f"Recording to {args.video_out}")
    log(f"Thresholds: settle<{args.settle_threshold} turn>{args.turn_threshold}, "
        f"settle_time={args.settle_time}s")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.video_out, fourcc, args.fps, (orig_w, orig_h))
    if not writer.isOpened():
        log(f"ERROR: Cannot open VideoWriter for {args.video_out}")
        sys.exit(1)

    # Encode on a background thread so the 4K software encode never stalls the
    # capture/preview loop. Frames are queued read-only: cap.read() hands back a
    # fresh array each call and the overlay draws on a separate preview image, so
    # the worker can encode a frame while the main loop still holds the same
    # buffer in its sharpness window. The queue is bounded — if the encoder falls
    # behind, put() blocks (backpressure) instead of growing memory unbounded.
    frame_queue: "queue.Queue" = queue.Queue(maxsize=64)
    WRITER_STOP = object()

    def writer_worker():
        while True:
            item = frame_queue.get()
            if item is WRITER_STOP:
                frame_queue.task_done()
                break
            writer.write(item)
            frame_queue.task_done()

    writer_thread = threading.Thread(target=writer_worker, daemon=True)
    writer_thread.start()

    settle_frames = max(1, int(args.settle_time * args.fps))
    buf = deque(maxlen=settle_frames)   # (frame_index, frame_bgr, sharpness)

    diffs = []                  # one motion value per recorded frame
    keyframes = []
    prev_small = None
    frame_idx = -1
    state = "WAITING"
    still_run = 0               # consecutive low-motion frames
    saw_turn = False
    turn_frames = []            # frame index of each detected page turn (for peaks.npy)
    paused = False
    muted = False
    flash_text, flash_until = "", 0.0

    def play_ding():
        if not muted:
            subprocess.Popen(
                ["afplay", "/System/Library/Sounds/Glass.aiff"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

    def commit_capture(reason):
        nonlocal flash_text, flash_until
        if not buf:
            return
        fi, best_frame, best_sharp = max(buf, key=lambda b: b[2])
        spread_start = keyframes[-1]["frame_index"] if keyframes else 0
        filename = f"frame{fi:06d}.jpg"
        cv2.imwrite(str(paths.images / filename), best_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        keyframes.append({
            "frame_index": fi,
            "time_sec": round(fi / args.fps, 2),
            "motion_value": round(float(diffs[fi]) if fi < len(diffs) else 0.0, 4),
            "sharpness": round(best_sharp, 1),
            "filename": filename,
            "spread_start": spread_start,
            "spread_end": fi,
            "spread_duration": round((fi - spread_start) / args.fps, 3),
            "source": "live",
        })
        flash_text = f"CAPTURED #{len(keyframes)} ({reason})"
        flash_until = time.time() + 0.8
        log(f"  Captured #{len(keyframes)}: frame {fi} "
            f"(sharp={best_sharp:.0f}, {reason})")
        play_ding()

    def undo_capture():
        nonlocal flash_text, flash_until
        if not keyframes:
            return
        kf = keyframes.pop()
        (paths.images / kf["filename"]).unlink(missing_ok=True)
        flash_text = f"UNDO #{len(keyframes) + 1}"
        flash_until = time.time() + 0.8
        log(f"  Undid capture {kf['filename']}")

    win = "ScanStudio - Live Capture"
    # KEEPRATIO so the frame is never stretched (or implicitly cropped) to fill
    # the window, and size the window to the actual preview so it opens crisp and
    # 1:1 — the full frame, just smaller — instead of being upscaled to fill a
    # default/stale window size.
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(win, pw, ph)
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            log("Camera stream ended.")
            break

        frame_queue.put(frame)      # background thread does the 4K encode
        frame_idx += 1

        # One expensive 4K downscale feeds both the preview and the analysis
        # frame; the analysis frame is then derived from the preview (a cheap
        # small->smaller resize) rather than touching the 4K array a second time.
        preview = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        if ph >= args.analysis_height:
            small = cv2.resize(preview, (aw, args.analysis_height), interpolation=cv2.INTER_AREA)
        else:
            small = cv2.resize(frame, (aw, args.analysis_height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        motion = float(np.mean(cv2.absdiff(prev_small, gray))) if prev_small is not None else 0.0
        prev_small = gray
        diffs.append(motion)

        # Short trailing mean for stable thresholding (online smoothing)
        win_n = min(args.smoothing_window, len(diffs))
        smooth = float(np.mean(diffs[-win_n:]))

        # cap.read() returns a fresh array each call, so the rolling sharpness
        # window can hold the frame directly — no per-frame full-res copy.
        buf.append((frame_idx, frame, laplacian_sharpness(gray)))

        if not paused:
            still = smooth < args.settle_threshold
            still_run = still_run + 1 if still else 0

            if state == "WAITING":
                if still and still_run >= settle_frames:
                    commit_capture("initial")
                    state = "SETTLED"
            elif state == "SETTLED":
                if smooth > args.turn_threshold:
                    state = "TURNING"
                    saw_turn = True
                    turn_frames.append(frame_idx)
            elif state == "TURNING":
                if still and still_run >= settle_frames and saw_turn:
                    commit_capture("settle")
                    state = "SETTLED"
                    saw_turn = False

        # Draw the HUD straight onto the downscaled preview (a fresh array each
        # frame), so there's no full-res copy and imshow blits a small image.
        disp = draw_overlay(preview, state, motion, smooth,
                            args.settle_threshold, args.turn_threshold,
                            len(keyframes), paused, flash_text, flash_until, muted,
                            res_text)
        cv2.imshow(win, disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("u"):
            undo_capture()
        elif key == ord("c"):
            commit_capture("manual")
            state = "SETTLED"
            saw_turn = False
        elif key == ord(" "):
            paused = not paused
            still_run = 0
        elif key == ord("m"):
            muted = not muted
            log(f"  Sound {'muted' if muted else 'unmuted'}")

    cap.release()
    frame_queue.put(WRITER_STOP)    # drain remaining frames, then stop the worker
    writer_thread.join()
    writer.release()
    cv2.destroyAllWindows()

    total_frames = frame_idx + 1
    elapsed = time.time() - t0
    log(f"Recorded {total_frames} frames in {elapsed:.1f}s "
        f"({total_frames / max(elapsed, 1e-3):.0f} fps), {len(keyframes)} keyframes")

    diffs_arr = np.array(diffs, dtype=np.float64)
    smoothed = uniform_filter1d(diffs_arr, size=args.smoothing_window) if len(diffs_arr) else diffs_arr
    np.save(str(paths.data / "motion_signal.npy"), diffs_arr)
    np.save(str(paths.data / "smoothed_signal.npy"), smoothed)

    # Emit the P2/P3 markers too, so `make finish`/`make all` see the front-half
    # dependency chain as satisfied and don't try to regenerate these keyframes.
    peaks_arr = np.array(sorted(turn_frames), dtype=np.int64)
    np.save(str(paths.data / "peaks.npy"), peaks_arr)
    spreads = build_spreads(peaks_arr, total_frames, args.fps)
    (paths.json / "spreads.json").write_text(json.dumps(spreads, indent=2))

    metadata = {
        "video_path": str(args.video_out),
        "fps": args.fps,
        "total_frames": total_frames,
        "duration_sec": total_frames / args.fps,
        "original_width": orig_w,
        "original_height": orig_h,
        "analysis_width": aw,
        "analysis_height": args.analysis_height,
        "frames_processed": total_frames,
        "smoothing_window": args.smoothing_window,
        "capture_source": "live",
    }
    (paths.json / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (paths.json / "keyframes.json").write_text(json.dumps(keyframes, indent=2))

    log(f"  Wrote {len(keyframes)} keyframes -> {paths.json / 'keyframes.json'}")
    log("")
    log("PHASE 0 COMPLETE")
    log(f"  Next: make finish VIDEO={args.video_out}")


if __name__ == "__main__":
    main()
