"""
ScanStudio Web UI

Single-page app for the book scanning pipeline.
Run: python ui/app.py
Open: http://localhost:5050
"""

import json
import os
import threading
from pathlib import Path

from flask import Flask, render_template, jsonify, request, send_file, send_from_directory

from engine import ProjectPaths, derive_output_dir
from engine.motion import compute_motion_signal, smooth_signal, plot_motion_signal
from engine.peaks import detect_peaks, rescue_missed_turns, build_spreads, plot_peaks
from engine.keyframes import select_all_keyframes, extract_frame
from engine.crop import (preview_crop, crop_otsu, crop_grabcut, apply_side_trim,
                          crop_manual, detect_crop_rect, detect_crop_quad,
                          normalize_quad, normalize_gutter, resolve_crop_quad,
                          resolve_gutter, quad_to_rect)
from engine.geometry import (quad_from_rect, gutter_endpoints,
                             gutter_from_points, remap_gutter, DEFAULT_GUTTER)
from engine.split import split_all_pages
from engine.pdf import build_pdf, build_binarized_pdf

import numpy as np

app = Flask(__name__)

# Project root (parent of ui/)
PROJECT_ROOT = Path(__file__).parent.parent
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
OUTPUT_DIR = PROJECT_ROOT / "output"

# In-memory progress tracking for long-running tasks
task_progress = {}


# ── Pages ─────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── API: Projects ─────────────────────────────────────────────

@app.route("/api/projects")
def list_projects():
    """List all existing projects with their status."""
    projects = []
    if OUTPUT_DIR.exists():
        for d in sorted(OUTPUT_DIR.iterdir()):
            if d.is_dir():
                paths = ProjectPaths(d)
                projects.append({
                    "name": d.name,
                    "path": str(d),
                    "has_motion": (paths.data / "motion_signal.npy").exists(),
                    "has_peaks": (paths.data / "peaks.npy").exists(),
                    "has_keyframes": (paths.json / "keyframes.json").exists(),
                    "has_pages": (paths.json / "pages.json").exists(),
                    "has_pdf": (paths.pdf / "book.pdf").exists(),
                    "keyframe_count": _count_keyframes(paths),
                    "page_count": _count_pages(paths),
                })
    return jsonify(projects)


@app.route("/api/analysis/<project>")
def get_analysis(project):
    """Plots available for a project, plus headline numbers."""
    paths = ProjectPaths(OUTPUT_DIR / project)
    known = [
        ("motion_plot", "Motion Signal",
         "Frame-to-frame difference across the whole video"),
        ("peaks_plot", "Page Turn Detection",
         "Detected turns, spread segmentation, and scan pacing"),
        ("selection_plot", "Keyframe Selection",
         "Which frame was chosen inside each spread"),
        ("comparison_plot", "Review Comparison",
         "Algorithm picks versus the corrected set"),
    ]
    plots = [{"name": n, "title": t, "desc": d}
             for n, t, d in known if (paths.plots / f"{n}.png").exists()]

    stats = {}
    meta_p = paths.json / "metadata.json"
    if meta_p.exists():
        md = json.loads(meta_p.read_text())
        stats["duration_sec"] = round(md.get("duration_sec", 0))
        stats["fps"] = md.get("fps")
        stats["resolution"] = f"{md.get('original_width')}×{md.get('original_height')}"
        stats["total_frames"] = md.get("total_frames")
    for key, f in (("keyframes", "keyframes.json"), ("pages", "pages.json")):
        fp = paths.json / f
        if fp.exists():
            stats[key] = len(json.loads(fp.read_text()))

    # storage actually on disk
    def dir_mb(d):
        if not d.exists():
            return 0
        return round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                     / 1024 / 1024, 1)
    stats["storage"] = {"images": dir_mb(paths.images),
                        "pages": dir_mb(paths.pages),
                        "pdf": dir_mb(paths.pdf)}
    return jsonify({"plots": plots, "stats": stats})


@app.route("/api/recordings")
def list_recordings():
    """List available video files."""
    recordings = []
    if RECORDINGS_DIR.exists():
        for f in sorted(RECORDINGS_DIR.iterdir()):
            if f.suffix.lower() in (".mp4", ".mov", ".avi"):
                stat = f.stat()
                recordings.append({
                    "name": f.name,
                    "path": str(f),
                    "size_mb": round(stat.st_size / 1024 / 1024, 1),
                })
    return jsonify(recordings)


# ── API: Processing ───────────────────────────────────────────

@app.route("/api/process/motion", methods=["POST"])
def run_motion():
    """Run Phase 1: Motion signal computation."""
    data = request.json
    video_path = data["video_path"]
    project_name = Path(video_path).stem
    paths = ProjectPaths(OUTPUT_DIR / project_name)
    paths.ensure("data", "json", "plots")

    task_id = f"motion_{project_name}"
    task_progress[task_id] = {"phase": "motion", "progress": 0, "status": "running"}

    def run():
        try:
            def on_progress(frame, total):
                task_progress[task_id]["progress"] = round(frame / total * 100)

            diffs, metadata = compute_motion_signal(video_path, on_progress=on_progress)
            smoothed = smooth_signal(diffs)
            metadata["smoothing_window"] = 15

            np.save(str(paths.data / "motion_signal.npy"), diffs)
            np.save(str(paths.data / "smoothed_signal.npy"), smoothed)
            (paths.json / "metadata.json").write_text(json.dumps(metadata, indent=2))
            plot_motion_signal(diffs, smoothed, metadata["fps"],
                               paths.plots / "motion_plot.png")

            task_progress[task_id] = {"phase": "motion", "progress": 100, "status": "done"}
        except Exception as e:
            task_progress[task_id] = {"phase": "motion", "progress": 0,
                                      "status": "error", "error": str(e)}

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"task_id": task_id, "project": project_name})


@app.route("/api/process/peaks", methods=["POST"])
def run_peaks():
    """Run Phase 2: Peak detection."""
    data = request.json
    project = data["project"]
    paths = ProjectPaths(OUTPUT_DIR / project)
    paths.ensure("data", "json", "plots")

    smoothed = np.load(str(paths.data / "smoothed_signal.npy"))
    metadata = json.loads((paths.json / "metadata.json").read_text())
    fps = metadata["fps"]

    peaks_arr = detect_peaks(smoothed, fps,
                              height=data.get("peak_height", 5.0),
                              distance_sec=data.get("min_distance", 1.5),
                              prominence=data.get("prominence", 3.0))
    peaks_arr = rescue_missed_turns(smoothed, fps, peaks_arr)
    spreads = build_spreads(peaks_arr, len(smoothed), fps)

    np.save(str(paths.data / "peaks.npy"), peaks_arr)
    (paths.json / "spreads.json").write_text(json.dumps(spreads, indent=2))
    plot_peaks(smoothed, peaks_arr, spreads, fps, paths.plots / "peaks_plot.png")

    return jsonify({
        "peaks": len(peaks_arr),
        "spreads": len(spreads),
        "median_duration": round(float(np.median([s["duration_sec"] for s in spreads])), 2),
    })


@app.route("/api/process/keyframes", methods=["POST"])
def run_keyframes():
    """Run Phase 3: Keyframe selection."""
    data = request.json
    project = data["project"]
    video_path = data["video_path"]
    paths = ProjectPaths(OUTPUT_DIR / project)
    paths.ensure("images", "json", "plots")

    spreads = json.loads((paths.json / "spreads.json").read_text())
    smoothed = np.load(str(paths.data / "smoothed_signal.npy"))
    metadata = json.loads((paths.json / "metadata.json").read_text())

    task_id = f"keyframes_{project}"
    task_progress[task_id] = {"phase": "keyframes", "progress": 0, "status": "running"}

    def run():
        try:
            def on_progress(current, total):
                task_progress[task_id]["progress"] = round(current / total * 100)

            kf_data = select_all_keyframes(
                video_path, spreads, smoothed, metadata["fps"],
                paths.images, on_progress=on_progress,
            )
            (paths.json / "keyframes.json").write_text(json.dumps(kf_data, indent=2))
            task_progress[task_id] = {"phase": "keyframes", "progress": 100,
                                       "status": "done", "count": len(kf_data)}
        except Exception as e:
            task_progress[task_id] = {"phase": "keyframes", "progress": 0,
                                      "status": "error", "error": str(e)}

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"task_id": task_id})


@app.route("/api/progress/<task_id>")
def get_progress(task_id):
    """Poll task progress."""
    return jsonify(task_progress.get(task_id, {"status": "unknown"}))


# ── API: Keyframe data ────────────────────────────────────────

@app.route("/api/keyframes/<project>")
def get_keyframes(project):
    """Get keyframe list for a project."""
    paths = ProjectPaths(OUTPUT_DIR / project)
    kfs, _ = _load_keyframes(paths)
    return jsonify(kfs)


@app.route("/api/keyframes/<project>/delete", methods=["POST"])
def delete_keyframe(project):
    """Delete a keyframe by frame_index."""
    data = request.json
    frame_index = data["frame_index"]
    paths = ProjectPaths(OUTPUT_DIR / project)

    keyframes = json.loads((paths.json / "keyframes.json").read_text())
    deleted = None
    remaining = []
    for kf in keyframes:
        if kf["frame_index"] == frame_index:
            deleted = kf
            img_path = paths.images / kf["filename"]
            if img_path.exists():
                img_path.unlink()
        else:
            remaining.append(kf)

    (paths.json / "keyframes.json").write_text(json.dumps(remaining, indent=2))
    _append_review_log(paths, {"action": "delete", "frame_index": frame_index,
                                 "reason": data.get("reason", "manual")})

    return jsonify({"deleted": deleted is not None, "remaining": len(remaining)})


@app.route("/api/keyframes/<project>/insert", methods=["POST"])
def insert_keyframe(project):
    """Insert a keyframe from a video frame."""
    data = request.json
    frame_index = data["frame_index"]
    video_path = data["video_path"]
    paths = ProjectPaths(OUTPUT_DIR / project)

    frame = extract_frame(video_path, frame_index)
    if frame is None:
        return jsonify({"error": "Could not read frame"}), 400

    filename = f"frame{frame_index:06d}.jpg"
    import cv2
    cv2.imwrite(str(paths.images / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    smoothed = np.load(str(paths.data / "smoothed_signal.npy"))
    metadata = json.loads((paths.json / "metadata.json").read_text())
    motion = float(smoothed[min(frame_index, len(smoothed) - 1)])

    new_kf = {
        "frame_index": frame_index,
        "time_sec": round(frame_index / metadata["fps"], 2),
        "motion_value": round(motion, 4),
        "sharpness": 0.0,
        "filename": filename,
        "source": "manual_insert",
    }

    keyframes = json.loads((paths.json / "keyframes.json").read_text())
    keyframes.append(new_kf)
    keyframes.sort(key=lambda k: k["frame_index"])
    (paths.json / "keyframes.json").write_text(json.dumps(keyframes, indent=2))

    _append_review_log(paths, {"action": "insert", "frame_index": frame_index})

    return jsonify(new_kf)


@app.route("/api/keyframes/<project>/update", methods=["POST"])
def update_keyframe(project):
    """Update keyframe metadata (gutter, cover, doc_start)."""
    data = request.json
    frame_index = data["frame_index"]
    paths = ProjectPaths(OUTPUT_DIR / project)

    keyframes = json.loads((paths.json / "keyframes.json").read_text())
    for kf in keyframes:
        if kf["frame_index"] == frame_index:
            for key in ("is_cover", "is_doc_start", "gutter", "gutter_pct",
                        "crop_bounds", "crop_rect", "crop_quad", "validated"):
                if key not in data:
                    continue
                old_val = kf.get(key)
                new_val = data[key]

                entry = {"action": "clear" if new_val is None else "set",
                         "field": key, "frame_index": frame_index,
                         "old": old_val, "new": new_val}

                # Record how far the override moved from what the detector
                # proposed. Per-edge bias (from the bounding box) answers
                # "which edge is systematically wrong"; per-corner keeps the
                # tilt, which a bounding box throws away.
                if key == "crop_quad" and new_val and kf.get("auto_crop_quad"):
                    auto_q = normalize_quad(kf["auto_crop_quad"])
                    new_q = normalize_quad(new_val)
                    if auto_q and new_q:
                        a_r, n_r = quad_to_rect(auto_q), quad_to_rect(new_q)
                        entry["auto"] = auto_q
                        entry["auto_source"] = kf.get("auto_crop_source")
                        entry["delta"] = {e: round(n_r[e] - a_r[e], 5)
                                           for e in ("x1", "y1", "x2", "y2")}
                        entry["delta_corners"] = [
                            [round(n[0] - o[0], 5), round(n[1] - o[1], 5)]
                            for n, o in zip(new_q, auto_q)
                        ]

                _append_review_log(paths, entry)

                if new_val is None:
                    kf.pop(key, None)
                else:
                    kf[key] = new_val
            break

    (paths.json / "keyframes.json").write_text(json.dumps(keyframes, indent=2))
    return jsonify({"ok": True})


@app.route("/api/keyframes/<project>/labels", methods=["GET"])
def get_labels(project):
    """Load saved review labels."""
    paths = ProjectPaths(OUTPUT_DIR / project)
    labels_path = paths.json / "review_labels.json"
    if labels_path.exists():
        return jsonify(json.loads(labels_path.read_text()))
    return jsonify({})


@app.route("/api/keyframes/<project>/labels", methods=["POST"])
def save_labels(project):
    """Save review labels (auto-saved on every change)."""
    paths = ProjectPaths(OUTPUT_DIR / project)
    paths.ensure("json")
    labels = request.json
    (paths.json / "review_labels.json").write_text(json.dumps(labels, indent=2))
    return jsonify({"ok": True})


# ── API: Crop preview ─────────────────────────────────────────

@app.route("/api/keyframes/<project>/restore-originals", methods=["POST"])
def restore_originals(project):
    """
    Re-extract every keyframe at full resolution from the source video.

    Cropping rewrites images in place, so this is the undo: it restores the
    raw frames and clears the "already cropped" markers, while keeping the
    crop boxes and gutters — those are stored in original-frame coordinates,
    so they stay valid and the next Apply crops exactly once.
    """
    import cv2
    paths = ProjectPaths(OUTPUT_DIR / project)
    kfs, kf_path = _load_keyframes(paths)
    meta_p = paths.json / "metadata.json"
    if not meta_p.exists():
        return jsonify({"error": "No metadata.json — cannot find the source video"}), 404
    video = Path(json.loads(meta_p.read_text()).get("video_path", ""))
    if not video.is_absolute():
        video = PROJECT_ROOT / video
    if not video.exists():
        return jsonify({"error": f"Source video not found: {video}"}), 404

    cap = cv2.VideoCapture(str(video))
    restored, failed = 0, []
    for kf in sorted(kfs, key=lambda k: k["frame_index"]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, kf["frame_index"])
        ok, frame = cap.read()
        if not ok or frame is None:
            failed.append(kf["frame_index"])
            continue
        cv2.imwrite(str(paths.images / kf["filename"]), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        for k in ("cropped", "applied_crop_method", "applied_x_range"):
            kf.pop(k, None)
        restored += 1
    cap.release()

    kf_path.write_text(json.dumps(kfs, indent=2))
    _append_review_log(paths, {"action": "restore_originals",
                               "restored": restored, "failed": failed})
    return jsonify({"restored": restored, "failed": failed})


@app.route("/api/crop-preview/<project>/<filename>")
def crop_preview(project, filename):
    """Return a cropped preview of an image."""
    import cv2
    import io
    paths = ProjectPaths(OUTPUT_DIR / project)
    img = cv2.imread(str(paths.images / filename))
    if img is None:
        return "Image not found", 404

    mode = request.args.get("mode", "double")
    side_trim = None
    if request.args.get("trim_left") and request.args.get("trim_right"):
        side_trim = {
            "left": float(request.args["trim_left"]),
            "right": float(request.args["trim_right"]),
        }

    manual = None
    if "quad" in request.args:
        try:
            manual = normalize_quad(json.loads(request.args["quad"]))
        except Exception:
            manual = None
    elif all(k in request.args for k in ("x1", "y1", "x2", "y2")):
        manual = {k: float(request.args[k]) for k in ("x1", "y1", "x2", "y2")}

    cropped, info = preview_crop(img, mode=mode, side_trim=side_trim,
                                  manual_rect=manual)

    _, buf = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


@app.route("/api/crop-rect/<project>/<filename>")
def crop_rect(project, filename):
    """
    Auto-detected crop QUAD (4 normalized corners) to seed the UI handles,
    plus the gutter in effect for this frame. Also records the algorithm's
    suggestion once, so a later manual override can be measured against it.
    """
    import cv2
    paths = ProjectPaths(OUTPUT_DIR / project)
    img = cv2.imread(str(paths.images / filename))
    if img is None:
        return jsonify({"error": "not found"}), 404
    mode = request.args.get("mode", "double")
    res = detect_crop_quad(img, mode=mode)

    kf_path = paths.json / "keyframes.json"
    payload = {"quad": res["quad"], "source": res["source"],
               "lines_used": res.get("lines_used")}

    if kf_path.exists():
        kfs = json.loads(kf_path.read_text())
        idx = next((i for i, k in enumerate(kfs)
                    if k.get("filename") == filename), None)
        if idx is not None:
            kf = kfs[idx]
            # inherited geometry, so the editor opens on what will be applied
            own_q = normalize_quad(kf.get("crop_quad"))
            inh_q, anchor = resolve_crop_quad(kfs, idx)
            payload["effective_quad"] = own_q or inh_q or res["quad"]
            payload["quad_src"] = ("manual" if own_q else
                                    "inherited" if inh_q else "auto")

            own_g = normalize_gutter(kf.get("gutter"))
            inh_g, g_anchor = resolve_gutter(kfs, idx)
            payload["gutter"] = own_g or inh_g or {"top": 0.5, "bot": 0.5}
            payload["gutter_src"] = ("manual" if own_g else
                                      "inherited" if inh_g else "auto")

            if "auto_crop_quad" not in kf:
                kf["auto_crop_quad"] = [[round(x, 5), round(y, 5)]
                                         for x, y in res["quad"]]
                kf["auto_crop_source"] = res["source"]
                kf_path.write_text(json.dumps(kfs, indent=2))
    return jsonify(payload)


# ── API: Video frame ──────────────────────────────────────────

@app.route("/api/video-frame")
def get_video_frame():
    """Get a single frame from a video file."""
    import cv2
    import io
    video_path = request.args["video"]
    frame_idx = int(request.args["frame"])

    frame = extract_frame(video_path, frame_idx)
    if frame is None:
        return "Frame not found", 404

    # Downscale for preview
    h, w = frame.shape[:2]
    max_w = int(request.args.get("max_width", 1200))
    if w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (max_w, int(h * scale)))

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


# ── API: Split & PDF ──────────────────────────────────────────

@app.route("/api/process/crop", methods=["POST"])
def run_crop():
    """Apply cropping to all keyframe images in-place."""
    import cv2
    data = request.json
    project = data["project"]
    mode = data.get("mode", "double")
    paths = ProjectPaths(OUTPUT_DIR / project)

    keyframes, _ = _load_keyframes(paths)

    task_id = f"crop_{project}"
    task_progress[task_id] = {"phase": "crop", "progress": 0, "status": "running"}

    def run():
        try:
            methods = {}

            def _inherit(idx, key):
                """Own value, else the most recent one set before it."""
                if keyframes[idx].get(key) is not None:
                    return keyframes[idx][key]
                for prev in reversed(keyframes[:idx]):
                    if prev.get(key) is not None:
                        return prev[key]
                return None

            skipped = 0
            for i, kf in enumerate(keyframes):
                # Cropping rewrites the image in place, so a second pass would
                # crop the crop — and the stored geometry is in *original*
                # frame coordinates. Any frame already cropped is left alone.
                if kf.get("cropped") or kf.get("applied_crop_method"):
                    skipped += 1
                    task_progress[task_id]["progress"] = round((i + 1) / len(keyframes) * 100)
                    continue
                img_path = paths.images / kf["filename"]
                if not img_path.exists():
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Manual rectangle wins, and propagates forward to later frames
                # (a book's geometry is stable; one good fit usually fits many).
                manual, _ = resolve_crop_quad(keyframes, i)
                if manual is None:
                    manual = normalize_quad(_inherit(i, "crop_quad"))

                # Covers get cropped too — no skipping
                side_trim = kf.get("crop_bounds")
                cropped, info = preview_crop(img, mode=mode, side_trim=side_trim,
                                              manual_rect=manual)
                cv2.imwrite(str(img_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # The gutter is stored relative to the crop box, so cropping
                # to that box cannot move it — no remapping needed. (The old
                # frame-relative fraction is what used to shift splits into
                # the text on a lopsided crop.)
                if mode == "double" and "x_range" in info:
                    kf["applied_x_range"] = [round(v, 5) for v in info["x_range"]]
                if info.get("method"):
                    kf["applied_crop_method"] = info["method"]
                kf["cropped"] = True

                m = info.get("method", "?")
                methods[m] = methods.get(m, 0) + 1
                task_progress[task_id]["progress"] = round((i + 1) / len(keyframes) * 100)

            # persist the remapped gutters so the split reads correct values
            (paths.json / "keyframes.json").write_text(json.dumps(keyframes, indent=2))

            task_progress[task_id] = {"phase": "crop", "progress": 100,
                                       "status": "done", "methods": methods,
                                       "skipped_already_cropped": skipped}
        except Exception as e:
            task_progress[task_id] = {"phase": "crop", "progress": 0,
                                      "status": "error", "error": str(e)}

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"task_id": task_id})


@app.route("/api/process/split", methods=["POST"])
def run_split():
    """Split cropped spreads into pages, as a background task with progress."""
    data = request.json
    project = data["project"]
    mode = data.get("mode", "double")
    gutter = data.get("gutter")
    paths = ProjectPaths(OUTPUT_DIR / project)
    paths.ensure("pages", "json")
    keyframes, _ = _load_keyframes(paths)

    task_id = f"split_{project}"
    task_progress[task_id] = {"phase": "split", "progress": 0, "status": "running"}

    def run():
        try:
            def on_progress(cur, total):
                task_progress[task_id]["progress"] = round(cur / total * 100)
            pages = split_all_pages(keyframes, paths.images, paths.pages,
                                    mode=mode, gutter=gutter,
                                    on_progress=on_progress,
                                    already_cropped=True)
            (paths.json / "pages.json").write_text(json.dumps(pages, indent=2))
            task_progress[task_id] = {"phase": "split", "progress": 100,
                                      "status": "done", "pages": len(pages)}
        except Exception as e:
            import traceback
            traceback.print_exc()
            task_progress[task_id] = {"phase": "split", "progress": 0,
                                      "status": "error", "error": str(e)}

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"task_id": task_id})


@app.route("/api/pages/<project>")
def get_pages(project):
    """Get page list for a project."""
    paths = ProjectPaths(OUTPUT_DIR / project)
    p_path = paths.json / "pages.json"
    if not p_path.exists():
        return jsonify([])
    return jsonify(json.loads(p_path.read_text()))


@app.route("/api/process/pdf", methods=["POST"])
def run_pdf():
    """Build PDF."""
    data = request.json
    project = data["project"]
    bw = data.get("bw", False)
    paths = ProjectPaths(OUTPUT_DIR / project)
    paths.ensure("pdf")

    pages = json.loads((paths.json / "pages.json").read_text())

    if bw:
        count = build_binarized_pdf(pages, paths.pages,
                                     paths.pdf / "book_bw.pdf")
    else:
        count = build_pdf(pages, paths.pages, paths.pdf / "book.pdf")

    return jsonify({"pages": count})



# ── Live capture ──────────────────────────────────────────────
# The capture window is a separate process: macOS only lets OpenCV windows
# live on a main thread, and Flask serves requests on worker threads. The
# app publishes json/live_status.json; the browser polls it through here.

import re
import subprocess
import sys as _sys

LIVE_DIR = Path(__file__).parent / "live"
LIVE_PROCS = {}

LIVE_DEFAULTS = {
    "camera": "auto", "resolution": "3840x2160", "fps": 30,
    "sound_set": "pluck", "guide": True,
    "settle_threshold": 2.0, "turn_threshold": 5.0, "settle_time": 0.4,
    "codec": "mp4v", "preview_height": 720, "analysis_height": 360,
    "smoothing_window": 15, "jpeg_quality": 95,
}


@app.route("/api/live/defaults")
def live_defaults():
    return jsonify(LIVE_DEFAULTS)


@app.route("/api/live/cameras")
def live_cameras():
    """Probe camera indices so the setup page can offer real choices —
    USB indices shuffle on reconnect, so a bare number is a guess."""
    import cv2, platform
    backend = {"Darwin": cv2.CAP_AVFOUNDATION,
               "Windows": cv2.CAP_DSHOW}.get(platform.system(), cv2.CAP_ANY)
    found = []
    for i in range(6):
        cap = cv2.VideoCapture(i, backend)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        ok, f = cap.read()
        if ok and f is not None:
            found.append({"index": i, "width": f.shape[1], "height": f.shape[0]})
        cap.release()
    return jsonify(found)


@app.route("/live-sounds/<path:fn>")
def live_sound(fn):
    # synthesized on first use, so there is nothing to install or ship
    if not (LIVE_DIR / "sounds" / fn).exists():
        _sys.path.insert(0, str(LIVE_DIR))
        from sounds import generate
        generate(LIVE_DIR / "sounds")
    return send_from_directory(LIVE_DIR / "sounds", fn)


@app.route("/api/live/start", methods=["POST"])
def live_start():
    d = {**LIVE_DEFAULTS, **(request.json or {})}
    name = str(d.get("name", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-]{0,63}", name):
        return jsonify({"error": "Name can use letters, numbers, - and _"}), 400

    proc = LIVE_PROCS.get(name)
    if proc and proc.poll() is None:
        return jsonify({"error": "A capture for this name is already running"}), 409

    out_dir = OUTPUT_DIR / name
    video = RECORDINGS_DIR / f"{name}.mp4"
    if ((out_dir / "json" / "keyframes.json").exists() or video.exists()) \
            and not d.get("overwrite"):
        return jsonify({"error": "exists",
                        "message": f"'{name}' already has a recording"}), 409

    try:
        w, h = (int(v) for v in str(d["resolution"]).lower().split("x"))
    except ValueError:
        return jsonify({"error": "Resolution must look like 3840x2160"}), 400

    (out_dir / "json").mkdir(parents=True, exist_ok=True)
    (out_dir / "json" / "live_status.json").unlink(missing_ok=True)
    RECORDINGS_DIR.mkdir(exist_ok=True)

    argv = [_sys.executable, str(LIVE_DIR / "live_capture.py"),
            str(out_dir), str(video),
            "--camera", str(d["camera"]),
            "--capture-width", str(w), "--capture-height", str(h),
            "--fps", str(float(d["fps"])),
            "--codec", str(d["codec"]),
            "--preview-height", str(int(d["preview_height"])),
            "--analysis-height", str(int(d["analysis_height"])),
            "--smoothing-window", str(int(d["smoothing_window"])),
            "--settle-threshold", str(float(d["settle_threshold"])),
            "--turn-threshold", str(float(d["turn_threshold"])),
            "--settle-time", str(float(d["settle_time"])),
            "--jpeg-quality", str(int(d["jpeg_quality"])),
            "--sound-set", str(d["sound_set"])]
    if not d.get("guide", True):
        argv.append("--no-guide")

    LIVE_PROCS[name] = subprocess.Popen(argv, cwd=str(PROJECT_ROOT))
    return jsonify({"project": name, "video": str(video)})


@app.route("/api/live/status/<name>")
def live_status(name):
    sp = OUTPUT_DIR / name / "json" / "live_status.json"
    st = {"running": True, "phase": "opening", "captures": 0}
    if sp.exists():
        try:
            st = json.loads(sp.read_text())
        except (json.JSONDecodeError, OSError):
            pass            # caught mid-write; the next poll will read it
    proc = LIVE_PROCS.get(name)
    if proc is not None and proc.poll() is not None and st.get("running"):
        # exited without reporting a clean finish
        st = {"running": False, "phase": "error", "captures": st.get("captures", 0),
              "error": f"Capture app exited unexpectedly (code {proc.returncode})"}
    st["video"] = str(RECORDINGS_DIR / f"{name}.mp4")
    return jsonify(st)

# ── API: Serve images ─────────────────────────────────────────

@app.route("/images/<project>/<filename>")
def serve_image(project, filename):
    return send_from_directory(OUTPUT_DIR / project / "images", filename)


@app.route("/pages/<project>/<filename>")
def serve_page(project, filename):
    return send_from_directory(OUTPUT_DIR / project / "pages", filename)


@app.route("/plots/<project>/<filename>")
def serve_plot(project, filename):
    return send_from_directory(OUTPUT_DIR / project / "plots", filename)


def _migrate_keyframes(keyframes: list) -> bool:
    """
    Bring older projects onto the quad + {pos,angle} gutter model.

    The UI used to store an axis-aligned `crop_rect` and a bare
    `gutter_pct`, while the CLI stored `crop_quad`. Both wrote the same
    file, so whichever saved last won and the other phase silently read a
    field that was never updated. One representation ends that.
    """
    changed = False
    for kf in keyframes:
        if kf.get("crop_quad") is None and kf.get("crop_rect") is not None:
            kf["crop_quad"] = quad_from_rect(kf.pop("crop_rect"))
            changed = True
        elif kf.get("crop_rect") is not None:
            kf.pop("crop_rect")          # quad already wins
            changed = True
        if kf.get("gutter") is None and kf.get("gutter_pct") is not None:
            v = float(kf.pop("gutter_pct"))
            kf["gutter"] = {"top": v, "bot": v}
            changed = True
        elif kf.get("gutter_pct") is not None:
            kf.pop("gutter_pct")
            changed = True
        g = kf.get("gutter")
        if g is not None and not (isinstance(g, dict) and "top" in g):
            kf["gutter"] = normalize_gutter(g)   # float or {pos,angle}
            changed = True
    return changed


def _load_keyframes(paths):
    """Read keyframes.json, migrating legacy geometry in place."""
    kf_path = paths.json / "keyframes.json"
    if not kf_path.exists():
        return [], kf_path
    kfs = json.loads(kf_path.read_text())
    if _migrate_keyframes(kfs):
        kf_path.write_text(json.dumps(kfs, indent=2))
    return kfs, kf_path


# ── Helpers ───────────────────────────────────────────────────

def _count_keyframes(paths):
    kf_path = paths.json / "keyframes.json"
    if kf_path.exists():
        return len(json.loads(kf_path.read_text()))
    return 0


def _count_pages(paths):
    p_path = paths.json / "pages.json"
    if p_path.exists():
        return len(json.loads(p_path.read_text()))
    return 0


def _append_review_log(paths, entry):
    from datetime import datetime
    entry["timestamp"] = datetime.now().isoformat()
    rl_path = paths.json / "review_log.json"
    if rl_path.exists():
        rl = json.loads(rl_path.read_text())
    else:
        rl = {"sessions": []}
    # Append to latest session or create new one
    if not rl["sessions"] or rl["sessions"][-1].get("closed"):
        rl["sessions"].append({"entries": [entry]})
    else:
        rl["sessions"][-1].setdefault("entries", []).append(entry)
    rl_path.write_text(json.dumps(rl, indent=2))


# ── Run ───────────────────────────────────────────────────────

if __name__ == "__main__":
    RECORDINGS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"ScanStudio UI: http://localhost:5050")
    print(f"Project root:  {PROJECT_ROOT}")
    print(f"Recordings:    {RECORDINGS_DIR}")
    print(f"Output:        {OUTPUT_DIR}")
    app.run(host="127.0.0.1", port=5050, debug=True)
