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

from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    send_file,
    send_from_directory,
)

from engine import ProjectPaths, derive_output_dir
from engine.motion import compute_motion_signal, smooth_signal, plot_motion_signal
from engine.peaks import detect_peaks, rescue_missed_turns, build_spreads, plot_peaks
from engine.keyframes import select_all_keyframes, extract_frame
from engine.crop import preview_crop, crop_otsu, crop_grabcut, apply_side_trim
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
                projects.append(
                    {
                        "name": d.name,
                        "path": str(d),
                        "has_motion": (paths.data / "motion_signal.npy").exists(),
                        "has_peaks": (paths.data / "peaks.npy").exists(),
                        "has_keyframes": (paths.json / "keyframes.json").exists(),
                        "has_pages": (paths.json / "pages.json").exists(),
                        "has_pdf": (paths.pdf / "book.pdf").exists(),
                        "keyframe_count": _count_keyframes(paths),
                        "page_count": _count_pages(paths),
                    }
                )
    return jsonify(projects)


@app.route("/api/recordings")
def list_recordings():
    """List available video files."""
    recordings = []
    if RECORDINGS_DIR.exists():
        for f in sorted(RECORDINGS_DIR.iterdir()):
            if f.suffix.lower() in (".mp4", ".mov", ".avi"):
                stat = f.stat()
                recordings.append(
                    {
                        "name": f.name,
                        "path": str(f),
                        "size_mb": round(stat.st_size / 1024 / 1024, 1),
                    }
                )
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
            plot_motion_signal(
                diffs, smoothed, metadata["fps"], paths.plots / "motion_plot.png"
            )

            task_progress[task_id] = {
                "phase": "motion",
                "progress": 100,
                "status": "done",
            }
        except Exception as e:
            task_progress[task_id] = {
                "phase": "motion",
                "progress": 0,
                "status": "error",
                "error": str(e),
            }

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

    peaks_arr = detect_peaks(
        smoothed,
        fps,
        height=data.get("peak_height", 5.0),
        distance_sec=data.get("min_distance", 1.5),
        prominence=data.get("prominence", 3.0),
    )
    peaks_arr = rescue_missed_turns(smoothed, fps, peaks_arr)
    spreads = build_spreads(peaks_arr, len(smoothed), fps)

    np.save(str(paths.data / "peaks.npy"), peaks_arr)
    (paths.json / "spreads.json").write_text(json.dumps(spreads, indent=2))
    plot_peaks(smoothed, peaks_arr, spreads, fps, paths.plots / "peaks_plot.png")

    return jsonify(
        {
            "peaks": len(peaks_arr),
            "spreads": len(spreads),
            "median_duration": round(
                float(np.median([s["duration_sec"] for s in spreads])), 2
            ),
        }
    )


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
                video_path,
                spreads,
                smoothed,
                metadata["fps"],
                paths.images,
                on_progress=on_progress,
            )
            (paths.json / "keyframes.json").write_text(json.dumps(kf_data, indent=2))
            task_progress[task_id] = {
                "phase": "keyframes",
                "progress": 100,
                "status": "done",
                "count": len(kf_data),
            }
        except Exception as e:
            task_progress[task_id] = {
                "phase": "keyframes",
                "progress": 0,
                "status": "error",
                "error": str(e),
            }

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
    kf_path = paths.json / "keyframes.json"
    if not kf_path.exists():
        return jsonify([])
    return jsonify(json.loads(kf_path.read_text()))


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
    _append_review_log(
        paths,
        {
            "action": "delete",
            "frame_index": frame_index,
            "reason": data.get("reason", "manual"),
        },
    )

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
            for key in ("is_cover", "is_doc_start", "gutter_pct", "crop_bounds"):
                if key in data:
                    if data[key] is None:
                        kf.pop(key, None)
                    else:
                        kf[key] = data[key]
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

    cropped, info = preview_crop(img, mode=mode, side_trim=side_trim)

    _, buf = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


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


@app.route("/api/process/split", methods=["POST"])
def run_split():
    """Run page splitting."""
    data = request.json
    project = data["project"]
    mode = data.get("mode", "double")
    gutter = data.get("gutter_pct", 0.5)
    paths = ProjectPaths(OUTPUT_DIR / project)
    paths.ensure("pages", "json")

    keyframes = json.loads((paths.json / "keyframes.json").read_text())
    pages = split_all_pages(
        keyframes, paths.images, paths.pages, mode=mode, gutter_pct=gutter
    )
    (paths.json / "pages.json").write_text(json.dumps(pages, indent=2))

    return jsonify({"pages": len(pages)})


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
        count = build_binarized_pdf(pages, paths.pages, paths.pdf / "book_bw.pdf")
    else:
        count = build_pdf(pages, paths.pages, paths.pdf / "book.pdf")

    return jsonify({"pages": count})


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
