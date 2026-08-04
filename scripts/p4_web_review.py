#!/usr/bin/env python3
"""
Phase 4: Review Keyframes (browser)

The same review as p4_review_keyframes, with Chrome as the display: this
server owns the state and the pipeline semantics — actions, validation
pinning, the boundary watchdog, insert, save — and the page renders frames
and geometry and sends keys back. Exists for the same reason p0_web_capture
does: on ChromeOS the Tk window fights the container's compositor (panels
clip off small screens) and the Tk scrubber software-decodes 4K per keypress.
In the browser the scrubber is a native <video> element — hardware decode,
instant seeks — served straight from the recording with HTTP Range support.

State written on Save is byte-compatible with the Tk review: keyframes.json
(actions applied, crop_quad/gutter/rotation overrides, crop_quad_track from
the watchdog), review_log.json session records, deleted images removed. The
one thing this front end skips is the post-save matplotlib comparison plot.

Usage:
  python scripts/p4_web_review.py output/mybook recordings/mybook.mp4
  ... then open http://localhost:8412 (the shared ScanStudio port) in Chrome.

  ChromeOS: forward the port first — Settings > Linux > Port forwarding.

Keys (in the browser tab): identical to the Tk review — 1-6 actions,
A/D or ←/→ navigate, G geometry, E next flagged, I insert (video scrubber),
C center guide, Ctrl+S save.
"""

import argparse
import copy
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from corner_net import model_gutter_frac, model_quad_offsets
from p5_crop import DEFAULT_SAFETY_MARGIN, _spread_tilt, crop_double_page, \
    crop_to_quad, detect_page_quad
from utils import (
    ProjectPaths,
    TRACK_DEADBAND_FRAC,
    WATCHDOG_ALERT_FRAC,
    consensus_geometry,
    detect_gutter,
    log,
    measure_quad_offsets,
    page_mask_robust,
    quad_edge_bases,
    resolve_crop_anchor,
    resolve_crop_quad,
    resolve_gutter,
    resolve_rotation,
    rigid_shift,
)
from webui import (
    DEFAULT_PORT,
    WebUIServer,
    chromeos_note,
    send_file,
    serve_forever_in_thread,
)

ACTIONS = ("keep", "dup", "occlusion", "other", "cover", "doc_start")
DELETES = ("dup", "occlusion", "other")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Phase 4: Review keyframes in the browser")
    p.add_argument("output_dir", help="Base output directory (e.g. output/mybook)")
    p.add_argument("video_path", nargs="?", default=None,
                   help="Recording, for the insert scrubber (optional)")
    p.add_argument("--mode", default="double", choices=["single", "double"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    return p.parse_args(argv)


class ReviewSession:
    """One browser review session. Socket-free: ``send`` is any callable.

    The state model is the Tk app's, minus rendering: ``actions`` and
    ``validated`` keyed by list index, 1-press pins with their undo values,
    a per-frame geometry cache, and the watchdog worker publishing tracked
    boxes into ``watch``. The browser holds only view state (current index,
    editor-in-progress); every mutation lands here first and is answered
    with the authoritative result.
    """

    def __init__(self, args, paths, send):
        self.args = args
        self.paths = paths
        self.mode = args.mode
        self.send = send

        self.keyframes = json.loads((paths.json / "keyframes.json").read_text())
        meta_path = paths.json / "metadata.json"
        self.fps = 30.0
        if meta_path.exists():
            self.fps = json.loads(meta_path.read_text()).get("fps", 30.0)
        sm_path = paths.data / "smoothed_signal.npy"
        self.smoothed = (
            np.load(str(sm_path)) if sm_path.exists() else np.zeros(1)
        )
        self.video_path = self._find_video(args.video_path)

        # The consensus vote (and with it the first watchdog scan) runs in
        # the background AFTER the first state reaches the browser. On a
        # fresh session the vote is seconds of 4K mask work — and the
        # U^2-Net backstop's first-ever use downloads its model — so voting
        # on the connect path left the operator staring at an empty page.
        # Until the vote lands, frames without their own box render with no
        # overlay rather than paying the per-frame auto-crop fallback.
        self._consensus = None
        self._consensus_done = self.mode != "double"
        self._closed = False

        self.actions = {}
        self.validated = set()
        self._keep_pins = {}
        self.session_log = []
        self.session_start = datetime.now()
        self.pending_inserts = []
        self._geom_cache = {}
        self._crop_preview_cache = {}
        self.finished = False   # set by Finish; ends the server (finish-web)

        # Watchdog results, keyed by index; guarded by _lock because the
        # worker writes while the handler thread reads.
        self.watch = {}
        self._watch_gen = 0
        self._watch_threads = []
        self._lock = threading.Lock()

        for i, kf in enumerate(self.keyframes):
            if kf.get("validated"):
                self.validated.add(i)
                self.actions[i] = "keep"
            if kf.get("is_cover"):
                self.actions[i] = "cover"
            if kf.get("is_doc_start"):
                self.actions[i] = "doc_start"

    def _find_video(self, given):
        for cand in (given,):
            if cand and Path(cand).exists():
                return Path(cand)
        # Fall back to the conventional recording location for this project.
        name = Path(self.args.output_dir).name
        cand = Path("recordings") / f"{name}.mp4"
        return cand if cand.exists() else None

    # ── Outbound state ───────────────────────────────────────

    def _kf_summary(self):
        out = []
        for i, kf in enumerate(self.keyframes):
            out.append({
                "frame_index": kf["frame_index"],
                "filename": kf["filename"],
                "time_sec": kf.get("time_sec", 0),
                "motion": kf.get("motion_value", 0),
                "sharpness": kf.get("sharpness", 0),
                "source": kf.get("source", "?"),
                "own_quad": kf.get("crop_quad") is not None,
                "own_gutter": kf.get("gutter") is not None,
            })
        return out

    def _state_msg(self):
        with self._lock:
            watch = {
                str(i): {"flagged": w["flagged"], "score": w["score"],
                         "measured": w["measured"]}
                for i, w in self.watch.items()
            }
        return {
            "type": "state",
            "mode": self.mode,
            "fps": self.fps,
            "keyframes": self._kf_summary(),
            "actions": {str(i): a for i, a in self.actions.items()},
            "validated": sorted(self.validated),
            "inserts": len(self.pending_inserts),
            "watch": watch,
        }

    def hello(self):
        self.send(self._state_msg())
        # The smoothed motion signal feeds the scrubber's readout. One-time,
        # ~6 bytes/frame as JSON; trivial on localhost.
        self.send({
            "type": "signal",
            "video": self.video_path is not None,
            "smoothed": [round(float(v), 2) for v in self.smoothed],
        })
        if self.mode == "double":
            threading.Thread(target=self._init_geometry, daemon=True).start()

    def _init_geometry(self):
        """Vote the consensus box off the connect path, then start tracking.

        Cached in json/ after the first run, so only a fresh project pays
        the vote. The state push at the end makes the browser re-request the
        current frame, whose geometry now resolves against the consensus.
        """
        try:
            self._consensus = consensus_geometry(
                self.paths.images, self.keyframes,
                cache_path=self.paths.json / "consensus_geometry.json",
                log_fn=log,
            )
        finally:
            self._consensus_done = True
        self._geom_cache.clear()
        if self._closed:
            return
        self.send(self._state_msg())
        self._start_watchdog()

    def close(self):
        self._closed = True
        self._stop_watchdog()

    # ── Inbound ──────────────────────────────────────────────

    def handle(self, m):
        t = m.get("type")
        idx = m.get("idx")
        if t == "show":
            self._send_frame(idx)
        elif t == "action":
            self._set_action(idx, m.get("action"))
        elif t == "seed":
            self._send_seed(idx)
        elif t == "confirm":
            self._confirm(idx, m)
        elif t == "reset_geom":
            self._reset_geom(idx)
        elif t == "insert":
            self._insert(int(m.get("frame_index")))
        elif t == "save":
            self._save()
        elif t == "finish":
            # Save-and-exit: the review is reentrant, so finishing is just a
            # save that also lets a chained `make finish-web` move on to P5.
            self._save()
            self.finished = True
            self.send({"type": "bye"})

    # ── Frame payloads ───────────────────────────────────────

    def _send_frame(self, idx):
        if not (0 <= idx < len(self.keyframes)):
            return
        kf = self.keyframes[idx]
        is_cover = kf.get("is_cover") or self.actions.get(idx) == "cover"
        geom = None
        if self.mode == "double" and not is_cover:
            geom = self._frame_geometry(idx)
        elif self.mode == "single":
            manual = kf.get("crop_quad")
            quad = manual if manual else self._auto_crop_quad(idx)
            if quad:
                geom = {"box": quad,
                        "box_src": "manual" if manual else "auto",
                        "frac": None, "line": None}
        own_g = kf.get("gutter") is not None
        with self._lock:
            w = self.watch.get(idx)
            watch = None if w is None else {
                "flagged": w["flagged"], "score": w["score"],
                "measured": w["measured"],
            }
        self.send({
            "type": "frame", "idx": idx,
            "geom": geom,
            "gutter_own": own_g,
            "gutter_tracked": (not own_g)
            and resolve_gutter(self.keyframes, idx) is not None,
            "watch": watch,
            "validated": idx in self.validated,
        })

    def _frame_geometry(self, idx):
        """Port of the Tk preview geometry: what p5 will crop, p6 will cut."""
        if idx in self._geom_cache:
            return self._geom_cache[idx]
        kf = self.keyframes[idx]
        geom = None
        try:
            img = cv2.imread(str(self.paths.images / kf["filename"]))
            h, w = img.shape[:2]
            quad = resolve_crop_quad(self.keyframes, idx)
            box_src = "manual" if kf.get("crop_quad") else "inherited"
            if kf.get("crop_quad") is None:
                with self._lock:
                    ws = self.watch.get(idx)
                    tq = (ws.get("quad") if ws is not None
                          else kf.get("crop_quad_track"))
                if tq is not None:
                    quad, box_src = tq, "track"
            if quad is None and self._consensus:
                quad = self._consensus["quad"]
                box_src = "consensus"
            if quad is None and not self._consensus_done:
                # Vote still running: show the frame with no overlay now
                # rather than paying the slow per-frame auto crop; the
                # post-vote state push makes the browser re-request this
                # frame (and the vote clears this cached None).
                self._geom_cache[idx] = None
                return None
            frac = kf.get("gutter")
            cropped = None
            if quad is not None:
                box = [(float(x), float(y)) for x, y in quad]
                if frac is None:
                    quad_px = np.array(
                        [[x * w, y * h] for x, y in box], dtype=np.float32
                    )
                    cropped = crop_to_quad(img, quad_px, 0.0)
            else:
                rot = resolve_rotation(self.keyframes, idx)
                if rot is None:
                    rot = _spread_tilt(page_mask_robust(img))
                quad_px, cropped = self._auto_spread_quad(
                    img, kf.get("crop_margin", DEFAULT_SAFETY_MARGIN), rot
                )
                box = [(x / w, y / h) for x, y in quad_px]
                box_src = "auto"
            if frac is None:
                prior = resolve_gutter(self.keyframes, idx)
                if prior is None:
                    # No operator hint yet: the corner model's spine (when
                    # a model exists) hints instead, and the shadow scan
                    # refines it exactly as it would an operator's.
                    prior = model_gutter_frac(img, box)
                frac = detect_gutter(cropped, prior=prior) / max(
                    1, cropped.shape[1]
                )
            tl, tr, br, bl = box
            geom = {
                "box": [[float(x), float(y)] for x, y in box],
                "box_src": box_src,
                "frac": float(frac),
                "line": [
                    [tl[0] + frac * (tr[0] - tl[0]),
                     tl[1] + frac * (tr[1] - tl[1])],
                    [bl[0] + frac * (br[0] - bl[0]),
                     bl[1] + frac * (br[1] - bl[1])],
                ],
            }
        except Exception:
            geom = None
        self._geom_cache[idx] = geom
        return geom

    @staticmethod
    def _auto_spread_quad(img, margin, rot):
        """The auto crop as a box on the raw frame (Tk _auto_spread_quad)."""
        h, w = img.shape[:2]
        cropped, method, (x0, y0, cw_, ch_) = crop_double_page(img, margin, rot)
        corners = np.array(
            [[x0, y0], [x0 + cw_, y0], [x0 + cw_, y0 + ch_], [x0, y0 + ch_]],
            dtype=np.float64,
        )
        if rot is None:
            rot = 0.0
        if abs(rot) > 0.2:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), -rot, 1.0)
            ones = np.hstack([corners, np.ones((4, 1))])
            corners = ones @ M.T
        return corners, cropped

    def _auto_crop_quad(self, idx):
        """Single mode: GrabCut auto box for the preview (Tk port, cached)."""
        if idx in self._crop_preview_cache:
            return self._crop_preview_cache[idx]
        kf = self.keyframes[idx]
        quad = None
        try:
            img = cv2.imread(str(self.paths.images / kf["filename"]))
            h, w = img.shape[:2]
            q = detect_page_quad(img)
            if q is None:
                q = np.array(
                    [[0.1 * w, 0.1 * h], [0.9 * w, 0.1 * h],
                     [0.9 * w, 0.9 * h], [0.1 * w, 0.9 * h]],
                    dtype=np.float32,
                )
            quad = [[float(x) / w, float(y) / h] for x, y in q]
        except Exception:
            quad = None
        self._crop_preview_cache[idx] = quad
        return quad

    # ── Actions & validation (Tk _set_action/_validate ports) ──

    def _set_action(self, idx, action):
        if action not in ACTIONS or not (0 <= idx < len(self.keyframes)):
            return
        if self.actions.get(idx) == action:
            del self.actions[idx]
            if action == "keep":
                self._unvalidate(idx)
        else:
            self.actions[idx] = action
            if action == "keep":
                self._validate(idx)
            elif idx in self.validated:
                self._unvalidate(idx)
        self.session_log.append({
            "time": datetime.now().isoformat(), "type": "action",
            "action": action, "frame": self.keyframes[idx]["frame_index"],
        })
        self.send(self._state_msg())
        self._send_frame(idx)

    def _validate(self, idx):
        self.validated.add(idx)
        self.keyframes[idx]["validated"] = True
        self._pin_geometry(idx)

    def _unvalidate(self, idx):
        self.validated.discard(idx)
        self.keyframes[idx].pop("validated", None)
        pins = self._keep_pins.pop(idx, None)
        if pins:
            kf = self.keyframes[idx]
            for field, prev in pins.items():
                if prev is None:
                    kf.pop(field, None)
                else:
                    kf[field] = prev
            self._geom_cache.pop(idx, None)

    def _pin_geometry(self, idx):
        kf = self.keyframes[idx]
        pins = {}
        if self.mode == "double":
            if kf.get("is_cover") or self.actions.get(idx) == "cover":
                return
            geom = self._frame_geometry(idx)
            if geom is None:
                return
            if kf.get("crop_quad") is None:
                pins["crop_quad"] = None
                kf["crop_quad"] = [
                    [round(x, 5), round(y, 5)] for x, y in geom["box"]
                ]
                if self._geom_cache.get(idx):
                    self._geom_cache[idx]["box_src"] = "manual"
            if kf.get("gutter") is None:
                pins["gutter"] = None
                kf["gutter"] = round(geom["frac"], 4)
        else:
            if kf.get("crop_quad") is None:
                quad = self._auto_crop_quad(idx)
                if quad:
                    pins["crop_quad"] = None
                    kf["crop_quad"] = [
                        [round(x, 5), round(y, 5)] for x, y in quad
                    ]
        if pins:
            self._keep_pins.setdefault(idx, {}).update(pins)

    # ── Geometry editor round trips ──────────────────────────

    def _send_seed(self, idx):
        """Editor seed: box, gutter, sources — the Tk _seed_split_editor."""
        kf = self.keyframes[idx]
        img = cv2.imread(str(self.paths.images / kf["filename"]))
        if img is None:
            self.send({"type": "notice", "text": "Could not read this frame."})
            return
        h, w = img.shape[:2]
        if self.mode == "single":
            stored = kf.get("crop_quad")
            if stored:
                quad, src = stored, "manual"
            else:
                q = detect_page_quad(img)
                if q is None:
                    quad, src = [[0.1, 0.1], [0.9, 0.1],
                                 [0.9, 0.9], [0.1, 0.9]], "auto*"
                else:
                    quad = [[float(x) / w, float(y) / h] for x, y in q]
                    src = "auto"
            self.actions[idx] = "keep"
            self.send({"type": "seed", "idx": idx, "W": w, "H": h,
                       "box": quad, "box_src": src,
                       "gutter": None, "auto_gutter": None,
                       "gutter_src": None})
            return
        if kf.get("is_cover") or self.actions.get(idx) == "cover":
            self.send({"type": "notice",
                       "text": "Covers are not split into pages."})
            return
        quad = resolve_crop_quad(self.keyframes, idx)
        src = "manual" if kf.get("crop_quad") else "inherited"
        if kf.get("crop_quad") is None:
            with self._lock:
                ws = self.watch.get(idx)
                tq = (ws.get("quad") if ws is not None
                      else kf.get("crop_quad_track"))
            if tq is not None:
                quad, src = tq, "track"
        if quad is None and self._consensus:
            quad, src = self._consensus["quad"], "consensus"
        if quad is None and not self._consensus_done:
            self.send({"type": "notice",
                       "text": "still measuring session geometry — "
                               "try G again in a moment"})
            return
        if quad is not None:
            quad_px = np.array([[x * w, y * h] for x, y in quad],
                               dtype=np.float32)
            cropped = crop_to_quad(img, quad_px, 0.0)
            box = [[float(x), float(y)] for x, y in quad]
        else:
            rot = resolve_rotation(self.keyframes, idx)
            if rot is None:
                rot = _spread_tilt(page_mask_robust(img))
            quad_px, cropped = self._auto_spread_quad(
                img, kf.get("crop_margin", DEFAULT_SAFETY_MARGIN), rot
            )
            box = [[float(x) / w, float(y) / h] for x, y in quad_px]
            src = "auto"
        prior = resolve_gutter(self.keyframes, idx - 1) if idx > 0 else None
        if prior is None:
            prior = model_gutter_frac(img, box)
        auto_g = detect_gutter(cropped, prior=prior) / max(1, cropped.shape[1])
        own_g = kf.get("gutter")
        # Entering the editor adopts Keep, exactly like the Tk app: tuning a
        # spread's geometry implies it's a page you're keeping.
        self.actions[idx] = "keep"
        self.send({
            "type": "seed", "idx": idx, "W": w, "H": h,
            "box": box, "box_src": src,
            "gutter": own_g if own_g is not None else auto_g,
            "auto_gutter": auto_g,
            "gutter_src": ("manual" if own_g is not None
                           else "tracked" if prior is not None else "auto"),
        })

    def _confirm(self, idx, m):
        """Editor confirm: the Tk _split_confirm/_crop_confirm + validate."""
        kf = self.keyframes[idx]
        if self.mode == "double":
            kf["gutter"] = round(float(m["gutter"]), 4)
            kf.pop("gutter_raw", None)
            if m.get("box_dirty"):
                kf["crop_quad"] = [
                    [round(float(x), 5), round(float(y), 5)]
                    for x, y in m["box"]
                ]
                kf["rotation_deg"] = round(float(m.get("rotation_deg", 0)), 3)
                kf.pop("crop_margin", None)
            self._geom_cache.clear()
            if m.get("box_dirty"):
                self._start_watchdog(idx)
            self.session_log.append({
                "time": datetime.now().isoformat(), "type": "split",
                "frame": kf["frame_index"], "gutter": kf["gutter"],
                "rotation_deg": kf.get("rotation_deg"),
                "crop_quad": kf.get("crop_quad"),
            })
        else:
            kf["crop_quad"] = [
                [round(float(x), 5), round(float(y), 5)] for x, y in m["box"]
            ]
            self._crop_preview_cache.pop(idx, None)
            self.session_log.append({
                "time": datetime.now().isoformat(), "type": "crop",
                "frame": kf["frame_index"], "crop_quad": kf["crop_quad"],
            })
        self.actions[idx] = "keep"
        self._validate(idx)
        self.send(self._state_msg())
        self._send_frame(idx)

    def _reset_geom(self, idx):
        """Editor reset: drop every override and re-derive (Tk _split_reset)."""
        kf = self.keyframes[idx]
        self._unvalidate(idx)
        for field in ("gutter", "gutter_raw", "rotation_deg", "crop_margin",
                      "crop_quad", "crop_quad_track"):
            kf.pop(field, None)
        self._geom_cache.clear()
        self._crop_preview_cache.pop(idx, None)
        if self.mode == "double":
            self._start_watchdog(idx)
        self.send(self._state_msg())
        self._send_seed(idx)

    # ── Insert (Tk _on_insert port; the frame comes from the video) ──

    def _insert(self, frame_idx):
        if self.video_path is None:
            self.send({"type": "notice", "text": "No recording available."})
            return
        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            self.send({"type": "notice",
                       "text": f"Could not read frame {frame_idx}."})
            return
        filename = f"frame{frame_idx:06d}.jpg"
        cv2.imwrite(str(self.paths.images / filename), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        motion = float(self.smoothed[min(frame_idx, len(self.smoothed) - 1)])
        new_kf = {
            "frame_index": frame_idx,
            "time_sec": round(frame_idx / self.fps, 2),
            "motion_value": round(motion, 4),
            "sharpness": 0.0,
            "filename": filename,
            "source": "manual_insert",
        }
        insert_at = 0
        for i, kf in enumerate(self.keyframes):
            if kf["frame_index"] > frame_idx:
                insert_at = i
                break
            insert_at = i + 1
        self.keyframes.insert(insert_at, new_kf)
        self.actions = {
            (k + 1 if k >= insert_at else k): v for k, v in self.actions.items()
        }
        self.validated = {
            i + 1 if i >= insert_at else i for i in self.validated
        }
        self._keep_pins = {
            (k + 1 if k >= insert_at else k): v
            for k, v in self._keep_pins.items()
        }
        self._geom_cache.clear()
        self._crop_preview_cache.clear()
        if self.mode == "double":
            self._start_watchdog()
        self.pending_inserts.append(new_kf)
        self.session_log.append({
            "time": datetime.now().isoformat(), "type": "insert",
            "frame": frame_idx,
        })
        log(f"Inserted frame {frame_idx}")
        self.send(self._state_msg())
        self.send({"type": "inserted", "idx": insert_at,
                   "frame_index": frame_idx, "motion": motion})

    # ── Save (Tk _save port, minus the matplotlib plot) ──────

    def _save(self):
        for i, kf in enumerate(self.keyframes):
            kf.pop("crop_quad_track", None)
            with self._lock:
                ws = self.watch.get(i)
                tq = ws.get("quad") if ws else None
            if tq is not None and kf.get("crop_quad") is None:
                kf["crop_quad_track"] = [
                    [round(float(x), 5), round(float(y), 5)] for x, y in tq
                ]

        del_indices = sorted(
            [i for i, a in self.actions.items() if a in DELETES], reverse=True
        )
        deleted_info = []
        for i in del_indices:
            kf = self.keyframes[i]
            deleted_info.append({
                "frame_index": kf["frame_index"],
                "filename": kf["filename"],
                "reason": self.actions[i],
            })
            img_path = self.paths.images / kf["filename"]
            if img_path.exists():
                img_path.unlink()

        cover_frames = {
            self.keyframes[i]["frame_index"]
            for i, a in self.actions.items() if a == "cover"
        }
        docstart_frames = {
            self.keyframes[i]["frame_index"]
            for i, a in self.actions.items() if a == "doc_start"
        }
        for i in del_indices:
            self.keyframes.pop(i)
        for kf in self.keyframes:
            fi = kf["frame_index"]
            kf["is_cover"] = fi in cover_frames
            kf["is_doc_start"] = fi in docstart_frames
            if not kf["is_cover"]:
                kf.pop("is_cover")
            if not kf["is_doc_start"]:
                kf.pop("is_doc_start")

        (self.paths.json / "keyframes.json").write_text(
            json.dumps(self.keyframes, indent=2)
        )

        now = datetime.now()
        session = {
            "timestamp": now.isoformat(),
            "started": self.session_start.isoformat(),
            "deletions": deleted_info,
            "insertions": [
                {"frame_index": ins["frame_index"]}
                for ins in self.pending_inserts
            ],
            "keyframe_count_after": len(self.keyframes),
            "events": self.session_log,
            "frontend": "web",
        }
        rl_path = self.paths.json / "review_log.json"
        rl = (json.loads(rl_path.read_text()) if rl_path.exists()
              else {"sessions": []})
        rl["sessions"].append(session)
        rl_path.write_text(json.dumps(rl, indent=2))

        self.session_log = []
        self.session_start = now
        self.pending_inserts = []
        self.actions = {}
        self._geom_cache.clear()
        self._crop_preview_cache.clear()
        self.validated = set()
        self._keep_pins = {}
        for i, kf in enumerate(self.keyframes):
            if kf.get("validated"):
                self.validated.add(i)
                self.actions[i] = "keep"
            if kf.get("is_cover"):
                self.actions[i] = "cover"
            if kf.get("is_doc_start"):
                self.actions[i] = "doc_start"
        # Deletions shifted every index the watchdog keyed on; rescan.
        with self._lock:
            self.watch.clear()
        if self.mode == "double":
            self._start_watchdog()
        log(f"Saved: {len(self.keyframes)} keyframes, "
            f"{len(deleted_info)} deleted")
        self.send({"type": "saved", "deleted": len(deleted_info),
                   "kept": len(self.keyframes)})
        self.send(self._state_msg())

    # ── Watchdog (Tk _watchdog_worker port; results push over ws) ──

    def _start_watchdog(self, from_idx=0):
        if self.mode != "double":
            return
        self._watch_gen += 1
        with self._lock:
            for i in [i for i in self.watch if i >= from_idx]:
                del self.watch[i]
        self._watch_threads = [t for t in self._watch_threads if t.is_alive()]
        t = threading.Thread(
            target=self._watchdog_worker,
            args=(self._watch_gen, copy.deepcopy(self.keyframes), from_idx),
            daemon=True,
        )
        self._watch_threads.append(t)
        t.start()

    def _stop_watchdog(self, timeout=3.0):
        self._watch_gen += 1
        deadline = time.monotonic() + timeout
        for t in self._watch_threads:
            t.join(max(0.0, deadline - time.monotonic()))
        self._watch_threads = []

    def _watchdog_worker(self, gen, kfs, from_idx):
        anchor_key, anchor, window = None, None, []
        quad0 = bases0 = axes = anchor_s = None
        anchor_model = False
        shift_uv = [0.0, 0.0]
        for idx, kf in enumerate(kfs):
            if self._watch_gen != gen:
                return
            if kf.get("is_cover"):
                continue
            quad_frac, a_idx = resolve_crop_anchor(kfs, idx)
            if quad_frac is None:
                if not self._consensus:
                    continue
                quad_frac, a_idx = self._consensus["quad"], "consensus"
            if a_idx != anchor_key:
                anchor_key, anchor, window = a_idx, None, []
                quad0 = None
                anchor_model = False
                shift_uv = [0.0, 0.0]
            if idx < from_idx:
                continue
            if anchor is None:
                if a_idx == "consensus":
                    anchor = (self._consensus["edge_ref"],
                              self._consensus["edge_rel"])
                else:
                    aimg = cv2.imread(
                        str(self.paths.images / kfs[a_idx]["filename"])
                    )
                    if aimg is None:
                        anchor = "unavailable"
                    else:
                        ah, aw = aimg.shape[:2]
                        aq = np.array(quad_frac, float) * [aw, ah]
                        # Corner model, when present, replaces the mask as
                        # the boundary the tracker measures — but anchor and
                        # tracked frames must use the *same* source, so its
                        # per-session bias cancels in the difference.
                        anchor = model_quad_offsets(aimg, aq)
                        anchor_model = anchor is not None
                        if anchor is None:
                            anchor = measure_quad_offsets(aimg, aq)
            if anchor == "unavailable":
                continue
            if a_idx == idx:
                self._publish_watch(gen, idx, 0.0, True, False, None)
                continue
            img = cv2.imread(str(self.paths.images / kf["filename"]))
            if img is None:
                continue
            h, w = img.shape[:2]
            if quad0 is None:
                quad0 = np.array(quad_frac, float) * [w, h]
                bases0, axes = quad_edge_bases(quad0)
                anchor_s = [b + o for b, o in zip(bases0, anchor[0])]
            tq = quad0 + shift_uv[0] * axes[0] + shift_uv[1] * axes[1]
            if anchor_model:
                # A rare failed inference is an unmeasured frame (flagged),
                # never a silent fall-through to the mask: mixing sources
                # would re-introduce the bias the anchor subtraction cancels.
                m = model_quad_offsets(img, tq)
                off, rel = m if m is not None else ([0.0] * 4, [False] * 4)
            else:
                off, rel = measure_quad_offsets(img, tq)
            bases, _ = quad_edge_bases(tq)
            window.append(([b + o for b, o in zip(bases, off)], rel))
            if len(window) > 3:
                window.pop(0)
            shift, resid, measured = rigid_shift(
                anchor_s, anchor[1],
                [s for s, _ in window], [r for _, r in window],
            )
            for ax in (0, 1):
                if shift[ax] is not None and (
                    abs(shift[ax] - shift_uv[ax]) > TRACK_DEADBAND_FRAC * w
                ):
                    shift_uv[ax] = shift[ax]
            flagged = (not measured) or resid > WATCHDOG_ALERT_FRAC * w
            if shift_uv[0] or shift_uv[1]:
                tq = quad0 + shift_uv[0] * axes[0] + shift_uv[1] * axes[1]
                tq_frac = [[float(x) / w, float(y) / h] for x, y in tq]
            else:
                tq_frac = None
            self._publish_watch(gen, idx, float(resid), measured, flagged,
                                tq_frac)
        if self._watch_gen == gen:
            self.send({"type": "watch_done"})

    def _publish_watch(self, gen, idx, score, measured, flagged, quad):
        if self._watch_gen != gen:
            return
        with self._lock:
            prev = self.watch.get(idx)
            self.watch[idx] = {"score": score, "measured": measured,
                               "flagged": flagged, "quad": quad}
            if prev is None or prev.get("quad") != quad:
                self._geom_cache.pop(idx, None)
        self.send({"type": "watch", "idx": idx, "score": round(score, 1),
                   "measured": measured, "flagged": flagged,
                   "moved": quad is not None})


def build_server(args):
    paths = ProjectPaths(args.output_dir)
    kf_path = paths.json / "keyframes.json"
    if not kf_path.exists():
        raise FileNotFoundError(f"{kf_path} not found. Run Phase 3 first.")
    html_path = Path(__file__).resolve().parent.parent / "web" / "review.html"
    if not html_path.exists():
        raise FileNotFoundError(f"review page not found: {html_path}")

    session_box = {}

    def make_session(send):
        s = ReviewSession(args, paths, send)
        session_box["s"] = s
        return s

    def route(handler, path, head_only):
        if path.startswith("/img/"):
            name = Path(path[5:]).name          # no traversal
            send_file(handler, paths.images / name, head_only)
            return True
        if path == "/video":
            s = session_box.get("s")
            video = s.video_path if s else None
            if video is None:
                handler.send_error(404)
            else:
                send_file(handler, video, head_only)
            return True
        return False

    return WebUIServer((args.host, args.port), html_path, make_session, route)


def main():
    args = parse_args()
    log("=" * 60)
    log(f"PHASE 4: Review Keyframes (browser, {args.mode} mode)")
    log("=" * 60)
    try:
        server = build_server(args)
    except (FileNotFoundError, OSError) as e:
        log(f"ERROR: {e}")
        import errno
        if isinstance(e, OSError) and e.errno == errno.EADDRINUSE:
            log("  The shared ScanStudio port is taken — another web app "
                "(capture or a review) is still running. Finish or Ctrl+C "
                "it, or pass --port.")
        sys.exit(1)
    port = server.server_address[1]
    log("")
    chromeos_note(port)
    log("")
    log("  Finish in the browser (Q) when done — or Ctrl+C here.")
    serve_forever_in_thread(server)
    try:
        server.done.wait()
        log("Review finished in the browser.")
    except KeyboardInterrupt:
        log("\nDone.")
    server.shutdown()


if __name__ == "__main__":
    main()
