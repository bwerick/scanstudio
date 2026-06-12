#!/usr/bin/env python3
"""
Phase 4: Review Keyframes (Reentrant)

Interactive GUI for reviewing keyframes. Modifies images/ in-place and
appends to the review log. Can be run multiple times.

Usage:
  python scripts/p4_review_keyframes.py output/mybook recordings/mybook.mp4

Keys:
  →/D  Next    ←/A  Prev
  1    Keep    2    Delete: Duplicate    3    Delete: Occlusion
  4    Delete: Other    5    Cover    6    Doc Start
  I    Insert frame (video scrubber)
  C    Toggle center guide
  G    Adjust split/rotation (←/→ gutter, [ / ] rotate, Enter save, Esc cancel, ⌫ reset)
  ⌘S   Save

Cropping and deskew are automatic (Phase 5). G previews that result and lets
you correct the gutter (split) and rotation per spread. A rotation correction
propagates forward to later spreads until the next correction.
"""

import argparse, json, sys, time, shutil, os
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from utils import log, ProjectPaths, ensure_dir, detect_gutter, page_mask, resolve_rotation
from p5_crop import crop_double_page, _spread_tilt

ACTIONS = {
    "keep": {"color": "#22c55e", "label": "Keep", "key": "1"},
    "dup": {"color": "#f59e0b", "label": "Delete: Duplicate", "key": "2"},
    "occlusion": {"color": "#ec4899", "label": "Delete: Occlusion", "key": "3"},
    "other": {"color": "#94a3b8", "label": "Delete: Other", "key": "4"},
    "cover": {"color": "#3b82f6", "label": "Cover (no split)", "key": "5"},
    "doc_start": {"color": "#a855f7", "label": "Doc Start", "key": "6"},
}


class VideoScrubber:
    def __init__(self, parent, video_path, start_frame, fps, smoothed, on_grab):
        self.fps, self.smoothed, self.on_grab = fps, smoothed, on_grab
        self.current_frame = start_frame
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.win = tk.Toplevel(parent)
        self.win.title("Video Scrubber — Find Missing Frame")
        self.win.configure(bg="#0a0a0a")
        self.win.geometry("1000x700")
        self.win.transient(parent)
        self.win.grab_set()

        top = tk.Frame(self.win, bg="#111")
        top.pack(fill="x")
        self.lbl_info = tk.Label(
            top, text="", font=("Menlo", 12), bg="#111", fg="#e2e8f0"
        )
        self.lbl_info.pack(side="left", padx=12, pady=6)
        self.lbl_motion = tk.Label(
            top, text="", font=("Menlo", 11), bg="#111", fg="#64748b"
        )
        self.lbl_motion.pack(side="right", padx=12, pady=6)

        self.canvas = tk.Canvas(self.win, bg="#111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=4)

        bot = tk.Frame(self.win, bg="#0a0a0a")
        bot.pack(fill="x", pady=8)
        tk.Label(
            bot,
            text="←/→ ±1   ↑/↓ ±5   Shift+←/→ ±30   Enter=Grab   Esc=Cancel",
            font=("Menlo", 10),
            bg="#0a0a0a",
            fg="#475569",
        ).pack()
        bf = tk.Frame(bot, bg="#0a0a0a")
        bf.pack(pady=4)
        grab = tk.Label(
            bf,
            text="  ✓ GRAB  ",
            font=("Menlo", 11, "bold"),
            bg="#22c55e",
            fg="white",
            relief="flat",
            padx=16,
            pady=4,
            cursor="hand2",
        )
        grab.bind("<Button-1>", lambda e: self._grab())
        grab.pack(side="left", padx=16)
        cancel = tk.Label(
            bf,
            text="Cancel",
            font=("Menlo", 10),
            bg="#1e293b",
            fg="#94a3b8",
            relief="flat",
            padx=8,
            pady=4,
            cursor="hand2",
        )
        cancel.bind("<Button-1>", lambda e: self._cancel())
        cancel.pack(side="left", padx=2)

        for key, delta in [
            ("<Right>", 1),
            ("<Left>", -1),
            ("<Up>", 5),
            ("<Down>", -5),
            ("<Shift-Right>", 30),
            ("<Shift-Left>", -30),
        ]:
            self.win.bind(key, lambda e, d=delta: self._step(d))
        self.win.bind("<Return>", lambda e: self._grab())
        self.win.bind("<Escape>", lambda e: self._cancel())
        self.photo = None
        self._show_frame()

    def _step(self, delta):
        self.current_frame = max(
            0, min(self.current_frame + delta, self.total_frames - 1)
        )
        self._show_frame()

    def _show_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return
        self.lbl_info.config(
            text=f"Frame {self.current_frame}  |  {self.current_frame/self.fps:.2f}s"
        )
        motion = self.smoothed[min(self.current_frame, len(self.smoothed) - 1)]
        mc = "#22c55e" if motion < 2.0 else "#f59e0b" if motion < 4.0 else "#ef4444"
        self.lbl_motion.config(text=f"Motion: {motion:.2f}", fg=mc)
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10:
            cw, ch = 960, 540
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        iw, ih = img.size
        scale = min(cw / iw, ch / ih, 1.0)
        img = img.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.photo, anchor="center")

    def _grab(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            self.on_grab(self.current_frame, frame)
        self.cap.release()
        self.win.destroy()

    def _cancel(self):
        self.cap.release()
        self.win.destroy()


class ReviewApp:
    def __init__(self, root, output_dir, video_path):
        self.root = root
        self.root.title("Phase 4: Review Keyframes")
        self.root.configure(bg="#0a0a0a")
        self.root.geometry("1200x800")

        self.paths = ProjectPaths(output_dir)
        self.video_path = video_path

        self.keyframes = json.loads((self.paths.json / "keyframes.json").read_text())
        self.smoothed = np.load(str(self.paths.data / "smoothed_signal.npy"))
        meta = json.loads((self.paths.json / "metadata.json").read_text())
        self.fps = meta["fps"]

        # State
        self.current_idx = 0
        self.actions = {}  # index_in_list -> action key
        self.pending_deletes = []  # indices to delete on save
        self.pending_inserts = []  # {frame_index, frame_bgr} to add on save
        self.photo = None
        self.show_center_guide = False
        self.session_log = []

        # Split / deskew adjust (overrides stored per-keyframe in keyframes.json).
        # Cropping/deskew themselves are automatic in Phase 5; G only tunes the
        # gutter and rotation of that auto result.
        self.split_mode = False
        self._split_frac = 0.5  # gutter as fraction of cropped spread width
        self._split_rot = None  # deskew angle (deg); None = auto until resolved
        self._split_rot_dirty = False  # True once the operator nudges rotation
        self._split_rot_src = "auto"  # "manual" / "inherited" / "auto", for the HUD
        self._split_cache_key = None
        self._split_crop = None  # cached cropped BGR preview
        self._split_auto_gutter = 0.5
        self._split_resolved_rot = 0.0
        self._split_photo = None  # cached PhotoImage so gutter nudges don't re-resize
        self._split_photo_key = None
        self._split_geom = (0, 0, 1, 1)  # ix0, iy0, dw, dh of the drawn preview
        self._gutter_cache = {}  # idx -> split-line endpoints (raw-frame fractions)

        # Restore cover/doc_start from keyframe data
        for i, kf in enumerate(self.keyframes):
            if kf.get("is_cover"):
                self.actions[i] = "cover"
            if kf.get("is_doc_start"):
                self.actions[i] = "doc_start"

        self._build_ui()
        self._bind_keys()
        self._show_current()

    def _build_ui(self):
        bg, fg, dim = "#0a0a0a", "#e2e8f0", "#64748b"
        top = tk.Frame(self.root, bg="#111", height=40)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(
            top, text="Review Keyframes", font=("Menlo", 13, "bold"), bg="#111", fg=fg
        ).pack(side="left", padx=12)
        self.lbl_counter = tk.Label(top, text="", font=("Menlo", 11), bg="#111", fg=dim)
        self.lbl_counter.pack(side="left", padx=8)
        self.lbl_stats = tk.Label(
            top, text="", font=("Menlo", 10), bg="#111", fg="#22c55e"
        )
        self.lbl_stats.pack(side="left", padx=8)
        self._button(
            top,
            self._save,
            text="Save (⌘S)",
            font=("Menlo", 10),
            bg="#3b82f6",
            fg="white",
            relief="flat",
            padx=10,
            pady=4,
        ).pack(side="right", padx=8, pady=6)

        main = tk.Frame(self.root, bg=bg)
        main.pack(fill="both", expand=True)

        img_frame = tk.Frame(main, bg=bg)
        img_frame.pack(side="left", fill="both", expand=True)
        self.lbl_info = tk.Label(img_frame, text="", font=("Menlo", 11), bg=bg, fg=dim)
        self.lbl_info.pack(pady=(8, 0))
        self.lbl_detail = tk.Label(
            img_frame, text="", font=("Menlo", 10), bg=bg, fg=dim
        )
        self.lbl_detail.pack(pady=(0, 4))
        self.canvas = tk.Canvas(img_frame, bg="#111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=12, pady=4)
        self.canvas.bind("<Configure>", lambda e: self._show_current())

        nav = tk.Frame(img_frame, bg=bg)
        nav.pack(pady=(0, 8))
        self._button(
            nav,
            self._go_prev,
            text="← Prev (A)",
            font=("Menlo", 11),
            bg="#1e293b",
            fg=fg,
            relief="flat",
            padx=16,
            pady=4,
        ).pack(side="left", padx=4)
        self._button(
            nav,
            self._go_next,
            text="Next (D) →",
            font=("Menlo", 11),
            bg="#1e293b",
            fg=fg,
            relief="flat",
            padx=16,
            pady=4,
        ).pack(side="left", padx=4)

        panel = tk.Frame(main, bg="#0f0f0f", width=230)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)
        tk.Label(panel, text="ACTION", font=("Menlo", 9), bg="#0f0f0f", fg=dim).pack(
            anchor="w", padx=12, pady=(12, 4)
        )
        self.action_buttons = {}
        for key, cfg in ACTIONS.items():
            btn = self._button(
                panel,
                lambda k=key: self._set_action(k),
                text=f"  {cfg['key']}  {cfg['label']}",
                font=("Menlo", 11),
                anchor="w",
                relief="flat",
                padx=8,
                pady=4,
                bg="#0f0f0f",
                fg="#94a3b8",
            )
            btn.pack(fill="x", padx=8, pady=2)
            self.action_buttons[key] = btn

        tk.Frame(panel, bg="#1e293b", height=1).pack(fill="x", padx=8, pady=8)
        self._button(
            panel,
            self._open_scrubber,
            text="  I   Insert Frame",
            font=("Menlo", 11),
            anchor="w",
            relief="flat",
            padx=8,
            pady=4,
            bg="#0f0f0f",
            fg="#3b82f6",
        ).pack(fill="x", padx=8, pady=2)

        self.lbl_action = tk.Label(
            panel, text="", font=("Menlo", 10, "bold"), bg="#0f0f0f", fg=dim
        )
        self.lbl_action.pack(anchor="w", padx=12, pady=(8, 4))

        hf = tk.Frame(panel, bg="#0f0f0f")
        hf.pack(side="bottom", fill="x", padx=12, pady=12)
        tk.Label(
            hf,
            text=(
                "1 Keep     2 Dup\n"
                "3 Occ      4 Other\n"
                "5 Cover    6 DocStart\n"
                "I Insert   C Center\n"
                "G Split    (auto-crop)\n"
                "←/A Prev   →/D Next\n"
                "⌘S Save"
            ),
            font=("Menlo", 9),
            bg="#0f0f0f",
            fg="#475569",
            justify="left",
        ).pack(anchor="w")

    @staticmethod
    def _button(parent, command, **kw):
        # macOS Aqua tk.Button ignores bg/fg, so use a clickable Label instead.
        kw.setdefault("cursor", "hand2")
        lbl = tk.Label(parent, **kw)
        lbl.bind("<Button-1>", lambda e: command())
        return lbl

    def _bind_keys(self):
        self.root.bind("<Right>", lambda e: self._on_arrow("right"))
        self.root.bind("<Left>", lambda e: self._on_arrow("left"))
        for k in "da":
            self.root.bind(k, lambda e, k=k: self._nav_key(k))
        for n, act in [
            ("1", "keep"),
            ("2", "dup"),
            ("3", "occlusion"),
            ("4", "other"),
            ("5", "cover"),
            ("6", "doc_start"),
        ]:
            self.root.bind(
                n,
                lambda e, a=act: (
                    self._set_action(a)
                    if not self._in_text() and not self.split_mode
                    else None
                ),
            )
        self.root.bind(
            "i",
            lambda e: (
                self._open_scrubber()
                if not self._in_text() and not self.split_mode
                else None
            ),
        )
        self.root.bind(
            "c",
            lambda e: (
                self._toggle_center()
                if not self._in_text() and not self.split_mode
                else None
            ),
        )
        self.root.bind(
            "g",
            lambda e: self._enter_split() if not self._in_text() else None,
        )
        self.root.bind(
            "G",
            lambda e: self._enter_split() if not self._in_text() else None,
        )
        self.root.bind(
            "<bracketleft>",
            lambda e: self._split_rotate(-0.25) if self.split_mode else None,
        )
        self.root.bind(
            "<bracketright>",
            lambda e: self._split_rotate(0.25) if self.split_mode else None,
        )
        self.root.bind(
            "<BackSpace>", lambda e: self._split_reset() if self.split_mode else None
        )
        self.root.bind(
            "<Return>",
            lambda e: self._split_confirm() if self.split_mode else None,
        )
        self.root.bind(
            "<Escape>",
            lambda e: self._split_cancel() if self.split_mode else None,
        )
        self.root.bind("<Command-s>", lambda e: self._save())

    def _on_arrow(self, direction):
        if self.split_mode:
            self._split_move(0.002 if direction == "right" else -0.002)
        elif not self._in_text():
            if direction == "right":
                self._go_next()
            else:
                self._go_prev()

    def _nav_key(self, k):
        if not self._in_text() and not self.split_mode:
            if k == "d":
                self._go_next()
            elif k == "a":
                self._go_prev()

    def _in_text(self):
        # No text-entry widgets remain; kept so key guards read uniformly.
        return False

    def _go_next(self):
        if self.current_idx < len(self.keyframes) - 1:
            self.current_idx += 1
            self._show_current()

    def _go_prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._show_current()

    def _show_current(self):
        if not self.keyframes:
            return
        kf = self.keyframes[self.current_idx]
        idx = self.current_idx

        self.lbl_info.config(
            text=f"[{idx+1}/{len(self.keyframes)}]  Frame {kf['frame_index']}  |  {kf.get('time_sec',0)}s"
        )
        m = kf.get("motion_value", 0)
        mc = "#22c55e" if m < 2.0 else "#f59e0b" if m < 3.0 else "#ef4444"
        self.lbl_detail.config(
            text=f"Motion: {m:.2f}  |  Sharp: {kf.get('sharpness',0):.0f}  |  Src: {kf.get('source','?')}",
            fg=mc,
        )

        self.lbl_counter.config(text=f"{idx+1} / {len(self.keyframes)}")
        n_del = len(
            [a for a in self.actions.values() if a in ("dup", "occlusion", "other")]
        )
        n_ins = len(self.pending_inserts)
        self.lbl_stats.config(text=f"Del:{n_del}  Ins:{n_ins}")

        img_path = self.paths.images / kf["filename"]
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10:
            return

        if self.split_mode:
            self._render_split(cw, ch)
            return

        try:
            img = Image.open(img_path)
            iw, ih = img.size
            scale = min(cw / iw, ch / ih, 1.0)
            dw, dh = int(iw * scale), int(ih * scale)
            img = img.resize((dw, dh), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")

            action = self.actions.get(idx)
            if action and action in ACTIONS:
                x0 = (cw - dw) // 2 - 3
                y0 = (ch - dh) // 2 - 3
                self.canvas.create_rectangle(
                    x0,
                    y0,
                    x0 + dw + 6,
                    y0 + dh + 6,
                    outline=ACTIONS[action]["color"],
                    width=3,
                )

            self.canvas.create_image(
                cw // 2, ch // 2, image=self.photo, anchor="center"
            )

            ix0, iy0 = (cw - dw) // 2, (ch - dh) // 2
            if self.show_center_guide:
                cx = ix0 + dw // 2
                self.canvas.create_line(
                    cx,
                    iy0 + int(dh * 0.03),
                    cx,
                    iy0 + int(dh * 0.97),
                    fill="#ff3333",
                    width=1,
                    dash=(6, 4),
                )

            # Show the split line exactly where p6 will cut: computed on the
            # cropped/deskewed spread (same path as G and Phase 5) and mapped
            # back onto this raw frame — including the tilt, since the cut is
            # vertical only in deskewed space. Solid if you've set an
            # override, dashed if it's the auto estimate. Press G to tune it.
            is_cover = kf.get("is_cover") or self.actions.get(idx) == "cover"
            if not is_cover:
                has_override = kf.get("gutter") is not None
                line = self._gutter_line(idx)
                if line:
                    (fxa, fya), (fxb, fyb) = line
                    self.canvas.create_line(
                        ix0 + fxa * dw,
                        iy0 + fya * dh,
                        ix0 + fxb * dw,
                        iy0 + fyb * dh,
                        fill="#22ff66",
                        width=2,
                        dash=() if has_override else (4, 4),
                    )
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(
                cw // 2, ch // 2, text=str(e), fill="#ef4444", font=("Menlo", 12)
            )

        cur_action = self.actions.get(idx)
        for key, btn in self.action_buttons.items():
            if key == cur_action:
                btn.config(
                    bg="#1e293b", fg=ACTIONS[key]["color"], font=("Menlo", 11, "bold")
                )
            else:
                btn.config(bg="#0f0f0f", fg="#94a3b8", font=("Menlo", 11))

        self.lbl_action.config(
            text=(
                f"→ {ACTIONS[cur_action]['label']}" if cur_action else "(not reviewed)"
            ),
            fg=ACTIONS[cur_action]["color"] if cur_action else "#475569",
        )

    def _set_action(self, action):
        idx = self.current_idx
        if self.actions.get(idx) == action:
            del self.actions[idx]
        else:
            self.actions[idx] = action
        self.session_log.append(
            {
                "time": datetime.now().isoformat(),
                "type": "action",
                "action": action,
                "frame": self.keyframes[idx]["frame_index"],
            }
        )
        self._show_current()

    def _toggle_center(self):
        self.show_center_guide = not self.show_center_guide
        self._show_current()

    def _open_scrubber(self):
        kf = self.keyframes[self.current_idx]
        VideoScrubber(
            self.root,
            self.video_path,
            kf["frame_index"],
            self.fps,
            self.smoothed,
            self._on_insert,
        )

    def _on_insert(self, frame_idx, frame_bgr):
        filename = f"frame{frame_idx:06d}.jpg"
        cv2.imwrite(
            str(self.paths.images / filename), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        motion = float(self.smoothed[min(frame_idx, len(self.smoothed) - 1)])
        new_kf = {
            "frame_index": frame_idx,
            "time_sec": round(frame_idx / self.fps, 2),
            "motion_value": round(motion, 4),
            "sharpness": 0.0,
            "filename": filename,
            "source": "manual_insert",
        }
        # Insert in sorted position
        insert_at = 0
        for i, kf in enumerate(self.keyframes):
            if kf["frame_index"] > frame_idx:
                insert_at = i
                break
            insert_at = i + 1
        self.keyframes.insert(insert_at, new_kf)
        # Shift action indices past the insertion point
        new_actions = {}
        for k, v in self.actions.items():
            new_actions[k + 1 if k >= insert_at else k] = v
        self.actions = new_actions
        self._gutter_cache.clear()  # indices shifted

        self.pending_inserts.append(new_kf)
        self.session_log.append(
            {"time": datetime.now().isoformat(), "type": "insert", "frame": frame_idx}
        )
        log(f"Inserted frame {frame_idx}")
        messagebox.showinfo(
            "Inserted",
            f"Frame {frame_idx} ({frame_idx/self.fps:.1f}s)\nMotion: {motion:.2f}",
        )
        self._show_current()

    # ── Split / deskew adjust ──
    def _gutter_line(self, idx):
        """Split-line endpoints in raw-frame fractional coords, or None.

        Reproduces the p5→p6 path at reduced resolution: crop + deskew with
        the resolved rotation (own override, inherited, or auto), take the
        gutter override or detect it on the crop, then map that vertical line
        back through the inverse rotation onto the raw frame. The line drawn
        in review is therefore the line p6 will actually cut, tilt included.
        Cached per frame; invalidated when an override changes."""
        if idx in self._gutter_cache:
            return self._gutter_cache[idx]
        kf = self.keyframes[idx]
        line = None
        try:
            img = cv2.imread(str(self.paths.images / kf["filename"]))
            h, w = img.shape[:2]
            s = min(1.0, 1200.0 / w)
            small = (
                cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                if s < 1.0
                else img
            )
            sh, sw = small.shape[:2]
            rot = resolve_rotation(self.keyframes, idx)
            if rot is None:
                rot = _spread_tilt(page_mask(small))
            cropped, _, (x0, cw_) = crop_double_page(small, 0.005, rot)
            frac = kf.get("gutter")
            if frac is None:
                frac = detect_gutter(cropped) / max(1, cropped.shape[1])
            gx = x0 + frac * cw_
            M = cv2.getRotationMatrix2D((sw / 2, sh / 2), rot, 1.0)
            Minv = cv2.invertAffineTransform(M)
            pts = np.array([[gx, 0, 1], [gx, sh, 1]], dtype=np.float64) @ Minv.T
            line = (
                (pts[0, 0] / sw, pts[0, 1] / sh),
                (pts[1, 0] / sw, pts[1, 1] / sh),
            )
        except Exception:
            line = None
        self._gutter_cache[idx] = line
        return line

    def _build_split_preview(self, idx):
        """Crop + deskew the frame exactly as p5 will, caching the result.

        Re-uses p5's ``crop_double_page`` so the gutter measured here (as a
        fraction of the cropped width) lands where p6 actually splits. Cached by
        (idx, rotation) so nudging the gutter doesn't re-crop."""
        rot = self._split_rot
        key = (idx, None if rot is None else round(rot, 3))
        if key == self._split_cache_key and self._split_crop is not None:
            return
        kf = self.keyframes[idx]
        img = cv2.imread(str(self.paths.images / kf["filename"]))
        if img is None:
            self._split_crop, self._split_auto_gutter = None, 0.5
            self._split_resolved_rot = rot or 0.0
            self._split_cache_key = key
            return
        if rot is None:
            rot = _spread_tilt(page_mask(img))
        cropped, _, _ = crop_double_page(img, 0.005, rot)
        gw = max(1, cropped.shape[1])
        self._split_resolved_rot = rot
        self._split_crop = cropped
        self._split_auto_gutter = detect_gutter(cropped) / gw
        self._split_cache_key = key

    def _enter_split(self):
        if self.split_mode:
            self._split_cancel()
            return
        idx = self.current_idx
        kf = self.keyframes[idx]
        if kf.get("is_cover") or self.actions.get(idx) == "cover":
            messagebox.showinfo("Split", "Covers are not split into pages.")
            return
        # Keep keyboard focus on the root so arrow keys reach the split handler.
        self.root.focus_set()
        self.split_mode = True
        own = kf.get("rotation_deg")
        # Baseline angle: this frame's own override, else one inherited from an
        # earlier correction (it propagates forward), else auto.
        self._split_rot = resolve_rotation(self.keyframes, idx)
        self._split_rot_dirty = own is not None
        self._split_rot_src = (
            "manual"
            if own is not None
            else "inherited" if self._split_rot is not None else "auto"
        )
        self._split_cache_key = None
        self._build_split_preview(idx)
        # Concretize the baseline angle so [ / ] nudge from the auto value.
        self._split_rot = self._split_resolved_rot
        self._split_frac = kf.get("gutter", self._split_auto_gutter)
        self._show_current()

    def _split_move(self, delta):
        if not self.split_mode:
            return
        self._split_frac = max(0.05, min(0.95, self._split_frac + delta))
        # Only the gutter line moves — redraw the overlay, not the (expensive)
        # base image, so nudging stays responsive on a 4K-ish frame.
        self._draw_split_overlay()

    def _split_rotate(self, delta):
        if not self.split_mode:
            return
        self._split_rot = (self._split_rot or 0.0) + delta
        self._split_rot_dirty = True
        self._split_rot_src = "manual"
        self._build_split_preview(self.current_idx)
        self._show_current()

    def _split_reset(self):
        if not self.split_mode:
            return
        kf = self.keyframes[self.current_idx]
        kf.pop("gutter", None)
        kf.pop("gutter_raw", None)
        kf.pop("rotation_deg", None)
        # Removing a rotation changes what later frames inherit, so drop all
        # cached split lines, not just this frame's.
        self._gutter_cache.clear()
        # An earlier correction may still propagate here after the reset.
        self._split_rot = resolve_rotation(self.keyframes, self.current_idx)
        self._split_rot_dirty = False
        self._split_rot_src = "inherited" if self._split_rot is not None else "auto"
        self._split_cache_key = None
        self._build_split_preview(self.current_idx)
        self._split_rot = self._split_resolved_rot
        self._split_frac = self._split_auto_gutter
        self._show_current()

    def _split_confirm(self):
        if not self.split_mode:
            return
        kf = self.keyframes[self.current_idx]
        kf["gutter"] = round(self._split_frac, 4)
        kf.pop("gutter_raw", None)  # legacy field from older sessions
        if self._split_rot_dirty:
            kf["rotation_deg"] = round(self._split_rot, 3)
            # The new rotation propagates to later frames' default split lines.
            self._gutter_cache.clear()
        else:
            self._gutter_cache.pop(self.current_idx, None)
        self.session_log.append(
            {
                "time": datetime.now().isoformat(),
                "type": "split",
                "frame": kf["frame_index"],
                "gutter": kf["gutter"],
                "rotation_deg": kf.get("rotation_deg"),
            }
        )
        self.split_mode = False
        self._show_current()

    def _split_cancel(self):
        self.split_mode = False
        self._show_current()

    def _render_split(self, cw, ch):
        """Draw the cropped preview, then the gutter overlay on top.

        The resized base image is cached (keyed by the crop result and canvas
        size), so only the cheap overlay redraws while the gutter is nudged."""
        self._build_split_preview(self.current_idx)
        self.canvas.delete("all")
        cropped = self._split_crop
        if cropped is None:
            self.canvas.create_text(
                cw // 2, ch // 2, text="(no preview)", fill="#ef4444"
            )
            return
        ih, iw = cropped.shape[:2]
        scale = min(cw / iw, ch / ih, 1.0)
        dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
        pkey = (self._split_cache_key, dw, dh)
        if pkey != self._split_photo_key or self._split_photo is None:
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb).resize((dw, dh), Image.LANCZOS)
            self._split_photo = ImageTk.PhotoImage(pil)
            self._split_photo_key = pkey
        self.photo = self._split_photo
        self.canvas.create_image(
            cw // 2, ch // 2, image=self._split_photo, anchor="center", tags="img"
        )
        self._split_geom = ((cw - dw) // 2, (ch - dh) // 2, dw, dh)
        self._draw_split_overlay()

    def _draw_split_overlay(self):
        """Redraw just the gutter line, auto-gutter ghost, and status text."""
        self.canvas.delete("ov")
        ix0, iy0, dw, dh = self._split_geom
        ax = ix0 + int(dw * self._split_auto_gutter)
        self.canvas.create_line(
            ax, iy0, ax, iy0 + dh, fill="#22ff66", width=1, dash=(3, 5), tags="ov"
        )
        gx = ix0 + int(dw * self._split_frac)
        self.canvas.create_line(
            gx, iy0, gx, iy0 + dh, fill="#22ff66", width=2, tags="ov"
        )
        self.canvas.create_text(
            self.canvas.winfo_width() // 2,
            iy0 + 14,
            text=(
                f"SPLIT — ←/→ gutter  [ / ] rotate "
                f"({self._split_resolved_rot:+.2f}° {self._split_rot_src})  "
                f"Enter save  Esc cancel  ⌫ reset   gutter={self._split_frac:.3f}"
            ),
            fill="#22ff66",
            font=("Menlo", 10),
            tags="ov",
        )

    # ── Save ──
    def _save(self):
        # Collect deletions
        del_indices = sorted(
            [i for i, a in self.actions.items() if a in ("dup", "occlusion", "other")],
            reverse=True,
        )
        deleted_info = []
        for i in del_indices:
            kf = self.keyframes[i]
            deleted_info.append(
                {
                    "frame_index": kf["frame_index"],
                    "filename": kf["filename"],
                    "reason": self.actions[i],
                }
            )
            # Delete image file
            img_path = self.paths.images / kf["filename"]
            if img_path.exists():
                img_path.unlink()

        # Capture cover/doc_start by frame_index BEFORE removing entries, so the
        # flags survive the index shifts that deletion causes.
        cover_frames = {
            self.keyframes[i]["frame_index"]
            for i, a in self.actions.items()
            if a == "cover"
        }
        docstart_frames = {
            self.keyframes[i]["frame_index"]
            for i, a in self.actions.items()
            if a == "doc_start"
        }

        # Remove deleted entries (reverse order to preserve indices)
        for i in del_indices:
            self.keyframes.pop(i)

        # Re-apply flags to the surviving keyframes, matched by frame_index. This
        # is what p5/p6 read to skip covers and reset page numbering; the gutter/
        # rotation overrides on each keyframe are left untouched.
        for kf in self.keyframes:
            fi = kf["frame_index"]
            kf["is_cover"] = fi in cover_frames
            kf["is_doc_start"] = fi in docstart_frames
            if not kf["is_cover"]:
                kf.pop("is_cover")
            if not kf["is_doc_start"]:
                kf.pop("is_doc_start")

        # Write keyframes.json
        (self.paths.json / "keyframes.json").write_text(
            json.dumps(self.keyframes, indent=2)
        )

        # Append to review log
        session = {
            "timestamp": datetime.now().isoformat(),
            "deletions": deleted_info,
            "insertions": [
                {"frame_index": ins["frame_index"]} for ins in self.pending_inserts
            ],
            "keyframe_count_after": len(self.keyframes),
        }

        rl_path = self.paths.json / "review_log.json"
        if rl_path.exists():
            rl = json.loads(rl_path.read_text())
        else:
            rl = {"sessions": []}
        rl["sessions"].append(session)
        rl_path.write_text(json.dumps(rl, indent=2))

        # Comparison plot
        self._generate_comparison_plot()

        # Reset pending
        self.pending_deletes = []
        self.pending_inserts = []
        self.actions = {}
        self._gutter_cache.clear()  # indices shifted after deletions

        # Re-flag covers/doc_starts from data
        for i, kf in enumerate(self.keyframes):
            if kf.get("is_cover"):
                self.actions[i] = "cover"
            if kf.get("is_doc_start"):
                self.actions[i] = "doc_start"

        log(
            f"Saved. {len(self.keyframes)} keyframes, {len(deleted_info)} deleted, {len(session['insertions'])} inserted"
        )
        messagebox.showinfo(
            "Saved",
            f"Keyframes: {len(self.keyframes)}\nDeleted: {len(deleted_info)}\nInserted: {len(session['insertions'])}",
        )
        self.current_idx = min(self.current_idx, len(self.keyframes) - 1)
        self._show_current()

    def _generate_comparison_plot(self):
        times = np.arange(len(self.smoothed)) / self.fps
        fig, ax = plt.subplots(1, 1, figsize=(22, 6))
        ax.plot(times, self.smoothed, linewidth=0.4, color="steelblue", alpha=0.7)
        ft = [kf["frame_index"] / self.fps for kf in self.keyframes]
        fm = [
            self.smoothed[min(kf["frame_index"], len(self.smoothed) - 1)]
            for kf in self.keyframes
        ]
        ax.plot(ft, fm, "g^", markersize=4, label=f"Current ({len(self.keyframes)})")
        ax.set_title(f"Current Keyframes: {len(self.keyframes)}")
        ax.set_xlabel("Time (s)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(str(self.paths.plots / "comparison_plot.png"), dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Review keyframes")
    parser.add_argument("output_dir")
    parser.add_argument("video")
    args = parser.parse_args()

    kf_json = Path(args.output_dir) / "json" / "keyframes.json"
    if not kf_json.exists():
        print(f"ERROR: {kf_json} not found. Run Phase 3 first.")
        sys.exit(1)

    root = tk.Tk()
    app = ReviewApp(root, args.output_dir, args.video)
    root.mainloop()


if __name__ == "__main__":
    main()
