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
  G    Adjust geometry — double: crop box + gutter; single: crop box
       Mouse: drag inside the box to move it, drag a corner or edge to resize,
       drag the gutter line to move the split. Keys: [ / ] tilt, ⇧+arrows
       resize; double adds ←/→ gutter + ↑/↓ move, single arrows move.
       Enter save, Esc cancel, ⌫ reset. Marks Keep.
  ⌘S   Save

Cropping and deskew are automatic (Phase 5) until corrected. In double mode, G
opens a geometry editor on the raw frame: a rotated crop box around the spread
with the gutter (split) line inside it. Corrections propagate forward to later
spreads until the next correction: the box and its tilt as fixed values (the
rig doesn't move between page turns), the gutter as a tracking prior — later
spreads re-detect the spine in a tight band around your last hint, so it
follows the book as it shifts and a correction stays the exception, not the
rule. Until the box is touched, Phase 5 keeps auto-detecting the spread (page
mask + safety margin); once a box is confirmed it is stored as 4 fractional
corners (``crop_quad``) and Phase 5 warps exactly that box.

Single mode (--mode single) reviews loose one-page-per-frame documents: each
keyframe is one page, so the gutter overlay is hidden. G opens the same crop
box editor (minus the gutter) because GrabCut auto-crop (Phase 5) can clip real
text or wander when page sizes vary across frames. Confirming stores the box as
4 corners on the keyframe; Phase 5 warps exactly that box instead of
auto-detecting. Unlike double mode the box does NOT propagate — loose pages
move and resize between frames.
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

from utils import (
    log,
    ProjectPaths,
    ensure_dir,
    detect_gutter,
    page_mask,
    resolve_rotation,
    resolve_gutter,
    resolve_crop_quad,
    bring_to_front,
)
from p5_crop import (
    crop_double_page,
    crop_to_quad,
    _spread_tilt,
    detect_page_quad,
    DEFAULT_SAFETY_MARGIN,
)

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
    def __init__(self, root, output_dir, video_path, mode="double"):
        self.root = root
        self.root.title(f"Phase 4: Review Keyframes ({mode} mode)")
        self.root.configure(bg="#0a0a0a")
        self.root.geometry("1200x800")

        self.paths = ProjectPaths(output_dir)
        self.video_path = video_path
        # 'double' = book spreads with a spine to split; 'single' = loose
        # one-page-per-frame docs. Single hides the gutter overlay; G opens the
        # crop editor instead of the split editor (there is no spine to cut).
        self.mode = mode

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

        # Split / geometry adjust, double mode (overrides stored per-keyframe
        # in keyframes.json). Until the operator touches the crop box, cropping
        # and deskew stay automatic in Phase 5; a confirmed box is stored as
        # crop_quad and propagates forward (see resolve_crop_quad).
        self.split_mode = False
        self._split_frac = 0.5  # gutter as fraction of the crop box's width
        self._split_auto_gutter = 0.5  # detector's suggestion, for the ghost line
        self._split_gutter_src = "auto"  # "manual" / "tracked" / "auto", for the HUD
        self._split_box_src = "auto"  # "manual" / "inherited" / "auto", for the HUD
        self._split_box_dirty = False  # True once the operator changes the box
        self._geom_cache = {}  # idx -> {"box","box_src","line"} raw-frame fractions

        # Crop box editor state, shared by both editors (they are mutually
        # exclusive): a rotated rectangle over the raw frame as center/size/
        # angle. Double mode adds the gutter line; single mode stores the box
        # per-frame with no propagation. GrabCut auto-crop can clip real text
        # or wander when page sizes vary across frames (receipts), so this is
        # the manual escape.
        self.crop_mode = False
        self._crop_cx = self._crop_cy = 0.0  # box center, raw-frame px
        self._crop_w = self._crop_h = 0.0  # box size, raw-frame px
        self._crop_angle = 0.0  # tilt, degrees
        self._crop_W = self._crop_H = 1  # frame dims, for step sizing + clamps
        self._crop_src = "auto"  # "auto" / "manual", for the single-mode HUD
        self._crop_base_photo = None  # cached base frame so edits only redraw overlay
        self._crop_base_key = None
        self._crop_geom = (0, 0, 1.0)  # ix0, iy0, px->canvas scale of the drawn frame
        self._drag = None  # active mouse drag: {"kind": ..., ...}
        # idx -> auto-crop box (fractional corners) for the single-mode preview,
        # computed lazily on first view and cached (like _geom_cache).
        self._crop_preview_cache = {}

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
        # The geometry editors are mouse-first: drag the box, its corners and
        # edges, or the gutter line. Outside an editor the canvas ignores the
        # mouse entirely.
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Motion>", self._on_mouse_hover)

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
        # G means "edit geometry": the spread split in double mode, the crop box
        # in single mode (where each frame is already one page).
        split_hint = (
            "G Split    (crop box)\n"
            if self.mode == "double"
            else "G Crop     (adjust box)\n"
        )
        tk.Label(
            hf,
            text=(
                "1 Keep     2 Dup\n"
                "3 Occ      4 Other\n"
                "5 Cover    6 DocStart\n"
                "I Insert   C Center\n"
                f"{split_hint}"
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
        for d in ("left", "right", "up", "down"):
            self.root.bind(
                f"<{d.capitalize()}>", lambda e, d=d: self._on_arrow(d, shift=False)
            )
            self.root.bind(
                f"<Shift-{d.capitalize()}>",
                lambda e, d=d: self._on_arrow(d, shift=True),
            )
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
                    if not self._in_text() and not self._editing()
                    else None
                ),
            )
        self.root.bind(
            "i",
            lambda e: (
                self._open_scrubber()
                if not self._in_text() and not self._editing()
                else None
            ),
        )
        self.root.bind(
            "c",
            lambda e: (
                self._toggle_center()
                if not self._in_text() and not self._editing()
                else None
            ),
        )
        for g in ("g", "G"):
            self.root.bind(
                g, lambda e: self._enter_geometry() if not self._in_text() else None
            )
        self.root.bind("<bracketleft>", lambda e: self._editor_rotate(-0.25))
        self.root.bind("<bracketright>", lambda e: self._editor_rotate(0.25))
        self.root.bind("<BackSpace>", lambda e: self._editor_reset())
        self.root.bind("<Return>", lambda e: self._editor_confirm())
        self.root.bind("<Escape>", lambda e: self._editor_cancel())
        self.root.bind("<Command-s>", lambda e: self._save())

    def _editing(self):
        """True while either geometry editor (split or crop) is open."""
        return self.split_mode or self.crop_mode

    # ── Editor key dispatch (split in double mode, crop in single) ──
    def _enter_geometry(self):
        if self.mode == "double":
            self._enter_split()
        else:
            self._enter_crop()

    def _editor_rotate(self, delta):
        if self.split_mode:
            self._split_rotate(delta)
        elif self.crop_mode:
            self._crop_rotate(delta)

    def _editor_reset(self):
        if self.split_mode:
            self._split_reset()
        elif self.crop_mode:
            self._crop_reset()

    def _editor_confirm(self):
        if self.split_mode:
            self._split_confirm()
        elif self.crop_mode:
            self._crop_confirm()

    def _editor_cancel(self):
        if self.split_mode:
            self._split_cancel()
        elif self.crop_mode:
            self._crop_cancel()

    def _on_arrow(self, direction, shift=False):
        if self.split_mode:
            if shift:
                # Resize the box about its center: ⇧←/→ width, ⇧↑/↓ height.
                self._apply_box_change(lambda: self._box_resize_key(direction))
            elif direction in ("left", "right"):
                self._split_move(0.002 if direction == "right" else -0.002)
            else:
                # ↑/↓ nudge the box vertically (←/→ belongs to the gutter;
                # pan sideways with the mouse).
                self._apply_box_change(lambda: self._box_pan_vertical(direction))
        elif self.crop_mode:
            self._crop_arrow(direction, shift)
        elif not self._in_text() and not shift:
            if direction == "right":
                self._go_next()
            elif direction == "left":
                self._go_prev()

    def _nav_key(self, k):
        if not self._in_text() and not self._editing():
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
        if self.crop_mode:
            self._render_crop(cw, ch)
            return
        self.canvas.configure(cursor="")  # clear any editor hover cursor

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

            # Show the crop box p5 will cut out and the split line p6 will cut
            # at, both computed on the same geometry (see _frame_geometry) and
            # drawn on this raw frame — tilt included. Dash patterns show
            # provenance: solid = this spread's own override, long dashes =
            # propagated/tracked from an earlier correction, short dashes =
            # pure auto guess. Press G to tune. Single mode has no spine, so
            # skip it entirely.
            is_cover = kf.get("is_cover") or self.actions.get(idx) == "cover"
            if self.mode == "double" and not is_cover:
                geom = self._frame_geometry(idx)
                if geom:
                    box_dash = {"manual": (), "inherited": (8, 3)}.get(
                        geom["box_src"], (4, 4)
                    )
                    pts = []
                    for fx, fy in geom["box"]:
                        pts += [ix0 + fx * dw, iy0 + fy * dh]
                    self.canvas.create_polygon(
                        pts, outline="#22ff66", fill="", width=2, dash=box_dash
                    )
                    own = kf.get("gutter") is not None
                    tracked = (not own) and resolve_gutter(
                        self.keyframes, idx
                    ) is not None
                    dash = () if own else (8, 3) if tracked else (4, 4)
                    (fxa, fya), (fxb, fyb) = geom["line"]
                    self.canvas.create_line(
                        ix0 + fxa * dw,
                        iy0 + fya * dh,
                        ix0 + fxb * dw,
                        iy0 + fyb * dh,
                        fill="#22ff66",
                        width=2,
                        dash=dash,
                    )

            # Single mode: preview the crop p5 will make, so you can spot a bad
            # auto-crop without entering the editor. A confirmed manual override
            # is solid; the auto-detected box (computed lazily, cached) is
            # dashed. Press G to adjust.
            if self.mode == "single":
                manual = kf.get("crop_quad")
                quad = manual if manual else self._auto_crop_quad(idx)
                if quad:
                    pts = []
                    for fx, fy in quad:
                        pts += [ix0 + fx * dw, iy0 + fy * dh]
                    self.canvas.create_polygon(
                        pts,
                        outline="#22ff66",
                        fill="",
                        width=2,
                        dash=() if manual else (4, 4),
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
        self._geom_cache.clear()  # indices shifted
        self._crop_preview_cache.clear()

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

    # ── Split / geometry adjust (double mode) ──
    def _resolve_margin(self, idx):
        """Auto-crop margin for keyframe ``idx``: own override, else default.

        Only feeds the auto crop on frames without a manual box (and the
        editor's auto seed). ``crop_margin`` is a legacy one-off from older
        review sessions — confirming a box replaces it — and it does not
        propagate forward."""
        return self.keyframes[idx].get("crop_margin", DEFAULT_SAFETY_MARGIN)

    @staticmethod
    def _auto_spread_quad(img, margin, rot):
        """The auto crop as a box on the raw frame: 4 px corners + the crop.

        Runs p5's ``crop_double_page`` and maps the resulting axis-aligned
        rectangle back through the inverse deskew rotation, giving the quad an
        editor or preview can draw on the raw frame. Returns
        ``(quad_px, cropped)`` — the corners as (tl, tr, br, bl) and the
        cropped BGR image for gutter detection."""
        h, w = img.shape[:2]
        cropped, method, (x0, y0, cw_, ch_) = crop_double_page(img, margin, rot)
        # crop_double_page only rotates when the angle is meaningful; the
        # fallback path never rotates at all. Mirror that so the box maps back
        # through the rotation that was actually applied.
        applied = 0.0 if (method == "fallback" or abs(rot) <= 0.2) else rot
        corners = np.array(
            [
                [x0, y0, 1.0],
                [x0 + cw_, y0, 1.0],
                [x0 + cw_, y0 + ch_, 1.0],
                [x0, y0 + ch_, 1.0],
            ],
            dtype=np.float64,
        )
        if applied:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), applied, 1.0)
            pts = corners @ cv2.invertAffineTransform(M).T
        else:
            pts = corners[:, :2]
        return [(float(x), float(y)) for x, y in pts], cropped

    def _frame_geometry(self, idx):
        """Crop box + split line for the always-on preview, or None.

        Reproduces the p5→p6 path at full resolution: the manual crop box in
        effect (own or propagated from an earlier correction), else the auto
        crop (page mask + margin) mapped back through the deskew rotation onto
        the raw frame. The split line is the gutter fraction applied between
        the box's left and right edges, so both overlays show exactly what p5
        will crop and p6 will cut, tilt included.

        Must use the full-resolution frame: the page mask, tilt, and bounds are
        resolution-dependent (a downscale can even flip ``crop_double_page``
        into its whole-frame fallback when the spread doesn't fill the frame).
        Returns ``{"box": 4 fractional corners, "box_src": "manual" /
        "inherited" / "auto", "line": 2 fractional endpoints}``. Cached per
        frame; invalidated when an override changes."""
        if idx in self._geom_cache:
            return self._geom_cache[idx]
        kf = self.keyframes[idx]
        geom = None
        try:
            img = cv2.imread(str(self.paths.images / kf["filename"]))
            h, w = img.shape[:2]
            quad = resolve_crop_quad(self.keyframes, idx)
            frac = kf.get("gutter")
            if quad is not None:
                box = [(float(x), float(y)) for x, y in quad]
                box_src = "manual" if kf.get("crop_quad") else "inherited"
                if frac is None:
                    quad_px = np.array(
                        [[x * w, y * h] for x, y in box], dtype=np.float32
                    )
                    cropped = crop_to_quad(img, quad_px, 0.0)
            else:
                rot = resolve_rotation(self.keyframes, idx)
                if rot is None:
                    rot = _spread_tilt(page_mask(img))
                quad_px, cropped = self._auto_spread_quad(
                    img, self._resolve_margin(idx), rot
                )
                box = [(x / w, y / h) for x, y in quad_px]
                box_src = "auto"
            if frac is None:
                # No own override: track the spine near the inherited prior
                # (the nearest earlier correction), else fall back to full auto.
                prior = resolve_gutter(self.keyframes, idx)
                frac = detect_gutter(cropped, prior=prior) / max(1, cropped.shape[1])
            tl, tr, br, bl = box
            geom = {
                "box": box,
                "box_src": box_src,
                "line": (
                    (tl[0] + frac * (tr[0] - tl[0]), tl[1] + frac * (tr[1] - tl[1])),
                    (bl[0] + frac * (br[0] - bl[0]), bl[1] + frac * (br[1] - bl[1])),
                ),
            }
        except Exception:
            geom = None
        self._geom_cache[idx] = geom
        return geom

    def _enter_split(self):
        # Spreads only. _enter_geometry already routes single mode to the crop
        # editor, so this guard is just belt-and-suspenders.
        if self.mode != "double":
            return
        if self.split_mode:
            self._split_cancel()
            return
        idx = self.current_idx
        kf = self.keyframes[idx]
        if kf.get("is_cover") or self.actions.get(idx) == "cover":
            messagebox.showinfo("Split", "Covers are not split into pages.")
            return
        if not self._seed_split_editor(idx):
            messagebox.showinfo("Split", "Could not read this frame.")
            return
        # Tuning a spread's geometry implies it's a page you're keeping, so
        # adopting G as a Keep too saves the separate "1" press (and overrides
        # a stale delete flag on a frame you've now decided to keep).
        self.actions[idx] = "keep"
        # Keep keyboard focus on the root so arrow keys reach the split handler.
        self.root.focus_set()
        self.split_mode = True
        self._crop_base_key = None  # force a base-frame redraw
        self._show_current()

    def _seed_split_editor(self, idx):
        """Load the editor's box + gutter from the frame's resolved geometry.

        The box seeds from this frame's own crop_quad, else one propagated
        from an earlier correction, else the auto crop (page mask + margin)
        mapped onto the raw frame — so the editor always opens showing exactly
        what Phase 5 would do. The gutter seeds from the frame's own override,
        else spine detection on that box's crop, tracking the nearest earlier
        correction as a prior (idx-1, so this frame's own override doesn't
        seed itself after a reset). Returns False if the frame can't be read."""
        kf = self.keyframes[idx]
        img = cv2.imread(str(self.paths.images / kf["filename"]))
        if img is None:
            return False
        h, w = img.shape[:2]
        self._crop_W, self._crop_H = w, h
        quad = resolve_crop_quad(self.keyframes, idx)
        if quad is not None:
            self._set_rect_from_quad([[x * w, y * h] for x, y in quad])
            self._split_box_src = "manual" if kf.get("crop_quad") else "inherited"
            cropped = crop_to_quad(
                img, np.array(self._quad_from_rect(), dtype=np.float32), 0.0
            )
        else:
            rot = resolve_rotation(self.keyframes, idx)
            if rot is None:
                rot = _spread_tilt(page_mask(img))
            quad_px, cropped = self._auto_spread_quad(
                img, self._resolve_margin(idx), rot
            )
            self._set_rect_from_quad(quad_px)
            self._split_box_src = "auto"
        self._split_box_dirty = False
        prior = resolve_gutter(self.keyframes, idx - 1) if idx > 0 else None
        self._split_auto_gutter = detect_gutter(cropped, prior=prior) / max(
            1, cropped.shape[1]
        )
        own_g = kf.get("gutter")
        self._split_frac = own_g if own_g is not None else self._split_auto_gutter
        self._split_gutter_src = (
            "manual"
            if own_g is not None
            else "tracked" if prior is not None else "auto"
        )
        return True

    def _split_move(self, delta):
        if not self.split_mode:
            return
        self._split_frac = max(0.05, min(0.95, self._split_frac + delta))
        self._split_gutter_src = "manual"
        # Only the gutter line moves — redraw the overlay, not the (expensive)
        # base image, so nudging stays responsive on a 4K-ish frame.
        self._draw_split_overlay()

    def _split_rotate(self, delta):
        if not self.split_mode:
            return

        def tilt():
            self._crop_angle += delta

        self._apply_box_change(tilt)

    def _split_reset(self):
        if not self.split_mode:
            return
        kf = self.keyframes[self.current_idx]
        kf.pop("gutter", None)
        kf.pop("gutter_raw", None)
        kf.pop("rotation_deg", None)
        kf.pop("crop_margin", None)
        kf.pop("crop_quad", None)
        # Removing overrides changes what later frames inherit, so drop all
        # cached previews, not just this frame's.
        self._geom_cache.clear()
        # An earlier correction may still propagate here after the reset — the
        # re-seed resolves the box/gutter chain afresh.
        self._seed_split_editor(self.current_idx)
        self._show_current()

    def _split_confirm(self):
        if not self.split_mode:
            return
        kf = self.keyframes[self.current_idx]
        kf["gutter"] = round(self._split_frac, 4)
        kf.pop("gutter_raw", None)  # legacy field from older sessions
        if self._split_box_dirty:
            # A touched box becomes this frame's own override: 4 fractional
            # corners for p5's warp, plus the tilt as rotation_deg so later
            # frames that still auto-crop deskew consistently. The box
            # replaces the legacy crop-margin override outright.
            kf["crop_quad"] = [
                [round(x / self._crop_W, 5), round(y / self._crop_H, 5)]
                for x, y in self._quad_from_rect()
            ]
            kf["rotation_deg"] = round(self._crop_angle, 3)
            kf.pop("crop_margin", None)
        # The gutter, rotation, and box all propagate forward as defaults, so a
        # confirm changes every later frame's inherited geometry — invalidate
        # the whole cache, not just this frame's.
        self._geom_cache.clear()
        self.session_log.append(
            {
                "time": datetime.now().isoformat(),
                "type": "split",
                "frame": kf["frame_index"],
                "gutter": kf["gutter"],
                "rotation_deg": kf.get("rotation_deg"),
                "crop_quad": kf.get("crop_quad"),
            }
        )
        self.split_mode = False
        self._show_current()

    def _split_cancel(self):
        self.split_mode = False
        self._show_current()

    def _render_split(self, cw, ch):
        """Draw the raw frame, then the crop box + gutter overlay on top."""
        self._render_frame_base(cw, ch)
        self._draw_split_overlay()

    def _draw_split_overlay(self):
        """Redraw the crop box, its handles, both gutter lines, and the HUD."""
        self.canvas.delete("ov")
        ix0, iy0, scale = self._crop_geom
        quad = self._quad_from_rect()
        pts = []
        for x, y in quad:
            pts += [ix0 + x * scale, iy0 + y * scale]
        self.canvas.create_polygon(pts, outline="#22ff66", fill="", width=2, tags="ov")
        for x, y in quad:
            hx, hy = ix0 + x * scale, iy0 + y * scale
            self.canvas.create_rectangle(
                hx - 4, hy - 4, hx + 4, hy + 4,
                outline="#22ff66", fill="#0a0a0a", tags="ov",
            )
        # Gutter (solid) and the detector's suggestion (dotted ghost), drawn
        # between the box's top and bottom edges so the tilt shows faithfully.
        for frac, width, dash in (
            (self._split_auto_gutter, 1, (3, 5)),
            (self._split_frac, 2, ()),
        ):
            (x1, y1), (x2, y2) = self._gutter_segment(frac)
            self.canvas.create_line(
                ix0 + x1 * scale, iy0 + y1 * scale,
                ix0 + x2 * scale, iy0 + y2 * scale,
                fill="#22ff66", width=width, dash=dash, tags="ov",
            )
        self.canvas.create_text(
            self.canvas.winfo_width() // 2,
            iy0 + 14,
            text=(
                f"SPLIT — drag box/corners/gutter  ←/→ gutter  ↑/↓ move  "
                f"⇧arrows size  [ / ] tilt ({self._crop_angle:+.2f}°)  "
                f"Enter save  Esc cancel  ⌫ reset   "
                f"gutter={self._split_frac:.3f} {self._split_gutter_src}  "
                f"box={self._split_box_src}"
            ),
            fill="#22ff66",
            font=("Menlo", 10),
            tags="ov",
        )

    # ── Crop adjust (single mode) ──
    def _auto_crop_quad(self, idx):
        """Auto-crop box for the always-on preview, as fractional corners.

        Mirrors what p5 will do with no override: the GrabCut detection, or its
        center-80% fallback when nothing is found. Cached per idx — the first
        view of a frame runs GrabCut (~1s), later views are instant; invalidated
        when indices shift (insert/delete)."""
        if idx in self._crop_preview_cache:
            return self._crop_preview_cache[idx]
        kf = self.keyframes[idx]
        quad = None
        try:
            img = cv2.imread(str(self.paths.images / kf["filename"]))
            h, w = img.shape[:2]
            q = detect_page_quad(img)
            if q is None:
                # p5's fallback when detection fails: center 80% of the frame.
                q = np.array(
                    [[0.1 * w, 0.1 * h], [0.9 * w, 0.1 * h],
                     [0.9 * w, 0.9 * h], [0.1 * w, 0.9 * h]],
                    dtype=np.float32,
                )
            quad = [(float(x) / w, float(y) / h) for x, y in q]
        except Exception:
            quad = None
        self._crop_preview_cache[idx] = quad
        return quad

    def _enter_crop(self):
        if self.crop_mode:
            self._crop_cancel()
            return
        idx = self.current_idx
        kf = self.keyframes[idx]
        img = cv2.imread(str(self.paths.images / kf["filename"]))
        if img is None:
            messagebox.showinfo("Crop", "Could not read this frame.")
            return
        self._crop_H, self._crop_W = img.shape[:2]
        # Tuning a crop implies it's a page you're keeping (mirrors the split
        # editor), so adopt Keep too and clear any stale delete flag.
        self.actions[idx] = "keep"
        stored = kf.get("crop_quad")
        if stored:
            quad = np.array(
                [[x * self._crop_W, y * self._crop_H] for x, y in stored],
                dtype=np.float32,
            )
            self._crop_src = "manual"
        else:
            quad = detect_page_quad(img)
            if quad is None:
                quad = self._default_quad()
                self._crop_src = "auto*"  # detector found nothing; centered guess
            else:
                self._crop_src = "auto"
        self._set_rect_from_quad(quad)
        self.crop_mode = True
        self._crop_base_key = None  # force a base-frame redraw
        self.root.focus_set()
        self._show_current()

    def _default_quad(self):
        """Centered box at 80% of the frame, for when detection fails."""
        W, H = self._crop_W, self._crop_H
        bw, bh = 0.8 * W, 0.8 * H
        cx, cy = W / 2, H / 2
        return np.array(
            [
                [cx - bw / 2, cy - bh / 2],
                [cx + bw / 2, cy - bh / 2],
                [cx + bw / 2, cy + bh / 2],
                [cx - bw / 2, cy + bh / 2],
            ],
            dtype=np.float32,
        )

    def _set_rect_from_quad(self, quad):
        """Load the editor's (center, size, angle) model from 4 corners (px).

        Works for any quad — a detector result or a stored override — by reading
        the center, the averaged side lengths, and the top-edge angle, so the
        round-trip is independent of OpenCV's minAreaRect angle convention."""
        tl, tr, br, bl = [np.asarray(p, dtype=float) for p in quad]
        c = (tl + tr + br + bl) / 4.0
        self._crop_cx, self._crop_cy = float(c[0]), float(c[1])
        self._crop_w = float((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2)
        self._crop_h = float((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2)
        self._crop_angle = float(np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0])))

    def _quad_from_rect(self):
        """The 4 corners (tl, tr, br, bl) in raw-frame px for the current box."""
        a = np.radians(self._crop_angle)
        ux, uy = np.cos(a), np.sin(a)  # unit vector along the top edge
        vx, vy = -np.sin(a), np.cos(a)  # unit vector down the left edge
        hw, hh = self._crop_w / 2, self._crop_h / 2
        cx, cy = self._crop_cx, self._crop_cy
        return [
            (cx - hw * ux - hh * vx, cy - hw * uy - hh * vy),
            (cx + hw * ux - hh * vx, cy + hw * uy - hh * vy),
            (cx + hw * ux + hh * vx, cy + hw * uy + hh * vy),
            (cx - hw * ux + hh * vx, cy - hw * uy + hh * vy),
        ]

    # ── Box math shared by both editors ──
    def _box_axes(self):
        """Unit vectors along the box's top edge (u) and down its left edge (v)."""
        a = np.radians(self._crop_angle)
        return (np.cos(a), np.sin(a)), (-np.sin(a), np.cos(a))

    def _gutter_segment(self, frac):
        """The gutter line's raw-frame px endpoints at ``frac`` of box width."""
        tl, tr, br, bl = self._quad_from_rect()
        return (
            (tl[0] + frac * (tr[0] - tl[0]), tl[1] + frac * (tr[1] - tl[1])),
            (bl[0] + frac * (br[0] - bl[0]), bl[1] + frac * (br[1] - bl[1])),
        )

    def _frac_of_point(self, pt):
        """Gutter fraction whose line passes through raw-frame point ``pt``."""
        (ux, uy), _ = self._box_axes()
        u = (pt[0] - self._crop_cx) * ux + (pt[1] - self._crop_cy) * uy
        return max(0.05, min(0.95, u / max(1e-6, self._crop_w) + 0.5))

    def _point_of_frac(self, frac):
        """Raw-frame point on the gutter line at ``frac`` (box mid-height)."""
        (ux, uy), _ = self._box_axes()
        u = (frac - 0.5) * self._crop_w
        return (self._crop_cx + u * ux, self._crop_cy + u * uy)

    def _apply_box_change(self, mutate):
        """Run a box mutation, keeping the editor's state coherent.

        In the split editor both gutter lines are pinned to their raw-frame
        spots across the change — the spine doesn't move because the crop did —
        and the box becomes an operator override. In the single-mode editor the
        box is simply marked manual. Ends with the cheap overlay redraw (the
        raw base frame never changes during an edit)."""
        if self.split_mode:
            g_pt = self._point_of_frac(self._split_frac)
            a_pt = self._point_of_frac(self._split_auto_gutter)
            mutate()
            self._split_frac = self._frac_of_point(g_pt)
            self._split_auto_gutter = self._frac_of_point(a_pt)
            self._split_box_dirty = True
            self._split_box_src = "manual"
            self._draw_split_overlay()
        else:
            mutate()
            self._crop_src = "manual"
            self._draw_crop_overlay()

    def _box_resize_key(self, direction):
        """⇧+arrow resize about the center: ←/→ width, ↑ taller, ↓ shorter."""
        W, H = self._crop_W, self._crop_H
        if direction == "left":
            self._crop_w = max(0.05 * W, self._crop_w - 0.01 * W)
        elif direction == "right":
            self._crop_w = min(1.5 * W, self._crop_w + 0.01 * W)
        elif direction == "up":
            self._crop_h = min(1.5 * H, self._crop_h + 0.01 * H)
        elif direction == "down":
            self._crop_h = max(0.05 * H, self._crop_h - 0.01 * H)

    def _box_pan_vertical(self, direction):
        H = self._crop_H
        self._crop_cy += -0.004 * H if direction == "up" else 0.004 * H
        self._crop_cy = max(0.0, min(H, self._crop_cy))

    def _drag_corner(self, k, fx, fy):
        """Resize by dragging corner ``k``, keeping the opposite corner pinned."""
        ox, oy = self._quad_from_rect()[(k + 2) % 4]
        (ux, uy), (vx, vy) = self._box_axes()
        du = (fx - ox) * ux + (fy - oy) * uy
        dv = (fx - ox) * vx + (fy - oy) * vy
        su = -1.0 if k in (0, 3) else 1.0  # the corner's side of the center
        sv = -1.0 if k in (0, 1) else 1.0
        w = max(0.05 * self._crop_W, su * du)
        h = max(0.05 * self._crop_H, sv * dv)
        self._crop_cx = ox + su * w / 2 * ux + sv * h / 2 * vx
        self._crop_cy = oy + su * w / 2 * uy + sv * h / 2 * vy
        self._crop_w, self._crop_h = w, h

    def _drag_edge(self, k, fx, fy):
        """Move edge ``k`` (0 top, 1 right, 2 bottom, 3 left) along its normal,
        keeping the opposite edge pinned."""
        (ux, uy), (vx, vy) = self._box_axes()
        cx, cy = self._crop_cx, self._crop_cy
        if k in (0, 2):
            vp = (fx - cx) * vx + (fy - cy) * vy
            hh = self._crop_h / 2
            if k == 0:
                nh = max(0.05 * self._crop_H, hh - vp)
                cv_ = hh - nh / 2
            else:
                nh = max(0.05 * self._crop_H, hh + vp)
                cv_ = -hh + nh / 2
            self._crop_cx, self._crop_cy = cx + cv_ * vx, cy + cv_ * vy
            self._crop_h = nh
        else:
            up = (fx - cx) * ux + (fy - cy) * uy
            hw = self._crop_w / 2
            if k == 3:
                nw = max(0.05 * self._crop_W, hw - up)
                cu = hw - nw / 2
            else:
                nw = max(0.05 * self._crop_W, hw + up)
                cu = -hw + nw / 2
            self._crop_cx, self._crop_cy = cx + cu * ux, cy + cu * uy
            self._crop_w = nw

    # ── Mouse editing (both geometry editors) ──
    @staticmethod
    def _seg_dist(p, a, b):
        """Distance from point ``p`` to segment ``a``–``b``."""
        px, py = p
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        t = 0.0 if l2 == 0 else max(
            0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2)
        )
        qx, qy = ax + t * dx, ay + t * dy
        return ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5

    def _hit_test(self, cx, cy):
        """What the pointer is over, in canvas px, or None.

        Priority: corner handles, then the gutter line (split editor only),
        then box edges, then the box interior — small targets win over the
        large ones they sit on."""
        ix0, iy0, scale = self._crop_geom
        pts = [
            (ix0 + x * scale, iy0 + y * scale) for x, y in self._quad_from_rect()
        ]
        for k, (hx, hy) in enumerate(pts):
            if abs(cx - hx) <= 10 and abs(cy - hy) <= 10:
                return {"kind": "corner", "corner": k}
        if self.split_mode:
            (x1, y1), (x2, y2) = self._gutter_segment(self._split_frac)
            a = (ix0 + x1 * scale, iy0 + y1 * scale)
            b = (ix0 + x2 * scale, iy0 + y2 * scale)
            if self._seg_dist((cx, cy), a, b) <= 8:
                return {"kind": "gutter"}
        for k in range(4):
            if self._seg_dist((cx, cy), pts[k], pts[(k + 1) % 4]) <= 8:
                return {"kind": "edge", "edge": k}
        fx, fy = (cx - ix0) / scale, (cy - iy0) / scale
        (ux, uy), (vx, vy) = self._box_axes()
        du = (fx - self._crop_cx) * ux + (fy - self._crop_cy) * uy
        dv = (fx - self._crop_cx) * vx + (fy - self._crop_cy) * vy
        if abs(du) <= self._crop_w / 2 and abs(dv) <= self._crop_h / 2:
            return {"kind": "move", "last": (fx, fy)}
        return None

    def _on_mouse_down(self, e):
        if not self._editing():
            return
        self._drag = self._hit_test(e.x, e.y)

    def _on_mouse_drag(self, e):
        if not self._editing() or not self._drag:
            return
        ix0, iy0, scale = self._crop_geom
        fx, fy = (e.x - ix0) / scale, (e.y - iy0) / scale
        kind = self._drag["kind"]
        if kind == "gutter":
            self._split_frac = self._frac_of_point((fx, fy))
            self._split_gutter_src = "manual"
            self._draw_split_overlay()
        elif kind == "move":
            lx, ly = self._drag["last"]
            self._drag["last"] = (fx, fy)

            def pan():
                self._crop_cx = max(
                    0.0, min(self._crop_W, self._crop_cx + fx - lx)
                )
                self._crop_cy = max(
                    0.0, min(self._crop_H, self._crop_cy + fy - ly)
                )

            self._apply_box_change(pan)
        elif kind == "corner":
            self._apply_box_change(
                lambda: self._drag_corner(self._drag["corner"], fx, fy)
            )
        elif kind == "edge":
            self._apply_box_change(
                lambda: self._drag_edge(self._drag["edge"], fx, fy)
            )

    def _on_mouse_up(self, _e):
        self._drag = None

    def _on_mouse_hover(self, e):
        """Cursor feedback so the drag targets are discoverable."""
        if not self._editing() or self._drag:
            return
        hit = self._hit_test(e.x, e.y)
        if not hit:
            cur = ""
        elif hit["kind"] == "corner":
            cur = "crosshair"
        elif hit["kind"] == "gutter":
            cur = "sb_h_double_arrow"
        elif hit["kind"] == "edge":
            cur = "sb_v_double_arrow" if hit["edge"] in (0, 2) else "sb_h_double_arrow"
        else:
            cur = "fleur"
        self.canvas.configure(cursor=cur)

    def _crop_arrow(self, direction, shift):
        W, H = self._crop_W, self._crop_H
        if shift:
            # Resize about the center: ←/→ width, ↑/↓ height.
            if direction == "left":
                self._crop_w = max(0.05 * W, self._crop_w - 0.01 * W)
            elif direction == "right":
                self._crop_w = min(1.5 * W, self._crop_w + 0.01 * W)
            elif direction == "up":
                self._crop_h = min(1.5 * H, self._crop_h + 0.01 * H)
            elif direction == "down":
                self._crop_h = max(0.05 * H, self._crop_h - 0.01 * H)
        else:
            # Pan the box, keeping its center within the frame.
            if direction == "left":
                self._crop_cx -= 0.004 * W
            elif direction == "right":
                self._crop_cx += 0.004 * W
            elif direction == "up":
                self._crop_cy -= 0.004 * H
            elif direction == "down":
                self._crop_cy += 0.004 * H
            self._crop_cx = max(0.0, min(W, self._crop_cx))
            self._crop_cy = max(0.0, min(H, self._crop_cy))
        self._crop_src = "manual"
        self._draw_crop_overlay()

    def _crop_rotate(self, delta):
        self._crop_angle += delta
        self._crop_src = "manual"
        self._draw_crop_overlay()

    def _crop_reset(self):
        """Drop the override and re-seed from the auto detector."""
        kf = self.keyframes[self.current_idx]
        kf.pop("crop_quad", None)
        img = cv2.imread(str(self.paths.images / kf["filename"]))
        quad = detect_page_quad(img) if img is not None else None
        if quad is None:
            quad = self._default_quad()
            self._crop_src = "auto*"
        else:
            self._crop_src = "auto"
        self._set_rect_from_quad(quad)
        self._show_current()

    def _crop_confirm(self):
        kf = self.keyframes[self.current_idx]
        quad = self._quad_from_rect()
        # Store corners as fractions of the frame so Phase 5 can map them back to
        # pixels and warp the box (independent of any later resize).
        kf["crop_quad"] = [
            [round(x / self._crop_W, 5), round(y / self._crop_H, 5)] for x, y in quad
        ]
        self.session_log.append(
            {
                "time": datetime.now().isoformat(),
                "type": "crop",
                "frame": kf["frame_index"],
            }
        )
        self.crop_mode = False
        self._show_current()

    def _crop_cancel(self):
        self.crop_mode = False
        self._show_current()

    def _render_frame_base(self, cw, ch):
        """Draw the raw frame scaled to the canvas, cached by frame + size.

        Both geometry editors draw on the raw frame so the operator sees the
        context around the box; only the cheap overlay redraws during edits."""
        kf = self.keyframes[self.current_idx]
        key = (self.current_idx, cw, ch)
        if key != self._crop_base_key or self._crop_base_photo is None:
            pil = Image.open(self.paths.images / kf["filename"])
            iw, ih = pil.size
            scale = min(cw / iw, ch / ih, 1.0)
            dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
            self._crop_base_photo = ImageTk.PhotoImage(
                pil.resize((dw, dh), Image.LANCZOS)
            )
            self._crop_base_key = key
            self._crop_geom = ((cw - dw) // 2, (ch - dh) // 2, scale)
        self.photo = self._crop_base_photo
        self.canvas.delete("all")
        self.canvas.create_image(
            cw // 2, ch // 2, image=self._crop_base_photo, anchor="center"
        )

    def _render_crop(self, cw, ch):
        """Draw the raw frame, then the adjustable crop box on top."""
        self._render_frame_base(cw, ch)
        self._draw_crop_overlay()

    def _draw_crop_overlay(self):
        """Redraw just the crop box, corner handles, and status text."""
        self.canvas.delete("ov")
        ix0, iy0, scale = self._crop_geom
        quad = self._quad_from_rect()
        pts = []
        for x, y in quad:
            pts += [ix0 + x * scale, iy0 + y * scale]
        self.canvas.create_polygon(
            pts, outline="#22ff66", fill="", width=2, tags="ov"
        )
        for x, y in quad:
            hx, hy = ix0 + x * scale, iy0 + y * scale
            self.canvas.create_rectangle(
                hx - 4, hy - 4, hx + 4, hy + 4,
                outline="#22ff66", fill="#0a0a0a", tags="ov",
            )
        self.canvas.create_text(
            self.canvas.winfo_width() // 2,
            iy0 + 14,
            text=(
                f"CROP — drag box/corners/edges  arrows move  ⇧+arrows size  "
                f"[ / ] tilt ({self._crop_angle:+.1f}°)  Enter save  "
                f"Esc cancel  ⌫ auto   src={self._crop_src}"
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
        self._geom_cache.clear()  # indices shifted after deletions
        self._crop_preview_cache.clear()

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
    parser.add_argument(
        "--mode",
        choices=["single", "double"],
        default="double",
        help="'double' shows the spread split line + G editor; "
        "'single' hides them for loose one-page docs (default: double)",
    )
    args = parser.parse_args()

    kf_json = Path(args.output_dir) / "json" / "keyframes.json"
    if not kf_json.exists():
        print(f"ERROR: {kf_json} not found. Run Phase 3 first.")
        sys.exit(1)

    root = tk.Tk()
    app = ReviewApp(root, args.output_dir, args.video, mode=args.mode)
    bring_to_front(root)
    root.mainloop()


if __name__ == "__main__":
    main()
