#!/usr/bin/env python3
"""
Phase 4: Review Keyframes

Interactive GUI for reviewing, deleting, and inserting keyframes.
Logs all changes for algorithm improvement.

Usage:
  python scripts/p4_review_keyframes.py output/audiq5 recordings/audiq5.mp4

Features:
  - Browse keyframes with keyboard shortcuts
  - Delete bad frames (duplicate, hand occlusion, other)
  - Insert missing frames via video scrubber
  - Generates comparison plot and review log

Keyboard shortcuts (main view):
  Right / D       Next keyframe
  Left / A        Previous keyframe
  1               Mark: Keep (good)
  2               Mark: Delete — Duplicate
  3               Mark: Delete — Occlusion
  4               Mark: Delete — Other
  5               Mark: Cover (no split in Phase 6)
  I               Insert missing frame (opens video scrubber)
  C               Toggle center guide line
  N               Focus note field
  Cmd+S           Save and export
  Escape          Close

Keyboard shortcuts (video scrubber):
  Right           Next frame (+1)
  Left            Previous frame (-1)
  Shift+Right     Jump forward (+30 frames)
  Shift+Left      Jump backward (-30 frames)
  Up              Jump forward (+5 frames)
  Down            Jump backward (-5 frames)
  Return          Grab this frame and insert
  Escape          Cancel

Inputs:
  - output/<n>/keyframes/keyframes.json    From Phase 3
  - output/<n>/keyframes/spread_*.jpg      Keyframe images
  - output/<n>/motion/smoothed_signal.npy  For motion overlay
  - output/<n>/motion/metadata.json        For fps
  - The original video file                For video scrubber

Outputs (in output/<n>/review/):
  - review_log.json         All actions taken during review
  - final_keyframes.json    The corrected keyframe list
  - comparison_plot.png     Algorithm vs final on motion signal
  - review_summary.txt      Human-readable summary
  - (inserted frame images are saved to keyframes/)

Requirements:
  pip install Pillow opencv-python numpy matplotlib
"""

import argparse
import json
import sys
import time
import shutil
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

from utils import log, ensure_dir

ACTIONS = {
    "keep": {"color": "#22c55e", "label": "Keep", "key": "1"},
    "dup": {"color": "#f59e0b", "label": "Delete: Duplicate", "key": "2"},
    "occlusion": {"color": "#ec4899", "label": "Delete: Occlusion", "key": "3"},
    "other": {"color": "#94a3b8", "label": "Delete: Other", "key": "4"},
    "cover": {"color": "#3b82f6", "label": "Cover (no split)", "key": "5"},
}


class VideoScrubber:
    """Popup window for frame-by-frame video browsing to find missing pages."""

    def __init__(self, parent, video_path, start_frame, fps, smoothed, on_grab):
        self.parent = parent
        self.video_path = video_path
        self.fps = fps
        self.smoothed = smoothed
        self.on_grab = on_grab  # callback when user grabs a frame
        self.current_frame = start_frame

        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Window
        self.win = tk.Toplevel(parent)
        self.win.title("Video Scrubber — Find Missing Frame")
        self.win.configure(bg="#0a0a0a")
        self.win.geometry("1000x700")
        self.win.transient(parent)
        self.win.grab_set()

        # Top info bar
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

        # Canvas
        self.canvas = tk.Canvas(self.win, bg="#111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=4)

        # Bottom controls
        bot = tk.Frame(self.win, bg="#0a0a0a")
        bot.pack(fill="x", pady=8)

        tk.Label(
            bot,
            text="←/→ ±1 frame   ↑/↓ ±5   Shift+←/→ ±30   Enter=Grab   Esc=Cancel",
            font=("Menlo", 10),
            bg="#0a0a0a",
            fg="#475569",
        ).pack()

        btn_frame = tk.Frame(bot, bg="#0a0a0a")
        btn_frame.pack(pady=4)
        tk.Button(
            btn_frame,
            text="◀ -30",
            font=("Menlo", 10),
            bg="#1e293b",
            fg="#e2e8f0",
            relief="flat",
            padx=8,
            command=lambda: self._step(-30),
        ).pack(side="left", padx=2)
        tk.Button(
            btn_frame,
            text="◀ -5",
            font=("Menlo", 10),
            bg="#1e293b",
            fg="#e2e8f0",
            relief="flat",
            padx=8,
            command=lambda: self._step(-5),
        ).pack(side="left", padx=2)
        tk.Button(
            btn_frame,
            text="◀ -1",
            font=("Menlo", 10),
            bg="#1e293b",
            fg="#e2e8f0",
            relief="flat",
            padx=8,
            command=lambda: self._step(-1),
        ).pack(side="left", padx=2)
        tk.Button(
            btn_frame,
            text="+1 ▶",
            font=("Menlo", 10),
            bg="#1e293b",
            fg="#e2e8f0",
            relief="flat",
            padx=8,
            command=lambda: self._step(1),
        ).pack(side="left", padx=2)
        tk.Button(
            btn_frame,
            text="+5 ▶",
            font=("Menlo", 10),
            bg="#1e293b",
            fg="#e2e8f0",
            relief="flat",
            padx=8,
            command=lambda: self._step(5),
        ).pack(side="left", padx=2)
        tk.Button(
            btn_frame,
            text="+30 ▶",
            font=("Menlo", 10),
            bg="#1e293b",
            fg="#e2e8f0",
            relief="flat",
            padx=8,
            command=lambda: self._step(30),
        ).pack(side="left", padx=2)

        tk.Button(
            btn_frame,
            text="  ✓ GRAB  ",
            font=("Menlo", 11, "bold"),
            bg="#22c55e",
            fg="white",
            relief="flat",
            padx=16,
            command=self._grab,
        ).pack(side="left", padx=16)
        tk.Button(
            btn_frame,
            text="Cancel",
            font=("Menlo", 10),
            bg="#1e293b",
            fg="#94a3b8",
            relief="flat",
            padx=8,
            command=self._cancel,
        ).pack(side="left", padx=2)

        # Bindings
        self.win.bind("<Right>", lambda e: self._step(1))
        self.win.bind("<Left>", lambda e: self._step(-1))
        self.win.bind("<Up>", lambda e: self._step(5))
        self.win.bind("<Down>", lambda e: self._step(-5))
        self.win.bind("<Shift-Right>", lambda e: self._step(30))
        self.win.bind("<Shift-Left>", lambda e: self._step(-30))
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

        # Info
        t = self.current_frame / self.fps
        self.lbl_info.config(text=f"Frame {self.current_frame}  |  {t:.2f}s")

        motion = self.smoothed[min(self.current_frame, len(self.smoothed) - 1)]
        motion_color = (
            "#22c55e" if motion < 2.0 else "#f59e0b" if motion < 4.0 else "#ef4444"
        )
        self.lbl_motion.config(text=f"Motion: {motion:.2f}", fg=motion_color)

        # Display
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
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
    """Main review application."""

    def __init__(self, root, output_dir, video_path):
        self.root = root
        self.root.title("Phase 4: Review Keyframes")
        self.root.configure(bg="#0a0a0a")
        self.root.geometry("1200x800")

        self.output_dir = Path(output_dir)
        self.video_path = video_path
        self.keyframes_dir = self.output_dir / "keyframes"
        self.review_dir = ensure_dir(self.output_dir / "review")

        # Load data
        self.keyframes = json.loads((self.keyframes_dir / "keyframes.json").read_text())
        self.smoothed = np.load(str(self.output_dir / "motion" / "smoothed_signal.npy"))
        meta = json.loads((self.output_dir / "motion" / "metadata.json").read_text())
        self.fps = meta["fps"]

        # State
        self.current_idx = 0
        self.actions = {}  # spread_index -> action key
        self.notes = {}  # spread_index -> note string
        self.insertions = []  # list of {frame_index, position, filename, ...}
        self.action_log = []  # chronological log of all actions
        self.photo = None
        self.show_center_guide = True  # vertical center line for spine alignment

        # Crop guides: percentage of image width from left edge (0.0 to 1.0)
        self.crop_mode = False  # True when adjusting crop lines
        self.crop_selected = "left"  # which line is selected: "left" or "right"
        self.global_crop_left = None  # global left crop (percentage), None = not set
        self.global_crop_right = None  # global right crop (percentage), None = not set
        self.per_frame_crops = {}  # spread_index -> {"left": pct, "right": pct}
        self.show_crop_guides = False  # show the crop lines on all frames

        # Restore crop bounds from previous review (if exists)
        prev_log_path = self.review_dir / "review_log.json"
        if prev_log_path.exists():
            try:
                prev_log = json.loads(prev_log_path.read_text())
                crop_info = prev_log.get("crop_bounds", {})
                if "global" in crop_info:
                    self.global_crop_left = crop_info["global"]["left"]
                    self.global_crop_right = crop_info["global"]["right"]
                    self.show_crop_guides = True
                if "per_frame" in crop_info:
                    self.per_frame_crops = {
                        int(k): v for k, v in crop_info["per_frame"].items()
                    }
            except Exception:
                pass

        # Also restore per-frame crops from keyframe data
        for kf in self.keyframes:
            if kf.get("crop_bounds"):
                si = kf.get("spread_index")
                if si:
                    self.per_frame_crops[si] = kf["crop_bounds"]

        self._build_ui()
        self._bind_keys()
        self._show_current()

    def _build_ui(self):
        bg = "#0a0a0a"
        fg = "#e2e8f0"
        dim = "#64748b"

        # Top bar
        top = tk.Frame(self.root, bg="#111", height=40)
        top.pack(fill="x")
        top.pack_propagate(False)

        self.lbl_title = tk.Label(
            top, text="Review Keyframes", font=("Menlo", 13, "bold"), bg="#111", fg=fg
        )
        self.lbl_title.pack(side="left", padx=12)

        self.lbl_counter = tk.Label(top, text="", font=("Menlo", 11), bg="#111", fg=dim)
        self.lbl_counter.pack(side="left", padx=8)

        self.lbl_stats = tk.Label(
            top, text="", font=("Menlo", 10), bg="#111", fg="#22c55e"
        )
        self.lbl_stats.pack(side="left", padx=8)

        tk.Button(
            top,
            text="Save & Export (⌘S)",
            font=("Menlo", 10),
            bg="#3b82f6",
            fg="white",
            relief="flat",
            padx=10,
            command=self._save_and_export,
        ).pack(side="right", padx=8, pady=6)

        # Main area
        main = tk.Frame(self.root, bg=bg)
        main.pack(fill="both", expand=True)

        # Image area
        img_frame = tk.Frame(main, bg=bg)
        img_frame.pack(side="left", fill="both", expand=True)

        self.lbl_info = tk.Label(img_frame, text="", font=("Menlo", 11), bg=bg, fg=dim)
        self.lbl_info.pack(pady=(8, 0))

        self.lbl_motion_info = tk.Label(
            img_frame, text="", font=("Menlo", 10), bg=bg, fg=dim
        )
        self.lbl_motion_info.pack(pady=(0, 4))

        self.canvas = tk.Canvas(img_frame, bg="#111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=12, pady=4)
        self.canvas.bind("<Configure>", lambda e: self._show_current())

        # Navigation
        nav = tk.Frame(img_frame, bg=bg)
        nav.pack(pady=(0, 8))
        tk.Button(
            nav,
            text="← Prev (A)",
            font=("Menlo", 11),
            bg="#1e293b",
            fg=fg,
            relief="flat",
            padx=16,
            command=self._go_prev,
        ).pack(side="left", padx=4)
        tk.Button(
            nav,
            text="Next (D) →",
            font=("Menlo", 11),
            bg="#1e293b",
            fg=fg,
            relief="flat",
            padx=16,
            command=self._go_next,
        ).pack(side="left", padx=4)

        # Right panel
        panel = tk.Frame(main, bg="#0f0f0f", width=230)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)

        tk.Label(panel, text="ACTION", font=("Menlo", 9), bg="#0f0f0f", fg=dim).pack(
            anchor="w", padx=12, pady=(12, 4)
        )

        self.action_buttons = {}
        for key, cfg in ACTIONS.items():
            btn = tk.Button(
                panel,
                text=f"  {cfg['key']}  {cfg['label']}",
                font=("Menlo", 11),
                anchor="w",
                relief="flat",
                padx=8,
                pady=4,
                bg="#0f0f0f",
                fg="#94a3b8",
                command=lambda k=key: self._set_action(k),
            )
            btn.pack(fill="x", padx=8, pady=2)
            self.action_buttons[key] = btn

        # Insert button
        tk.Frame(panel, bg="#1e293b", height=1).pack(fill="x", padx=8, pady=8)
        tk.Button(
            panel,
            text="  I   Insert Missing Frame",
            font=("Menlo", 11),
            anchor="w",
            relief="flat",
            padx=8,
            pady=4,
            bg="#0f0f0f",
            fg="#3b82f6",
            command=self._open_scrubber,
        ).pack(fill="x", padx=8, pady=2)

        # Current action display
        self.lbl_action = tk.Label(
            panel, text="", font=("Menlo", 10, "bold"), bg="#0f0f0f", fg=dim
        )
        self.lbl_action.pack(anchor="w", padx=12, pady=(8, 4))

        # Note field
        tk.Label(panel, text="NOTE (N)", font=("Menlo", 9), bg="#0f0f0f", fg=dim).pack(
            anchor="w", padx=12, pady=(16, 4)
        )
        self.note_entry = tk.Text(
            panel,
            font=("Menlo", 10),
            bg="#111",
            fg=fg,
            insertbackground=fg,
            relief="flat",
            height=3,
            wrap="word",
            highlightthickness=1,
            highlightcolor="#334155",
            highlightbackground="#1e293b",
        )
        self.note_entry.pack(fill="x", padx=8, pady=2)
        self.note_entry.bind("<Return>", self._on_note_enter)

        # Help
        help_frame = tk.Frame(panel, bg="#0f0f0f")
        help_frame.pack(side="bottom", fill="x", padx=12, pady=12)
        tk.Label(
            help_frame,
            text="1    Keep\n2    Del: Duplicate\n3    Del: Occlusion\n4    Del: Other\n5    Cover (no split)\nI    Insert frame\nC    Toggle center guide\nL    Set crop guides\n     (←/→ move, Tab switch\n      Enter=global, Esc=cancel)\nShift+L  Per-frame crop\n←/A  Prev  →/D  Next\n⌘S   Save & Export",
            font=("Menlo", 9),
            bg="#0f0f0f",
            fg="#475569",
            justify="left",
        ).pack(anchor="w")

    def _bind_keys(self):
        self.root.bind("<Right>", lambda e: self._on_right())
        self.root.bind("<Left>", lambda e: self._on_left())
        self.root.bind(
            "d",
            lambda e: (
                self._go_next() if not self._in_text() and not self.crop_mode else None
            ),
        )
        self.root.bind(
            "a",
            lambda e: (
                self._go_prev() if not self._in_text() and not self.crop_mode else None
            ),
        )
        self.root.bind(
            "1",
            lambda e: (
                self._set_action("keep")
                if not self._in_text() and not self.crop_mode
                else None
            ),
        )
        self.root.bind(
            "2",
            lambda e: (
                self._set_action("dup")
                if not self._in_text() and not self.crop_mode
                else None
            ),
        )
        self.root.bind(
            "3",
            lambda e: (
                self._set_action("occlusion")
                if not self._in_text() and not self.crop_mode
                else None
            ),
        )
        self.root.bind(
            "4",
            lambda e: (
                self._set_action("other")
                if not self._in_text() and not self.crop_mode
                else None
            ),
        )
        self.root.bind(
            "5",
            lambda e: (
                self._set_action("cover")
                if not self._in_text() and not self.crop_mode
                else None
            ),
        )
        self.root.bind(
            "i",
            lambda e: (
                self._open_scrubber()
                if not self._in_text() and not self.crop_mode
                else None
            ),
        )
        self.root.bind(
            "c",
            lambda e: (
                self._toggle_center_guide()
                if not self._in_text() and not self.crop_mode
                else None
            ),
        )
        self.root.bind(
            "l",
            lambda e: (
                self._enter_crop_mode(per_frame=False) if not self._in_text() else None
            ),
        )
        self.root.bind(
            "L",
            lambda e: (
                self._enter_crop_mode(per_frame=True) if not self._in_text() else None
            ),
        )
        self.root.bind(
            "<Tab>", lambda e: self._crop_switch_line() if self.crop_mode else None
        )
        self.root.bind(
            "<Return>", lambda e: self._crop_confirm() if self.crop_mode else None
        )
        self.root.bind(
            "<Escape>", lambda e: self._crop_cancel() if self.crop_mode else None
        )
        self.root.bind(
            "n",
            lambda e: (
                self._focus_note()
                if not self._in_text() and not self.crop_mode
                else None
            ),
        )
        self.root.bind("<Command-s>", lambda e: self._save_and_export())

    def _on_right(self):
        if self.crop_mode:
            self._crop_move(0.005)
        elif not self._in_text():
            self._go_next()

    def _on_left(self):
        if self.crop_mode:
            self._crop_move(-0.005)
        elif not self._in_text():
            self._go_prev()

    def _in_text(self):
        return self.root.focus_get() == self.note_entry

    def _current_kf(self):
        return self.keyframes[self.current_idx]

    def _go_next(self):
        self._save_note()
        if self.current_idx < len(self.keyframes) - 1:
            self.current_idx += 1
            self._show_current()

    def _go_prev(self):
        self._save_note()
        if self.current_idx > 0:
            self.current_idx -= 1
            self._show_current()

    def _show_current(self):
        if not self.keyframes:
            return

        kf = self._current_kf()
        si = kf["spread_index"]

        # Info
        self.lbl_info.config(
            text=f"Spread #{si}  |  Frame {kf['frame_index']}  |  {kf['time_sec']}s"
        )

        motion_val = kf["motion_value"]
        motion_color = (
            "#22c55e"
            if motion_val < 2.0
            else "#f59e0b" if motion_val < 3.0 else "#ef4444"
        )
        self.lbl_motion_info.config(
            text=f"Motion: {motion_val:.2f}  |  Sharpness: {kf['sharpness']:.0f}  |  "
            f"Spread: {kf['spread_duration']:.1f}s",
            fg=motion_color,
        )

        # Counter
        self.lbl_counter.config(text=f"{self.current_idx + 1} / {len(self.keyframes)}")

        # Stats
        n_keep = sum(1 for a in self.actions.values() if a == "keep")
        n_cover = sum(1 for a in self.actions.values() if a == "cover")
        n_del = sum(
            1 for a in self.actions.values() if a in ("dup", "occlusion", "other")
        )
        n_ins = len(self.insertions)
        self.lbl_stats.config(
            text=f"Keep:{n_keep}  Cover:{n_cover}  Del:{n_del}  Ins:{n_ins}  Reviewed:{len(self.actions)}"
        )

        # Image
        img_path = self.keyframes_dir / kf["filename"]
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        try:
            img = Image.open(img_path)
            iw, ih = img.size
            scale = min(cw / iw, ch / ih, 1.0)
            img = img.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")

            # Border color based on action
            action = self.actions.get(si)
            if action and action in ACTIONS:
                border_color = ACTIONS[action]["color"]
                x0 = (cw - int(iw * scale)) // 2 - 3
                y0 = (ch - int(ih * scale)) // 2 - 3
                self.canvas.create_rectangle(
                    x0,
                    y0,
                    x0 + int(iw * scale) + 6,
                    y0 + int(ih * scale) + 6,
                    outline=border_color,
                    width=3,
                )

            self.canvas.create_image(
                cw // 2, ch // 2, image=self.photo, anchor="center"
            )

            # Center guide line (for spine alignment)
            if self.show_center_guide:
                img_display_w = int(iw * scale)
                img_display_h = int(ih * scale)
                img_x0 = (cw - img_display_w) // 2
                img_y0 = (ch - img_display_h) // 2
                center_x = img_x0 + img_display_w // 2
                # Draw from top to bottom of the image area
                self.canvas.create_line(
                    center_x,
                    img_y0 + int(img_display_h * 0.03),
                    center_x,
                    img_y0 + int(img_display_h * 0.97),
                    fill="#ff3333",
                    width=1,
                    dash=(6, 4),
                )

            # Crop guide lines
            crop_left, crop_right = self._get_crop_for_current()
            if crop_left is not None and crop_right is not None:
                img_display_w = int(iw * scale)
                img_display_h = int(ih * scale)
                img_x0 = (cw - img_display_w) // 2
                img_y0 = (ch - img_display_h) // 2

                lx = img_x0 + int(img_display_w * crop_left)
                rx = img_x0 + int(img_display_w * crop_right)
                y_top = img_y0 + int(img_display_h * 0.01)
                y_bot = img_y0 + int(img_display_h * 0.99)

                # Left crop line
                l_color = (
                    "#00ffff"
                    if (self.crop_mode and self.crop_selected == "left")
                    else "#00aaaa"
                )
                l_width = 3 if (self.crop_mode and self.crop_selected == "left") else 2
                self.canvas.create_line(
                    lx, y_top, lx, y_bot, fill=l_color, width=l_width
                )

                # Right crop line
                r_color = (
                    "#00ffff"
                    if (self.crop_mode and self.crop_selected == "right")
                    else "#00aaaa"
                )
                r_width = 3 if (self.crop_mode and self.crop_selected == "right") else 2
                self.canvas.create_line(
                    rx, y_top, rx, y_bot, fill=r_color, width=r_width
                )

                # Shade the outside regions
                self.canvas.create_rectangle(
                    img_x0, y_top, lx, y_bot, fill="black", stipple="gray25", outline=""
                )
                self.canvas.create_rectangle(
                    rx,
                    y_top,
                    img_x0 + img_display_w,
                    y_bot,
                    fill="black",
                    stipple="gray25",
                    outline="",
                )

                # Label in crop mode
                if self.crop_mode:
                    sel = self.crop_selected.upper()
                    mode_label = "PER-FRAME" if self._crop_per_frame else "GLOBAL"
                    self.canvas.create_text(
                        cw // 2,
                        img_y0 + 15,
                        text=f"CROP MODE ({mode_label}) — ←/→ move {sel} line, Tab switch, Enter confirm, Esc cancel",
                        fill="#00ffff",
                        font=("Menlo", 10),
                    )
                    self.canvas.create_text(
                        lx,
                        y_bot + 12,
                        text=f"L:{crop_left:.1%}",
                        fill=l_color,
                        font=("Menlo", 9),
                    )
                    self.canvas.create_text(
                        rx,
                        y_bot + 12,
                        text=f"R:{crop_right:.1%}",
                        fill=r_color,
                        font=("Menlo", 9),
                    )
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(
                cw // 2, ch // 2, text=f"Error: {e}", fill="#ef4444", font=("Menlo", 12)
            )

        # Action buttons highlight
        current_action = self.actions.get(si)
        for key, btn in self.action_buttons.items():
            cfg = ACTIONS[key]
            if key == current_action:
                btn.config(bg="#1e293b", fg=cfg["color"], font=("Menlo", 11, "bold"))
            else:
                btn.config(bg="#0f0f0f", fg="#94a3b8", font=("Menlo", 11))

        if current_action:
            self.lbl_action.config(
                text=f"→ {ACTIONS[current_action]['label']}",
                fg=ACTIONS[current_action]["color"],
            )
        else:
            self.lbl_action.config(text="(not reviewed)", fg="#475569")

        # Note
        self.note_entry.delete("1.0", "end")
        note = self.notes.get(si, "")
        if note:
            self.note_entry.insert("1.0", note)

    def _set_action(self, action_key):
        kf = self._current_kf()
        si = kf["spread_index"]
        old = self.actions.get(si)

        if old == action_key:
            del self.actions[si]  # toggle off
            self.action_log.append(
                {
                    "time": datetime.now().isoformat(),
                    "type": "unset_action",
                    "spread_index": si,
                    "frame_index": kf["frame_index"],
                }
            )
        else:
            self.actions[si] = action_key
            self.action_log.append(
                {
                    "time": datetime.now().isoformat(),
                    "type": "set_action",
                    "action": action_key,
                    "spread_index": si,
                    "frame_index": kf["frame_index"],
                    "time_sec": kf["time_sec"],
                }
            )

        self._show_current()

    def _save_note(self):
        if not self.keyframes:
            return
        si = self._current_kf()["spread_index"]
        text = self.note_entry.get("1.0", "end").strip()
        if text:
            self.notes[si] = text
        else:
            self.notes.pop(si, None)

    def _on_note_enter(self, event):
        self._save_note()
        self.root.focus_set()
        return "break"

    def _focus_note(self):
        self.note_entry.focus_set()

    def _toggle_center_guide(self):
        self.show_center_guide = not self.show_center_guide
        self._show_current()

    # ── Crop guide methods ────────────────────────────────────

    def _get_crop_for_current(self):
        """Get the active crop lines for the current frame. Returns (left_pct, right_pct) or (None, None)."""
        si = self._current_kf()["spread_index"]
        # Per-frame override takes priority
        if si in self.per_frame_crops:
            c = self.per_frame_crops[si]
            return c["left"], c["right"]
        # Global crop
        if self.global_crop_left is not None and self.global_crop_right is not None:
            return self.global_crop_left, self.global_crop_right
        # In crop mode with temp values
        if self.crop_mode:
            return self._crop_temp_left, self._crop_temp_right
        return None, None

    def _enter_crop_mode(self, per_frame=False):
        """Enter crop line editing mode."""
        if self.crop_mode:
            # Already in crop mode — toggle off
            self._crop_cancel()
            return

        self.crop_mode = True
        self._crop_per_frame = per_frame
        self.crop_selected = "left"

        # Start from existing values or defaults
        si = self._current_kf()["spread_index"]
        if per_frame and si in self.per_frame_crops:
            self._crop_temp_left = self.per_frame_crops[si]["left"]
            self._crop_temp_right = self.per_frame_crops[si]["right"]
        elif self.global_crop_left is not None:
            self._crop_temp_left = self.global_crop_left
            self._crop_temp_right = self.global_crop_right
        else:
            self._crop_temp_left = 0.15
            self._crop_temp_right = 0.85

        self._show_current()

    def _crop_move(self, delta):
        """Move the selected crop line by delta (fraction of image width)."""
        if not self.crop_mode:
            return
        if self.crop_selected == "left":
            self._crop_temp_left = max(0.0, min(0.48, self._crop_temp_left + delta))
        else:
            self._crop_temp_right = max(0.52, min(1.0, self._crop_temp_right + delta))
        self._show_current()

    def _crop_switch_line(self):
        """Switch between left and right crop line."""
        if not self.crop_mode:
            return
        self.crop_selected = "right" if self.crop_selected == "left" else "left"
        self._show_current()

    def _crop_confirm(self):
        """Confirm crop lines — save as global or per-frame."""
        if not self.crop_mode:
            return

        if self._crop_per_frame:
            si = self._current_kf()["spread_index"]
            self.per_frame_crops[si] = {
                "left": round(self._crop_temp_left, 4),
                "right": round(self._crop_temp_right, 4),
            }
            log(
                f"Per-frame crop set for spread {si}: L={self._crop_temp_left:.1%}, R={self._crop_temp_right:.1%}"
            )
        else:
            self.global_crop_left = round(self._crop_temp_left, 4)
            self.global_crop_right = round(self._crop_temp_right, 4)
            self.show_crop_guides = True
            log(
                f"Global crop set: L={self.global_crop_left:.1%}, R={self.global_crop_right:.1%}"
            )

        self.crop_mode = False
        self._show_current()

    def _crop_cancel(self):
        """Cancel crop mode without saving."""
        self.crop_mode = False
        self._show_current()

    def _open_scrubber(self):
        kf = self._current_kf()
        # Start scrubber at the midpoint between this spread and the next
        start_frame = kf["frame_index"]

        def on_grab(frame_idx, frame_bgr):
            self._insert_frame(frame_idx, frame_bgr)

        VideoScrubber(
            self.root, self.video_path, start_frame, self.fps, self.smoothed, on_grab
        )

    def _insert_frame(self, frame_idx, frame_bgr):
        """Save a grabbed frame and add it to the insertion list."""
        kf = self._current_kf()

        # Save the frame image
        filename = f"insert_frame{frame_idx:06d}.jpg"
        filepath = self.keyframes_dir / filename
        cv2.imwrite(str(filepath), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        motion = float(self.smoothed[min(frame_idx, len(self.smoothed) - 1)])

        insertion = {
            "frame_index": frame_idx,
            "time_sec": round(frame_idx / self.fps, 2),
            "motion_value": round(motion, 4),
            "filename": filename,
            "inserted_near_spread": kf["spread_index"],
        }
        self.insertions.append(insertion)

        self.action_log.append(
            {
                "time": datetime.now().isoformat(),
                "type": "insert",
                "frame_index": frame_idx,
                "time_sec": insertion["time_sec"],
                "near_spread": kf["spread_index"],
            }
        )

        log(
            f"Inserted frame {frame_idx} ({frame_idx/self.fps:.1f}s) near spread {kf['spread_index']}"
        )
        messagebox.showinfo(
            "Frame Inserted",
            f"Frame {frame_idx} ({frame_idx/self.fps:.1f}s) added.\n"
            f"Motion: {motion:.2f}",
        )
        self._show_current()

    def _save_and_export(self):
        self._save_note()
        log("Saving review results...")

        # Build final keyframe list
        final_keyframes = []
        deleted = []

        for kf in self.keyframes:
            si = kf["spread_index"]
            action = self.actions.get(si, "keep")
            if action == "keep" or action == "cover":
                entry = {**kf}
                if action == "cover":
                    entry["is_cover"] = True
                # Add per-frame crop if set
                if si in self.per_frame_crops:
                    entry["crop_bounds"] = self.per_frame_crops[si]
                final_keyframes.append(entry)
            elif action not in ACTIONS:
                # Unrecognized action = keep
                final_keyframes.append(kf)
            else:
                deleted.append({**kf, "reason": action, "note": self.notes.get(si, "")})

        # Add insertions
        for ins in self.insertions:
            final_keyframes.append(ins)

        # Sort by frame index
        final_keyframes.sort(key=lambda x: x["frame_index"])

        # Save final keyframes
        final_path = self.review_dir / "final_keyframes.json"
        final_path.write_text(json.dumps(final_keyframes, indent=2))

        # Save review log
        crop_info = {}
        if self.global_crop_left is not None:
            crop_info["global"] = {
                "left": self.global_crop_left,
                "right": self.global_crop_right,
            }
        if self.per_frame_crops:
            crop_info["per_frame"] = {
                str(k): v for k, v in self.per_frame_crops.items()
            }

        review_log = {
            "timestamp": datetime.now().isoformat(),
            "video_path": self.video_path,
            "original_keyframes": len(self.keyframes),
            "final_keyframes": len(final_keyframes),
            "deleted": len(deleted),
            "inserted": len(self.insertions),
            "reviewed": len(self.actions),
            "crop_bounds": crop_info,
            "deletions": deleted,
            "insertions": self.insertions,
            "notes": {str(k): v for k, v in self.notes.items()},
            "action_history": self.action_log,
        }
        log_path = self.review_dir / "review_log.json"
        log_path.write_text(json.dumps(review_log, indent=2))

        # Generate comparison plot
        self._generate_comparison_plot(final_keyframes, deleted)

        # Summary
        summary_lines = [
            "REVIEW SUMMARY",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Original keyframes: {len(self.keyframes)}",
            f"Deleted: {len(deleted)}",
            f"Inserted: {len(self.insertions)}",
            f"Final keyframes: {len(final_keyframes)}",
            "",
        ]

        if deleted:
            summary_lines.append("DELETED:")
            for d in deleted:
                note = f" — {d['note']}" if d.get("note") else ""
                summary_lines.append(
                    f"  Spread {d['spread_index']}: frame {d['frame_index']} "
                    f"({d['time_sec']}s) [{d['reason']}]{note}"
                )
            summary_lines.append("")

        if self.insertions:
            summary_lines.append("INSERTED:")
            for ins in self.insertions:
                summary_lines.append(
                    f"  Frame {ins['frame_index']} ({ins['time_sec']}s) "
                    f"near spread {ins['inserted_near_spread']}"
                )
            summary_lines.append("")

        summary_text = "\n".join(summary_lines)
        summary_path = self.review_dir / "review_summary.txt"
        summary_path.write_text(summary_text)

        # Copy final keyframe images to review/final_keyframes/
        final_kf_dir = ensure_dir(self.review_dir / "final_keyframes")
        for i, kf in enumerate(final_keyframes):
            src = self.keyframes_dir / kf["filename"]
            if src.exists():
                dst = final_kf_dir / f"{i+1:04d}_{kf['filename']}"
                shutil.copy2(src, dst)

        log(f"  Review log:      {log_path}")
        log(f"  Final keyframes: {final_path} ({len(final_keyframes)} frames)")
        log(f"  Summary:         {summary_path}")
        log(f"  Final images:    {final_kf_dir}/")

        messagebox.showinfo(
            "Saved",
            f"Review exported to {self.review_dir}\n\n"
            f"Original: {len(self.keyframes)}\n"
            f"Deleted: {len(deleted)}\n"
            f"Inserted: {len(self.insertions)}\n"
            f"Final: {len(final_keyframes)}",
        )

    def _generate_comparison_plot(self, final_keyframes, deleted):
        """Plot comparing algorithm picks vs final corrected picks."""
        times = np.arange(len(self.smoothed)) / self.fps

        fig, axes = plt.subplots(2, 1, figsize=(22, 10))

        # Algorithm picks
        ax = axes[0]
        ax.plot(times, self.smoothed, linewidth=0.4, color="steelblue", alpha=0.7)
        orig_frames = [kf["frame_index"] for kf in self.keyframes]
        orig_motions = [kf["motion_value"] for kf in self.keyframes]
        ax.plot(
            [f / self.fps for f in orig_frames],
            orig_motions,
            "g^",
            markersize=4,
            label=f"Algorithm ({len(self.keyframes)})",
        )

        # Mark deleted
        del_frames = [d["frame_index"] for d in deleted]
        del_motions = [d["motion_value"] for d in deleted]
        if del_frames:
            ax.plot(
                [f / self.fps for f in del_frames],
                del_motions,
                "rx",
                markersize=8,
                markeredgewidth=2,
                label=f"Deleted ({len(deleted)})",
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Motion")
        ax.set_title("Algorithm Selection (green = kept, red X = deleted)")
        ax.legend()

        # Final picks
        ax = axes[1]
        ax.plot(times, self.smoothed, linewidth=0.4, color="steelblue", alpha=0.7)
        final_frames = [kf["frame_index"] for kf in final_keyframes]
        final_motions = [
            self.smoothed[min(f, len(self.smoothed) - 1)] for f in final_frames
        ]
        ax.plot(
            [f / self.fps for f in final_frames],
            final_motions,
            "g^",
            markersize=4,
            label=f"Final ({len(final_keyframes)})",
        )

        # Mark insertions
        ins_frames = [ins["frame_index"] for ins in self.insertions]
        ins_motions = [
            self.smoothed[min(f, len(self.smoothed) - 1)] for f in ins_frames
        ]
        if ins_frames:
            ax.plot(
                [f / self.fps for f in ins_frames],
                ins_motions,
                "b*",
                markersize=10,
                label=f"Inserted ({len(self.insertions)})",
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Motion")
        ax.set_title("Final Selection (green = kept, blue star = inserted)")
        ax.legend()

        plt.tight_layout()
        plt.savefig(str(self.review_dir / "comparison_plot.png"), dpi=150)
        plt.close()
        log(f"  Comparison plot: {self.review_dir / 'comparison_plot.png'}")


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Review keyframes")
    parser.add_argument("output_dir", help="Base output directory (e.g. output/audiq5)")
    parser.add_argument("video", help="Path to original video file")
    args = parser.parse_args()

    # Verify inputs
    kf_json = Path(args.output_dir) / "keyframes" / "keyframes.json"
    if not kf_json.exists():
        print(f"ERROR: {kf_json} not found. Run Phase 3 first.")
        sys.exit(1)
    if not Path(args.video).exists():
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)

    root = tk.Tk()
    app = ReviewApp(root, args.output_dir, args.video)
    root.mainloop()


if __name__ == "__main__":
    main()
