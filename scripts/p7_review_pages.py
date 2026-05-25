#!/usr/bin/env python3
"""
Phase 7: Page Quality Review (Optional)

Browse through the final split pages to inspect quality, make notes,
and flag pages that are especially good or bad. Outputs a shareable
report for algorithm improvement.

Usage:
  python scripts/p7_review_pages.py output/audiq5

Keyboard shortcuts:
  Right / D       Next page
  Left / A        Previous page
  1               Flag: Great quality
  2               Flag: Acceptable
  3               Flag: Poor quality
  4               Flag: Crop issue
  0 / Backspace   Clear flag
  N               Focus note field
  Cmd+S           Save and export report
  Escape          Close

Inputs:
  - output/<n>/pages/pages.json     Page metadata from Phase 6
  - output/<n>/pages/*.jpg          Page images

Outputs (in output/<n>/page_review/):
  - page_review.json                All flags and notes
  - page_review_report.md           Shareable markdown report

Requirements:
  pip install Pillow
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from utils import log, ensure_dir


FLAGS = {
    "great": {"color": "#22c55e", "label": "Great quality", "key": "1"},
    "acceptable": {"color": "#3b82f6", "label": "Acceptable", "key": "2"},
    "poor": {"color": "#f59e0b", "label": "Poor quality", "key": "3"},
    "crop_issue": {"color": "#ef4444", "label": "Crop issue", "key": "4"},
}


class PageReviewApp:
    def __init__(self, root, output_dir):
        self.root = root
        self.root.title("Phase 7: Page Quality Review")
        self.root.configure(bg="#0a0a0a")
        self.root.geometry("900x800")

        self.output_dir = Path(output_dir)
        self.pages_dir = self.output_dir / "pages"
        self.review_dir = ensure_dir(self.output_dir / "page_review")

        # Load pages
        pages_json = self.pages_dir / "pages.json"
        if not pages_json.exists():
            messagebox.showerror(
                "Error", f"{pages_json} not found.\nRun Phase 6 first."
            )
            sys.exit(1)

        self.pages = json.loads(pages_json.read_text())

        # State
        self.current_idx = 0
        self.flags = {}  # page_num -> flag key
        self.notes = {}  # page_num -> note string
        self.photo = None

        # Load existing review if present
        existing = self.review_dir / "page_review.json"
        if existing.exists():
            try:
                data = json.loads(existing.read_text())
                self.flags = {int(k): v for k, v in data.get("flags", {}).items()}
                self.notes = {int(k): v for k, v in data.get("notes", {}).items()}
            except Exception:
                pass

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

        tk.Label(
            top,
            text="Page Quality Review",
            font=("Menlo", 13, "bold"),
            bg="#111",
            fg=fg,
        ).pack(side="left", padx=12)

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

        self.lbl_detail = tk.Label(
            img_frame, text="", font=("Menlo", 10), bg=bg, fg=dim
        )
        self.lbl_detail.pack(pady=(0, 4))

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
        panel = tk.Frame(main, bg="#0f0f0f", width=220)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)

        tk.Label(panel, text="FLAG", font=("Menlo", 9), bg="#0f0f0f", fg=dim).pack(
            anchor="w", padx=12, pady=(12, 4)
        )

        self.flag_buttons = {}
        for key, cfg in FLAGS.items():
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
                command=lambda k=key: self._set_flag(k),
            )
            btn.pack(fill="x", padx=8, pady=2)
            self.flag_buttons[key] = btn

        tk.Button(
            panel,
            text="  0  Clear Flag",
            font=("Menlo", 11),
            anchor="w",
            relief="flat",
            padx=8,
            pady=4,
            bg="#0f0f0f",
            fg="#475569",
            command=self._clear_flag,
        ).pack(fill="x", padx=8, pady=2)

        self.lbl_flag = tk.Label(
            panel, text="", font=("Menlo", 10, "bold"), bg="#0f0f0f", fg=dim
        )
        self.lbl_flag.pack(anchor="w", padx=12, pady=(8, 4))

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
            height=4,
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
            text="1  Great\n2  Acceptable\n3  Poor\n4  Crop issue\n0  Clear\n←/A  Prev\n→/D  Next\nN    Note\n⌘S   Export",
            font=("Menlo", 9),
            bg="#0f0f0f",
            fg="#475569",
            justify="left",
        ).pack(anchor="w")

    def _bind_keys(self):
        self.root.bind("<Right>", lambda e: self._go_next())
        self.root.bind("<Left>", lambda e: self._go_prev())
        self.root.bind("d", lambda e: self._go_next() if not self._in_text() else None)
        self.root.bind("a", lambda e: self._go_prev() if not self._in_text() else None)
        self.root.bind(
            "1", lambda e: self._set_flag("great") if not self._in_text() else None
        )
        self.root.bind(
            "2", lambda e: self._set_flag("acceptable") if not self._in_text() else None
        )
        self.root.bind(
            "3", lambda e: self._set_flag("poor") if not self._in_text() else None
        )
        self.root.bind(
            "4", lambda e: self._set_flag("crop_issue") if not self._in_text() else None
        )
        self.root.bind(
            "0", lambda e: self._clear_flag() if not self._in_text() else None
        )
        self.root.bind(
            "<BackSpace>", lambda e: self._clear_flag() if not self._in_text() else None
        )
        self.root.bind(
            "n", lambda e: self._focus_note() if not self._in_text() else None
        )
        self.root.bind("<Command-s>", lambda e: self._save_and_export())

    def _in_text(self):
        return self.root.focus_get() == self.note_entry

    def _current_page(self):
        return self.pages[self.current_idx]

    def _go_next(self):
        self._save_note()
        if self.current_idx < len(self.pages) - 1:
            self.current_idx += 1
            self._show_current()

    def _go_prev(self):
        self._save_note()
        if self.current_idx > 0:
            self.current_idx -= 1
            self._show_current()

    def _show_current(self):
        if not self.pages:
            return

        pg = self._current_page()
        pn = pg["page_num"]

        self.lbl_info.config(text=f"Page {pn}  |  {pg['type']}  |  {pg['filename']}")
        self.lbl_detail.config(
            text=f"Size: {pg['size']}  |  Source: {pg.get('source', '?')}  |  "
            f"Crop: {pg.get('crop_method', 'n/a')}"
        )

        self.lbl_counter.config(text=f"{self.current_idx + 1} / {len(self.pages)}")

        n_flagged = len(self.flags)
        n_noted = len(self.notes)
        self.lbl_stats.config(text=f"Flagged:{n_flagged}  Notes:{n_noted}")

        # Image
        img_path = self.pages_dir / pg["filename"]
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

            flag = self.flags.get(pn)
            if flag and flag in FLAGS:
                border_color = FLAGS[flag]["color"]
                disp_w, disp_h = int(iw * scale), int(ih * scale)
                x0 = (cw - disp_w) // 2 - 3
                y0 = (ch - disp_h) // 2 - 3
                self.canvas.create_rectangle(
                    x0,
                    y0,
                    x0 + disp_w + 6,
                    y0 + disp_h + 6,
                    outline=border_color,
                    width=3,
                )

            self.canvas.create_image(
                cw // 2, ch // 2, image=self.photo, anchor="center"
            )
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(
                cw // 2, ch // 2, text=f"Error: {e}", fill="#ef4444", font=("Menlo", 12)
            )

        # Flag buttons
        current_flag = self.flags.get(pn)
        for key, btn in self.flag_buttons.items():
            cfg = FLAGS[key]
            if key == current_flag:
                btn.config(bg="#1e293b", fg=cfg["color"], font=("Menlo", 11, "bold"))
            else:
                btn.config(bg="#0f0f0f", fg="#94a3b8", font=("Menlo", 11))

        if current_flag:
            self.lbl_flag.config(
                text=f"→ {FLAGS[current_flag]['label']}",
                fg=FLAGS[current_flag]["color"],
            )
        else:
            self.lbl_flag.config(text="(not flagged)", fg="#475569")

        # Note
        self.note_entry.delete("1.0", "end")
        note = self.notes.get(pn, "")
        if note:
            self.note_entry.insert("1.0", note)

    def _set_flag(self, flag_key):
        pn = self._current_page()["page_num"]
        if self.flags.get(pn) == flag_key:
            del self.flags[pn]
        else:
            self.flags[pn] = flag_key
        self._auto_save()
        self._show_current()

    def _clear_flag(self):
        pn = self._current_page()["page_num"]
        self.flags.pop(pn, None)
        self._auto_save()
        self._show_current()

    def _save_note(self):
        if not self.pages:
            return
        pn = self._current_page()["page_num"]
        text = self.note_entry.get("1.0", "end").strip()
        if text:
            self.notes[pn] = text
        else:
            self.notes.pop(pn, None)
        self._auto_save()

    def _on_note_enter(self, event):
        self._save_note()
        self.root.focus_set()
        return "break"

    def _focus_note(self):
        self.note_entry.focus_set()

    def _auto_save(self):
        data = {
            "flags": {str(k): v for k, v in self.flags.items()},
            "notes": {str(k): v for k, v in self.notes.items()},
        }
        (self.review_dir / "page_review.json").write_text(json.dumps(data, indent=2))

    def _save_and_export(self):
        self._save_note()

        # Build report
        lines = [
            "# Page Quality Review Report",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Total pages: {len(self.pages)}",
            f"Flagged: {len(self.flags)}",
            f"Notes: {len(self.notes)}",
            "",
        ]

        # Summary counts
        lines.append("## Summary")
        counts = {}
        for f in self.flags.values():
            counts[f] = counts.get(f, 0) + 1
        for key, cfg in FLAGS.items():
            lines.append(f"- {cfg['label']}: {counts.get(key, 0)}")
        lines.append(f"- Not flagged: {len(self.pages) - len(self.flags)}")
        lines.append("")

        # Flagged pages by category
        for key, cfg in FLAGS.items():
            flagged = [
                (pg, self.notes.get(pg["page_num"], ""))
                for pg in self.pages
                if self.flags.get(pg["page_num"]) == key
            ]
            if not flagged:
                continue
            lines.append(f"## {cfg['label']} ({len(flagged)})")
            for pg, note in flagged:
                note_str = f' — "{note}"' if note else ""
                lines.append(
                    f"- Page {pg['page_num']} ({pg['type']}) | {pg['filename']} | "
                    f"{pg['size']} | crop: {pg.get('crop_method', 'n/a')}{note_str}"
                )
            lines.append("")

        # All notes
        noted = [
            (pg, self.notes[pg["page_num"]])
            for pg in self.pages
            if pg["page_num"] in self.notes
        ]
        if noted:
            lines.append("## All Notes")
            for pg, note in noted:
                flag_name = FLAGS.get(self.flags.get(pg["page_num"], ""), {}).get(
                    "label", "unflagged"
                )
                lines.append(f"- Page {pg['page_num']} ({flag_name}): \"{note}\"")
            lines.append("")

        report_text = "\n".join(lines)

        # Save
        report_path = self.review_dir / "page_review_report.md"
        report_path.write_text(report_text)

        data = {
            "flags": {str(k): v for k, v in self.flags.items()},
            "notes": {str(k): v for k, v in self.notes.items()},
        }
        (self.review_dir / "page_review.json").write_text(json.dumps(data, indent=2))

        log(f"Saved: {report_path}")
        messagebox.showinfo(
            "Saved",
            f"Report: {report_path}\n\n"
            f"Flagged: {len(self.flags)}\n"
            f"Notes: {len(self.notes)}",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Phase 7: Page quality review (optional)"
    )
    parser.add_argument("output_dir", help="Base output directory (e.g. output/audiq5)")
    args = parser.parse_args()

    root = tk.Tk()
    app = PageReviewApp(root, args.output_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
