#!/usr/bin/env python3
"""Phase 7: Page Review (Optional)
Usage: python scripts/p7_review_pages.py output/mybook

Keys: →/D next  ←/A prev
      x   drop page (toggle; removed from pages/ + pages.json on Save)
      f   First Page — this page starts a new document (one scan → many PDFs).
          The note box then names it; the name becomes the PDF's filename slug.
      g   Geometry — nudge the page image inside its own frame:
          arrows translate, ⇧+arrows go 5× further, [ / ] tilt ±0.25°,
          { / } tilt ±1.25°, ⏎ keep, ⎋ cancel, ⌫ reset to as-scanned.
      ⌘S / Ctrl+S  save (⌘S on macOS, Ctrl+S elsewhere)

Typing in the note box takes the keyboard: X/F/G and the arrows deliberately
do nothing there, or they would fire mid-word. ⎋ or ⇥ hands the keys back, as
does clicking the page.

Geometry is non-destructive. The first time a page is nudged its untouched
JPEG is stashed in pages_orig/; every Save re-renders pages/<file> from that
pristine copy, so adjusting a page twice costs one re-encode rather than two
and ⌫ restores exactly what P6 produced. P8/P9 read pages/ as always and need
to know nothing about it. (P6 clears pages_orig/ when it regenerates pages/,
since those copies would no longer be the right baseline.)

Document boundaries come in from P4's "Doc Start" spread flag, which P6 lands
on the first page of that spread; F refines it to the exact page. P9 emits one
PDF per document *and* the combined whole-scan PDF, so `make pdf` / `make bw`
keep working unchanged.
"""

import argparse, json, shutil
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from utils import (
    log,
    ProjectPaths,
    ensure_dir,
    bring_to_front,
    bind_save,
    mono,
    SAVE_LABEL,
    segment_documents,
    slugify,
)

ROT_STEP = 0.25  # degrees per [ / ] press (matches P4's box tilt step)
ROT_COARSE = 1.25  # degrees per { / } press
PAN_STEP = 0.004  # translate, as a fraction of the page's own width/height
PAN_COARSE = 5.0  # ⇧+arrow multiplier
JPEG_QUALITY = 95  # P6's quality, so a re-render doesn't degrade the page


class PageReviewApp:
    def __init__(self, root, output_dir):
        self.root = root
        self.root.title("Page Review")
        self.root.configure(bg="#0a0a0a")
        self.root.geometry("900x800")
        self.paths = ProjectPaths(output_dir)
        self.review_dir = ensure_dir(self.paths.reports)
        self.pages = json.loads((self.paths.json / "pages.json").read_text())
        self.current_idx = 0
        self.notes = {}
        self.photo = None
        self.drops = set()  # page_nums marked to drop; removed on Save
        # page_num -> {"rot": deg CCW, "dx": frac of width, "dy": frac of height}.
        # Applied to pages/ on Save, from the pristine copy in pages_orig/.
        self.geometry = {}
        # page_nums that begin a new document. Seeded from pages.json (P4's
        # spread-level Doc Start, propagated by P6) and unioned with what a
        # previous P7 session saved, so neither source silently loses a tag.
        self.doc_starts = {
            pg["page_num"] for pg in self.pages if pg.get("is_doc_start")
        }
        self.geom_mode = False
        self._geom_undo = None  # geometry as of entering the editor, for ⎋
        self._disp_base = None  # page scaled to the canvas, cached for live edits
        self._disp_key = None
        self._note_pn = None  # page the note box currently holds text for
        existing = self.paths.json / "page_review.json"
        if existing.exists():
            try:
                d = json.loads(existing.read_text())
                self.notes = {int(k): v for k, v in d.get("notes", {}).items()}
                self.drops = set(d.get("drops", []))
                self.geometry = {
                    int(k): v for k, v in d.get("geometry", {}).items()
                }
                self.doc_starts |= set(d.get("doc_starts", []))
            except:
                pass
        self._build_ui()
        self._bind_keys()
        self._show_current()

    @staticmethod
    def _button(parent, command, **kw):
        # macOS Aqua tk.Button ignores bg/fg (X11 Tk honours it), so use a clickable
        # Label instead — one styling path that looks the same on both.
        kw.setdefault("cursor", "hand2")
        lbl = tk.Label(parent, **kw)
        lbl.bind("<Button-1>", lambda e: command())
        return lbl

    def _build_ui(self):
        bg, fg, dim = "#0a0a0a", "#e2e8f0", "#64748b"
        top = tk.Frame(self.root, bg="#111", height=40)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(
            top, text="Page Review", font=mono(13, "bold"), bg="#111", fg=fg
        ).pack(side="left", padx=12)
        self.lbl_counter = tk.Label(top, text="", font=mono(11), bg="#111", fg=dim)
        self.lbl_counter.pack(side="left", padx=8)
        self.lbl_docs = tk.Label(
            top, text="", font=mono(10), bg="#111", fg="#a855f7"
        )
        self.lbl_docs.pack(side="left", padx=8)
        self._button(
            top,
            self._save,
            text=f"Save ({SAVE_LABEL})",
            font=mono(10),
            bg="#3b82f6",
            fg="white",
            relief="flat",
            padx=10,
            pady=4,
        ).pack(side="right", padx=8, pady=6)
        main = tk.Frame(self.root, bg=bg)
        main.pack(fill="both", expand=True)
        self.lbl_info = tk.Label(main, text="", font=mono(11), bg=bg, fg=dim)
        self.lbl_info.pack(pady=(8, 4))
        self.canvas = tk.Canvas(main, bg="#111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=12, pady=4)
        self.canvas.bind("<Configure>", lambda e: self._show_current())
        # Clicking the page is the obvious way out of the note box.
        self.canvas.bind("<Button-1>", lambda e: self._leave_note())
        nav = tk.Frame(main, bg=bg)
        nav.pack(pady=(0, 8))
        self._button(
            nav,
            self._prev,
            text="← Prev",
            font=mono(11),
            bg="#1e293b",
            fg=fg,
            relief="flat",
            padx=16,
            pady=4,
        ).pack(side="left", padx=4)
        self._button(
            nav,
            self._next,
            text="Next →",
            font=mono(11),
            bg="#1e293b",
            fg=fg,
            relief="flat",
            padx=16,
            pady=4,
        ).pack(side="left", padx=4)
        self._button(
            nav,
            self._toggle_drop,
            text="X Drop",
            font=mono(11),
            bg="#1e293b",
            fg="#ef4444",
            relief="flat",
            padx=16,
            pady=4,
        ).pack(side="left", padx=4)
        self._button(
            nav,
            self._toggle_doc_start,
            text="F First Page",
            font=mono(11),
            bg="#1e293b",
            fg="#a855f7",
            relief="flat",
            padx=16,
            pady=4,
        ).pack(side="left", padx=4)
        self._button(
            nav,
            self._toggle_geom,
            text="G Geometry",
            font=mono(11),
            bg="#1e293b",
            fg="#22ff66",
            relief="flat",
            padx=16,
            pady=4,
        ).pack(side="left", padx=4)
        # Doubles as the document title on a First Page (see _note_label).
        self.lbl_note = tk.Label(main, text="", font=mono(9), bg=bg, fg=dim)
        self.lbl_note.pack(anchor="w", padx=12)
        self.note_entry = tk.Text(
            main,
            font=mono(10),
            bg="#111",
            fg=fg,
            insertbackground=fg,
            relief="flat",
            height=2,
            wrap="word",
        )
        self.note_entry.pack(fill="x", padx=12, pady=4)
        # Widget-level, so they beat the Text class bindings: ⎋ never reaches
        # _geom_cancel and ⇥ inserts no tab. "break" is what stops both.
        for k in ("<Escape>", "<Tab>"):
            self.note_entry.bind(k, lambda e: (self._leave_note(), "break")[1])
        for ev in ("<FocusIn>", "<FocusOut>"):
            self.note_entry.bind(ev, lambda e: self._note_label(self._note_pn))

    def _bind_keys(self):
        for d in ("Left", "Right", "Up", "Down"):
            self.root.bind(f"<{d}>", lambda e, d=d: self._on_arrow(d, coarse=False))
            self.root.bind(
                f"<Shift-{d}>", lambda e, d=d: self._on_arrow(d, coarse=True)
            )
        self.root.bind("d", lambda e: self._nav_key(self._next))
        self.root.bind("a", lambda e: self._nav_key(self._prev))
        self.root.bind("x", lambda e: self._guard(self._toggle_drop))
        for k in ("f", "F"):
            self.root.bind(k, lambda e: self._guard(self._toggle_doc_start))
        for k in ("g", "G"):
            self.root.bind(k, lambda e: None if self._in_note() else self._toggle_geom())
        self.root.bind("<bracketleft>", lambda e: self._geom_rotate(ROT_STEP))
        self.root.bind("<bracketright>", lambda e: self._geom_rotate(-ROT_STEP))
        self.root.bind("<braceleft>", lambda e: self._geom_rotate(ROT_COARSE))
        self.root.bind("<braceright>", lambda e: self._geom_rotate(-ROT_COARSE))
        self.root.bind("<BackSpace>", lambda e: self._geom_reset())
        self.root.bind("<Return>", lambda e: self._geom_confirm())
        self.root.bind("<Escape>", lambda e: self._geom_cancel())
        bind_save(self.root, self._save)

    def _in_note(self):
        # focus_get() resolves the focused widget's Tk name in the widget
        # tree, and raises for names that aren't in it — notably
        # '__tk__messagebox' while a messagebox dialog is open (a <Configure>
        # redraw can land exactly then, e.g. under Crostini's window manager).
        # Whatever holds focus in that case, it isn't the note box.
        try:
            return self.root.focus_get() == self.note_entry
        except (KeyError, tk.TclError):
            return False

    def _leave_note(self):
        """Hand the keyboard back to the review keys, harvesting what was typed.

        Without this the note box is a trap: it owns every keystroke, and
        _guard/_nav_key correctly refuse to act while it has focus."""
        if not self._in_note():
            return
        self._save_note()
        self.root.focus_set()
        self._note_label(self._note_pn)

    def _guard(self, fn, *args):
        """Run a review action unless a text field or the editor owns the key."""
        if self._in_note() or self.geom_mode:
            return
        fn(*args)

    def _nav_key(self, fn):
        if not self._in_note() and not self.geom_mode:
            fn()

    def _cur(self):
        return self.pages[self.current_idx] if self.pages else None

    def _prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._show_current()

    def _next(self):
        if self.current_idx < len(self.pages) - 1:
            self.current_idx += 1
            self._show_current()

    # ── Documents ──
    def _doc_position(self, idx):
        """(document number, page number within it, total documents)."""
        starts = [
            i for i, pg in enumerate(self.pages) if pg["page_num"] in self.doc_starts
        ]
        if not starts or starts[0] != 0:
            starts = [0] + starts  # pages before the first tag are their own doc
        doc_no = sum(1 for s in starts if s <= idx)
        within = idx - max(s for s in starts if s <= idx) + 1
        return doc_no, within, len(starts)

    def _toggle_doc_start(self):
        pg = self._cur()
        if pg is None:
            return
        pn = pg["page_num"]
        # The very first page always starts a document, so tagging it is a
        # no-op — _show_current says so rather than ignoring the keypress.
        if self.current_idx == 0:
            return
        if pn in self.doc_starts:
            self.doc_starts.discard(pn)
        else:
            self.doc_starts.add(pn)
        # Deliberately does NOT focus the note box: F is often pressed in a
        # run with D between tags, and a focused text field would swallow the
        # navigation keys. Click the box when you want to name the document.
        self._auto_save()
        self._show_current()

    # ── Geometry: nudge the page image inside its own frame ──
    def _geom(self, pn):
        return self.geometry.get(pn) or {"rot": 0.0, "dx": 0.0, "dy": 0.0}

    @staticmethod
    def _is_identity(g):
        return not (g["rot"] or g["dx"] or g["dy"])

    @staticmethod
    def _transform(img, g):
        """Rotate about the centre, then translate; white-fill what rotates in.

        ``rot`` is degrees counter-clockwise (so ``[`` matches the direction
        P4's box tilt turns the cropped content) and ``dx``/``dy`` are
        fractions of the page's own width/height, so the same stored values
        apply identically to the canvas preview and the full-resolution save."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img.rotate(
            g["rot"],
            resample=Image.BICUBIC,
            translate=(g["dx"] * img.width, g["dy"] * img.height),
            fillcolor=(255, 255, 255),
        )

    def _toggle_geom(self):
        if self.geom_mode:
            self._geom_confirm()
            return
        pg = self._cur()
        if pg is None:
            return
        self.geom_mode = True
        self._geom_undo = dict(self._geom(pg["page_num"]))
        self.root.focus_set()  # keep arrows away from the note box
        self._show_current()

    def _geom_edit(self, mutate):
        if not self.geom_mode:
            return
        pn = self._cur()["page_num"]
        g = dict(self._geom(pn))
        mutate(g)
        self.geometry[pn] = g
        self._show_current()

    def _geom_rotate(self, delta):
        self._geom_edit(lambda g: g.update(rot=round(g["rot"] + delta, 3)))

    def _geom_pan(self, direction, coarse):
        step = PAN_STEP * (PAN_COARSE if coarse else 1.0)
        dx = {"Left": -step, "Right": step}.get(direction, 0.0)
        dy = {"Up": -step, "Down": step}.get(direction, 0.0)

        def move(g):
            g["dx"] = round(max(-0.5, min(0.5, g["dx"] + dx)), 5)
            g["dy"] = round(max(-0.5, min(0.5, g["dy"] + dy)), 5)

        self._geom_edit(move)

    def _on_arrow(self, direction, coarse):
        if self.geom_mode:
            self._geom_pan(direction, coarse)
        elif not self._in_note() and not coarse:
            if direction == "Right":
                self._next()
            elif direction == "Left":
                self._prev()

    def _geom_reset(self):
        """⌫ — back to as-scanned. Save restores the pristine copy."""
        if not self.geom_mode or self._in_note():
            return
        self.geometry.pop(self._cur()["page_num"], None)
        self._show_current()

    def _geom_confirm(self):
        if not self.geom_mode or self._in_note():
            return
        pn = self._cur()["page_num"]
        if self._is_identity(self._geom(pn)):
            self.geometry.pop(pn, None)
        self.geom_mode = False
        self._geom_undo = None
        self._auto_save()
        self._show_current()

    def _geom_cancel(self):
        if not self.geom_mode:
            return
        pn = self._cur()["page_num"]
        if self._geom_undo and not self._is_identity(self._geom_undo):
            self.geometry[pn] = self._geom_undo
        else:
            self.geometry.pop(pn, None)
        self.geom_mode = False
        self._geom_undo = None
        self._show_current()

    def _show_current(self):
        if not self.pages:
            self.canvas.delete("all")
            self.lbl_info.config(text="No pages left.")
            return
        pg = self.pages[self.current_idx]
        pn = pg["page_num"]
        doc_no, within, n_docs = self._doc_position(self.current_idx)
        if self.current_idx == 0:
            start_mark = "  ◆ FIRST PAGE (implicit — page 1 always starts doc 1)"
        elif pn in self.doc_starts:
            start_mark = "  ◆ FIRST PAGE"
        else:
            start_mark = ""
        self.lbl_info.config(
            text=f"Page {pn} | {pg['type']} | {pg['filename']}{start_mark}"
        )
        self.lbl_counter.config(text=f"{self.current_idx+1}/{len(self.pages)}")
        self.lbl_docs.config(
            text=(f"doc {doc_no}/{n_docs} · p{within}" if n_docs > 1 else "")
        )
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10:
            return
        try:
            key = (self.current_idx, cw, ch)
            if key != self._disp_key or self._disp_base is None:
                img = Image.open(self.paths.pages / pg["filename"])
                iw, ih = img.size
                scale = min(cw / iw, ch / ih, 1.0)
                self._disp_base = img.resize(
                    (max(1, int(iw * scale)), max(1, int(ih * scale))), Image.LANCZOS
                )
                self._disp_key = key
            g = self._geom(pn)
            shown = self._disp_base
            if not self._is_identity(g):
                shown = self._transform(shown, g)
            self.photo = ImageTk.PhotoImage(shown)
            self.canvas.delete("all")
            self.canvas.create_image(
                cw // 2, ch // 2, image=self.photo, anchor="center"
            )
            dw, dh = shown.size
            ix0, iy0 = (cw - dw) // 2, (ch - dh) // 2
            if pn in self.doc_starts:
                self.canvas.create_rectangle(
                    ix0 - 3, iy0 - 3, ix0 + dw + 3, iy0 + dh + 3,
                    outline="#a855f7", width=3,
                )
            if self.geom_mode:
                self.canvas.create_rectangle(
                    ix0, iy0, ix0 + dw, iy0 + dh, outline="#22ff66", width=1
                )
                self.canvas.create_text(
                    cw // 2,
                    iy0 + 6,
                    anchor="n",
                    justify="center",
                    width=max(240, cw - 24),
                    text=(
                        f"GEOMETRY   tilt {g['rot']:+.2f}°   "
                        f"shift {g['dx']:+.3f}, {g['dy']:+.3f}\n"
                        "arrows move · ⇧arrows 5× · [ ] tilt · { } tilt 5× · "
                        "⏎ keep · ⎋ cancel · ⌫ reset"
                    ),
                    fill="#22ff66",
                    font=mono(10),
                )
            elif not self._is_identity(g):
                self.canvas.create_text(
                    cw // 2,
                    iy0 + 6,
                    anchor="n",
                    text=f"adjusted {g['rot']:+.2f}° — G to edit",
                    fill="#22ff66",
                    font=mono(10),
                )
            if pn in self.drops:
                self.canvas.create_rectangle(
                    2, 2, cw - 2, ch - 2, outline="#ef4444", width=4
                )
                self.canvas.create_text(
                    cw // 2,
                    24,
                    text="DROPPED — x to undo",
                    fill="#ef4444",
                    font=mono(13, "bold"),
                )
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(
                cw // 2, ch // 2, text=str(e), fill="#ef4444", font=mono(12)
            )
        self._save_note()  # whatever is in the box belongs to the page it came from
        self._note_label(pn)
        self._note_pn = pn
        self.note_entry.delete("1.0", "end")
        n = self.notes.get(pn, "")
        if n:
            self.note_entry.insert("1.0", n)

    def _note_label(self, pn):
        # The way out is only worth screen space while the box has the keys.
        hint = "   ⎋ or ⇥ back to the keys" if self._in_note() else ""
        if pn in self.doc_starts:
            self.lbl_note.config(
                text="◆ document title — click here to name this document's PDF"
                + hint,
                fg="#a855f7",
            )
        else:
            self.lbl_note.config(text="note" + hint, fg="#64748b")

    def _toggle_drop(self):
        pn = self.pages[self.current_idx]["page_num"]
        if pn in self.drops:
            self.drops.discard(pn)
        else:
            self.drops.add(pn)
        self._auto_save()
        # Auto-advance so consecutive blanks can be dropped in one tap each.
        if self.current_idx < len(self.pages) - 1:
            self.current_idx += 1
        self._show_current()

    def _save_note(self):
        """Harvest the note box into the page it was loaded for.

        Keyed on ``_note_pn`` rather than the current index, so text typed and
        then interrupted by anything that redraws — F, G, a canvas resize —
        lands on the right page instead of being overwritten by the reload."""
        pn = self._note_pn
        if pn is None:
            return
        t = self.note_entry.get("1.0", "end").strip()
        if t:
            self.notes[pn] = t
        else:
            self.notes.pop(pn, None)
        self._auto_save()

    def _auto_save(self):
        d = {
            "notes": {str(k): v for k, v in self.notes.items()},
            "drops": sorted(self.drops),
            "geometry": {str(k): v for k, v in self.geometry.items()},
            "doc_starts": sorted(self.doc_starts),
        }
        (self.paths.json / "page_review.json").write_text(json.dumps(d, indent=2))

    # ── Save: apply drops, re-render geometry, stamp document starts ──
    def _apply_drops(self):
        """Delete dropped page images (and their pristine copies); returns count."""
        if not self.drops:
            return 0
        kept = []
        for pg in self.pages:
            if pg["page_num"] in self.drops:
                for p in (
                    self.paths.pages / pg["filename"],
                    self.paths.pages_orig / pg["filename"],
                ):
                    if p.exists():
                        p.unlink()
                self.notes.pop(pg["page_num"], None)
                self.geometry.pop(pg["page_num"], None)
                self.doc_starts.discard(pg["page_num"])
            else:
                kept.append(pg)
        dropped_n = len(self.pages) - len(kept)
        self.pages = kept
        self.drops.clear()
        self.current_idx = max(0, min(self.current_idx, len(self.pages) - 1))
        return dropped_n

    def _apply_geometry(self):
        """Re-render adjusted pages from their pristine copy; returns count.

        Always rendering from pages_orig/ (stashed on first adjustment) makes
        this idempotent: re-saving, or nudging the same page again next
        session, re-encodes the original once rather than warping an
        already-warped JPEG. Clearing a page's adjustment restores the
        original and drops the copy, so pages_orig/ holds exactly the pages
        that currently differ from what P6 produced."""
        n = 0
        for pg in self.pages:
            pn, fn = pg["page_num"], pg["filename"]
            dst, orig = self.paths.pages / fn, self.paths.pages_orig / fn
            g = self.geometry.get(pn)
            if g and not self._is_identity(g):
                try:
                    if not orig.exists():
                        if not dst.exists():
                            continue
                        ensure_dir(self.paths.pages_orig)
                        shutil.copy2(dst, orig)
                    self._transform(Image.open(orig), g).save(
                        dst, quality=JPEG_QUALITY
                    )
                    pg["geometry"] = {k: g[k] for k in ("rot", "dx", "dy")}
                    n += 1
                except Exception as e:
                    log(f"  WARNING: could not re-render {fn}: {e}")
            elif orig.exists():
                shutil.copy2(orig, dst)
                orig.unlink()
                pg.pop("geometry", None)
        if self.paths.pages_orig.exists() and not any(
            self.paths.pages_orig.iterdir()
        ):
            self.paths.pages_orig.rmdir()
        return n

    def _stamp_doc_starts(self):
        """Write document boundaries + titles onto the pages.json entries.

        This is the only thing P9 reads to split the scan into per-document
        PDFs — the note on a First Page becomes that document's title (and its
        filename slug). The first page of the scan is a document start
        implicitly, so it is never stamped — but it does keep its title, or
        dropping the old page 1 would silently discard the name of whichever
        page inherits the slot."""
        for i, pg in enumerate(self.pages):
            pn = pg["page_num"]
            tagged = pn in self.doc_starts
            if i > 0 and tagged:
                pg["is_doc_start"] = True
            else:
                pg.pop("is_doc_start", None)
            title = self.notes.get(pn, "").strip().splitlines() if tagged else []
            if title and slugify(title[0]):
                pg["doc_title"] = title[0].strip()
            else:
                pg.pop("doc_title", None)

    def _save(self):
        self._save_note()
        dropped_n = self._apply_drops()
        rendered_n = self._apply_geometry()
        self._stamp_doc_starts()
        (self.paths.json / "pages.json").write_text(json.dumps(self.pages, indent=2))
        self._auto_save()
        docs = segment_documents(self.pages)
        lines = [
            "# Page Review",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Total: {len(self.pages)}, Dropped: {dropped_n}, "
            f"Adjusted: {rendered_n}, Documents: {len(docs)}, "
            f"Notes: {len(self.notes)}",
            "",
        ]
        if len(docs) > 1:
            lines.append(f"## Documents ({len(docs)})")
            for i, doc in enumerate(docs, 1):
                first, last = doc["pages"][0], doc["pages"][-1]
                title = doc["title"] or "(untitled)"
                lines.append(
                    f"- {i:02d}. {title} — pages {first['page_num']}–"
                    f"{last['page_num']} ({len(doc['pages'])})"
                )
            lines.append("")
        # Adjusted pages get their transform recorded here too, so a later
        # session can see at a glance which pages were nudged and by how much.
        adjusted = [pg for pg in self.pages if pg.get("geometry")]
        if adjusted:
            lines.append(f"## Adjusted ({len(adjusted)})")
            for pg in adjusted:
                g = pg["geometry"]
                lines.append(
                    f"- Page {pg['page_num']} | {pg['filename']} — "
                    f"tilt {g['rot']:+.2f}°, shift {g['dx']:+.3f}, {g['dy']:+.3f}"
                )
            lines.append("")
        noted = [
            (pg, self.notes[pg["page_num"]])
            for pg in self.pages
            if self.notes.get(pg["page_num"])
        ]
        if noted:
            lines.append(f"## Notes ({len(noted)})")
            for pg, note in noted:
                mark = "◆ " if pg.get("is_doc_start") else ""
                lines.append(
                    f"- {mark}Page {pg['page_num']} | {pg['filename']} — \"{note}\""
                )
            lines.append("")
        (self.paths.reports / "page_review_report.md").write_text("\n".join(lines))
        self._disp_key = None  # a re-rendered page must be reloaded from disk
        log(
            f"Saved. {len(self.pages)} pages, {dropped_n} dropped, "
            f"{rendered_n} adjusted, {len(docs)} document(s)"
        )
        messagebox.showinfo(
            "Saved",
            f"Pages: {len(self.pages)}, Dropped: {dropped_n}\n"
            f"Adjusted: {rendered_n}, Documents: {len(docs)}",
        )
        self._show_current()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()
    root = tk.Tk()
    PageReviewApp(root, args.output_dir)
    bring_to_front(root)
    root.mainloop()


if __name__ == "__main__":
    main()
