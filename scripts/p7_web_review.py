#!/usr/bin/env python3
"""
Phase 7: Page Review (browser)

The same optional page review as p7_review_pages, in Chrome: drop pages,
tag First Pages (one scan → many PDFs), name documents, nudge a page's
geometry. The browser holds the working state (fast local toggles, live CSS
preview of a nudge) and mirrors every change here, where it lands in
page_review.json exactly as the Tk app's auto-save does; Save applies drops,
re-renders adjusted pages from their pristine copies, and stamps document
starts — the same code contract P9 reads.

Usage:
  python scripts/p7_web_review.py output/mybook
  ... then open http://localhost:8412 (the shared ScanStudio port) in Chrome.

  ChromeOS: forward the port first — Settings > Linux > Port forwarding.

Keys (in the browser tab): identical to the Tk review — X drop, F First
Page, G geometry (arrows move, ⇧arrows 5×, [ ] tilt, { } tilt 5×), ⏎ keep,
⎋ cancel, ⌫ reset, A/D or ←/→ navigate, Ctrl+S save.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

from utils import ProjectPaths, ensure_dir, log, segment_documents, slugify
from webui import (
    DEFAULT_PORT,
    WebUIServer,
    chromeos_note,
    send_file,
    serve_forever_in_thread,
)

JPEG_QUALITY = 95   # P6's quality, so a re-render doesn't degrade the page


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Phase 7: Page review in the browser")
    p.add_argument("output_dir", help="Base output directory (e.g. output/mybook)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    return p.parse_args(argv)


def _is_identity(g):
    return not (g.get("rot") or g.get("dx") or g.get("dy"))


def _transform(img, g):
    """PIL render of a nudge — the authoritative one the browser previews."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img.rotate(
        g["rot"],
        resample=Image.BICUBIC,
        translate=(g["dx"] * img.width, g["dy"] * img.height),
        fillcolor=(255, 255, 255),
    )


class PageSession:
    """One browser page-review session; socket-free (Tk app's state model)."""

    def __init__(self, paths, send):
        self.paths = paths
        self.send = send
        self.pages = json.loads((paths.json / "pages.json").read_text())
        self.notes = {}
        self.drops = set()
        self.geometry = {}
        self.doc_starts = {
            pg["page_num"] for pg in self.pages if pg.get("is_doc_start")
        }
        self.finished = False   # set by Finish; ends the server (finish-web)
        existing = paths.json / "page_review.json"
        if existing.exists():
            try:
                d = json.loads(existing.read_text())
                self.notes = {int(k): v for k, v in d.get("notes", {}).items()}
                self.drops = set(d.get("drops", []))
                self.geometry = {
                    int(k): v for k, v in d.get("geometry", {}).items()
                }
                self.doc_starts |= set(d.get("doc_starts", []))
            except (json.JSONDecodeError, ValueError):
                pass

    def hello(self):
        self.send(self._state_msg())

    def _state_msg(self):
        return {
            "type": "state",
            "pages": [
                {"page_num": pg["page_num"], "filename": pg["filename"],
                 "kind": pg.get("type", "?")}
                for pg in self.pages
            ],
            "notes": {str(k): v for k, v in self.notes.items()},
            "drops": sorted(self.drops),
            "geometry": {str(k): v for k, v in self.geometry.items()},
            "doc_starts": sorted(self.doc_starts),
        }

    def handle(self, m):
        t = m.get("type")
        if t == "update":
            # The browser mirrors its working state after every change —
            # the Tk app's _auto_save, with the network in the middle.
            self.notes = {
                int(k): v for k, v in m.get("notes", {}).items() if v.strip()
            }
            self.drops = set(m.get("drops", []))
            self.geometry = {
                int(k): g for k, g in m.get("geometry", {}).items()
                if not _is_identity(g)
            }
            self.doc_starts = set(m.get("doc_starts", []))
            self._auto_save()
        elif t == "save":
            self._save()
        elif t == "finish":
            self._save()
            self.finished = True
            self.send({"type": "bye"})

    def _auto_save(self):
        d = {
            "notes": {str(k): v for k, v in self.notes.items()},
            "drops": sorted(self.drops),
            "geometry": {str(k): v for k, v in self.geometry.items()},
            "doc_starts": sorted(self.doc_starts),
        }
        (self.paths.json / "page_review.json").write_text(json.dumps(d, indent=2))

    # ── Save: apply drops, re-render geometry, stamp doc starts (Tk port) ──

    def _apply_drops(self):
        if not self.drops:
            return 0
        kept = []
        for pg in self.pages:
            if pg["page_num"] in self.drops:
                for p in (self.paths.pages / pg["filename"],
                          self.paths.pages_orig / pg["filename"]):
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
        return dropped_n

    def _apply_geometry(self):
        n = 0
        for pg in self.pages:
            pn, fn = pg["page_num"], pg["filename"]
            dst, orig = self.paths.pages / fn, self.paths.pages_orig / fn
            g = self.geometry.get(pn)
            if g and not _is_identity(g):
                try:
                    if not orig.exists():
                        if not dst.exists():
                            continue
                        ensure_dir(self.paths.pages_orig)
                        shutil.copy2(dst, orig)
                    _transform(Image.open(orig), g).save(
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
        for i, pg in enumerate(self.pages):
            pn = pg["page_num"]
            tagged = pn in self.doc_starts
            if i > 0 and tagged:
                pg["is_doc_start"] = True
            else:
                pg.pop("is_doc_start", None)
            title = (self.notes.get(pn, "").strip().splitlines()
                     if tagged else [])
            if title and slugify(title[0]):
                pg["doc_title"] = title[0].strip()
            else:
                pg.pop("doc_title", None)

    def _save(self):
        dropped_n = self._apply_drops()
        rendered_n = self._apply_geometry()
        self._stamp_doc_starts()
        (self.paths.json / "pages.json").write_text(
            json.dumps(self.pages, indent=2)
        )
        self._auto_save()
        docs = segment_documents(self.pages)
        report = [
            "# Page Review",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Total: {len(self.pages)}, Dropped: {dropped_n}, "
            f"Adjusted: {rendered_n}, Documents: {len(docs)}, "
            f"Notes: {len(self.notes)}",
        ]
        ensure_dir(self.paths.reports)
        (self.paths.reports / "page_review.md").write_text("\n".join(report))
        log(f"Saved: {len(self.pages)} pages, {dropped_n} dropped, "
            f"{rendered_n} adjusted, {len(docs)} documents")
        self.send({"type": "saved", "pages": len(self.pages),
                   "dropped": dropped_n, "adjusted": rendered_n,
                   "documents": len(docs)})
        self.send(self._state_msg())


def build_server(args):
    paths = ProjectPaths(args.output_dir)
    pages_path = paths.json / "pages.json"
    if not pages_path.exists():
        raise FileNotFoundError(f"{pages_path} not found. Run Phase 6 first.")
    html_path = Path(__file__).resolve().parent.parent / "web" / "pages.html"
    if not html_path.exists():
        raise FileNotFoundError(f"pages page not found: {html_path}")

    def route(handler, path, head_only):
        if path.startswith("/page/"):
            name = Path(path[6:]).name          # no traversal
            send_file(handler, paths.pages / name, head_only)
            return True
        return False

    return WebUIServer(
        (args.host, args.port), html_path,
        lambda send: PageSession(paths, send), route,
    )


def main():
    args = parse_args()
    log("=" * 60)
    log("PHASE 7: Page Review (browser)")
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
        log("Page review finished in the browser.")
    except KeyboardInterrupt:
        log("\nDone.")
    server.shutdown()


if __name__ == "__main__":
    main()
