#!/usr/bin/env python3
"""Phase 9: Build PDF
Usage: python scripts/p9_build_pdf.py output/mybook
       python scripts/p9_build_pdf.py output/mybook --source bw

One scan can hold several documents (chapters, articles, a run of receipts).
Where each begins is marked during review — P4's "Doc Start" on a spread,
refined to the exact page with P7's F — and arrives here as ``is_doc_start``
on the pages.json entries. When there is more than one, this writes a PDF per
document *alongside* the combined whole-scan PDF, so `make pdf`, `make bw` and
anything else expecting <NAME>.pdf keep working untouched. The per-document
files are named after the combined one: mybook_01_preface.pdf, and for
--source bw, mybook_bw_01_preface.pdf. json/documents.json records the
segments and which PDF each produced."""

import argparse, json, sys, time, tempfile, os
from pathlib import Path
import cv2
from PIL import Image
from utils import (
    log,
    ProjectPaths,
    ensure_dir,
    check_overwrite,
    segment_documents,
)

def build_pdf(pages, source_dir, output_path, quality, lossless=False):
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.utils import ImageReader
    if not pages: return
    c = None; count = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, pg in enumerate(pages):
            ip = source_dir / pg["filename"]
            # P8 writes lossless PNG; pages.json still records the .jpg source name.
            if not ip.exists(): ip = ip.with_suffix(".png")
            if not ip.exists(): continue
            if lossless:
                # Bitonal B&W: embed 1-bit losslessly (reportlab → CCITT/Flate).
                # JPEG re-encoding here would ring around the edges and bloat the
                # file — defeating the whole point of crisp binarization.
                src = Image.open(str(ip)).convert("1", dither=Image.Dither.NONE)
                iw, ih = src.size; draw = ImageReader(src)
            else:
                img = cv2.imread(str(ip))
                if img is None: continue
                ih, iw = img.shape[:2]
                draw = os.path.join(tmp, f"p_{i:04d}.jpg")
                cv2.imwrite(draw, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            pw = 595; ph = pw * ih / iw
            if c is None: c = rl_canvas.Canvas(str(output_path), pagesize=(pw, ph))
            c._pagesize = (pw, ph)
            c.drawImage(draw, 0, 0, width=pw, height=ph)
            c.showPage(); count += 1
    if c: c.save()
    sz = output_path.stat().st_size / (1024*1024)
    log(f"  {output_path} ({sz:.1f} MB, {count} pages)")

def doc_filename(stem, i, doc):
    """<stem>_03_chapter-name.pdf — the slug is dropped when untitled."""
    slug = doc["slug"]
    return f"{stem}_{i:02d}{'_' + slug if slug else ''}.pdf"

def write_manifest(paths, docs, source, names):
    """Record the segments in json/documents.json, merging across sources.

    A project is usually built twice (color, then B&W), so the per-source PDF
    names accumulate on one entry instead of the second run erasing the first."""
    mf = paths.json / "documents.json"
    prev = {}
    if mf.exists():
        try:
            prev = {d["index"]: d.get("pdfs", {})
                    for d in json.loads(mf.read_text()).get("documents", [])}
        except Exception:
            prev = {}
    out = []
    for i, (doc, name) in enumerate(zip(docs, names), 1):
        pdfs = dict(prev.get(i, {})); pdfs[source] = name
        out.append({
            "index": i,
            "title": doc["title"],
            "slug": doc["slug"],
            "page_start": doc["pages"][0]["page_num"],
            "page_end": doc["pages"][-1]["page_num"],
            "n_pages": len(doc["pages"]),
            "pdfs": pdfs,
        })
    mf.write_text(json.dumps({"documents": out}, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--source", default="pages", choices=["pages", "bw"])
    parser.add_argument("--pdf-name", default=None,
                        help="Output filename (default: <project>.pdf)")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--split-docs", choices=["auto", "never"], default="auto",
                        help="'auto' also writes one PDF per document when "
                             "pages.json has document starts (default)")
    args = parser.parse_args()

    log("=" * 60); log("PHASE 9: Build PDF"); log("=" * 60)
    paths = ProjectPaths(args.output_dir)
    ensure_dir(paths.pdf)
    source_dir = paths.pages if args.source == "pages" else (paths.base / "bw")
    pages = json.loads((paths.json / "pages.json").read_text())
    pdf_name = args.pdf_name or f"{paths.base.name}.pdf"
    pdf_path = paths.pdf / pdf_name
    if pdf_path.exists() and not check_overwrite(pdf_path): return

    lossless = args.source == "bw"
    # The combined PDF is always written: it is what the Makefile tracks as this
    # phase's output, and the whole scan in one file stays useful even when it
    # is split.
    build_pdf(pages, source_dir, pdf_path, args.jpeg_quality, lossless=lossless)

    docs = segment_documents(pages)
    if args.split_docs == "auto" and len(docs) > 1:
        log(f"Splitting into {len(docs)} documents...")
        stem = pdf_path.stem
        names = [doc_filename(stem, i, d) for i, d in enumerate(docs, 1)]
        for doc, name in zip(docs, names):
            log(f"  {doc['title'] or '(untitled)'}: "
                f"pages {doc['pages'][0]['page_num']}-{doc['pages'][-1]['page_num']}")
            build_pdf(doc["pages"], source_dir, paths.pdf / name,
                      args.jpeg_quality, lossless=lossless)
        write_manifest(paths, docs, args.source, names)
    elif len(docs) <= 1:
        # No boundaries left (all tags cleared in review): a manifest from an
        # earlier run would describe PDFs this project no longer splits into.
        (paths.json / "documents.json").unlink(missing_ok=True)

    log("PHASE 9 COMPLETE")

if __name__ == "__main__": main()
