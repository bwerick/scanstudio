#!/usr/bin/env python3
"""Phase 9: Build PDF
Usage: python scripts/p9_build_pdf.py output/mybook
       python scripts/p9_build_pdf.py output/mybook --source bw"""

import argparse, json, sys, time, tempfile, os
from pathlib import Path
import cv2
from utils import log, ProjectPaths, ensure_dir, check_overwrite

def build_pdf(pages, source_dir, output_path, quality):
    from reportlab.pdfgen import canvas as rl_canvas
    if not pages: return
    c = None; count = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, pg in enumerate(pages):
            ip = source_dir / pg["filename"]
            if not ip.exists(): continue
            img = cv2.imread(str(ip))
            if img is None: continue
            ih, iw = img.shape[:2]
            pw = 595; ph = pw * ih / iw
            if c is None: c = rl_canvas.Canvas(str(output_path), pagesize=(pw, ph))
            c._pagesize = (pw, ph)
            tp = os.path.join(tmp, f"p_{i:04d}.jpg")
            cv2.imwrite(tp, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            c.drawImage(tp, 0, 0, width=pw, height=ph)
            c.showPage(); count += 1
    if c: c.save()
    sz = output_path.stat().st_size / (1024*1024)
    log(f"  {output_path} ({sz:.1f} MB, {count} pages)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--source", default="pages", choices=["pages", "bw"])
    parser.add_argument("--pdf-name", default="book.pdf")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()

    log("=" * 60); log("PHASE 9: Build PDF"); log("=" * 60)
    paths = ProjectPaths(args.output_dir)
    ensure_dir(paths.pdf)
    source_dir = paths.pages if args.source == "pages" else (paths.base / "bw")
    pages = json.loads((paths.json / "pages.json").read_text())
    pdf_path = paths.pdf / args.pdf_name
    if pdf_path.exists() and not check_overwrite(pdf_path): return

    build_pdf(pages, source_dir, pdf_path, args.jpeg_quality)
    log("PHASE 9 COMPLETE")

if __name__ == "__main__": main()
