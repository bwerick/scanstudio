#!/usr/bin/env python3
"""Phase 9: Build PDF
Usage: python scripts/p9_build_pdf.py output/mybook
       python scripts/p9_build_pdf.py output/mybook --source bw"""

import argparse, json, sys, time, tempfile, os
from pathlib import Path
import cv2
from PIL import Image
from utils import log, ProjectPaths, ensure_dir, check_overwrite

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--source", default="pages", choices=["pages", "bw"])
    parser.add_argument("--pdf-name", default=None,
                        help="Output filename (default: <project>.pdf)")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()

    log("=" * 60); log("PHASE 9: Build PDF"); log("=" * 60)
    paths = ProjectPaths(args.output_dir)
    ensure_dir(paths.pdf)
    source_dir = paths.pages if args.source == "pages" else (paths.base / "bw")
    pages = json.loads((paths.json / "pages.json").read_text())
    pdf_name = args.pdf_name or f"{paths.base.name}.pdf"
    pdf_path = paths.pdf / pdf_name
    if pdf_path.exists() and not check_overwrite(pdf_path): return

    build_pdf(pages, source_dir, pdf_path, args.jpeg_quality, lossless=args.source == "bw")
    log("PHASE 9 COMPLETE")

if __name__ == "__main__": main()
