#!/usr/bin/env python3
"""
Phase 9: Build PDF

Assembles individual page images into a PDF document.

Usage:
  python scripts/p9_build_pdf.py output/audiq5
  python scripts/p9_build_pdf.py output/audiq5 --source bw
  python scripts/p9_build_pdf.py output/audiq5 --pdf-name "Audi_Q5_Manual.pdf"

Inputs:
  - output/<n>/pages/*.jpg    (default) or
  - output/<n>/bw/*.jpg       (with --source bw)
  - output/<n>/pages/pages.json   Page ordering

Outputs (in output/<n>/pdf/):
  - book.pdf (or custom name)

Requirements:
  pip install opencv-python reportlab
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from utils import log, ensure_dir, check_overwrite


def build_pdf(
    pages: list[dict], source_dir: Path, output_path: Path, jpeg_quality: int
):
    """Assemble page images into a PDF."""
    from reportlab.pdfgen import canvas as rl_canvas
    import tempfile
    import os

    if not pages:
        log("  No pages to build PDF from!")
        return

    c = None
    page_count = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, pg in enumerate(pages):
            img_path = source_dir / pg["filename"]
            if not img_path.exists():
                log(f"  WARNING: {pg['filename']} not found, skipping")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                log(f"  WARNING: Cannot read {pg['filename']}, skipping")
                continue

            ih, iw = img.shape[:2]

            # PDF page sized to match image aspect ratio
            # Use portrait A4 width as reference (595 points)
            page_width = 595
            page_height = page_width * ih / iw

            if c is None:
                c = rl_canvas.Canvas(
                    str(output_path), pagesize=(page_width, page_height)
                )

            c._pagesize = (page_width, page_height)

            tmp = os.path.join(tmpdir, f"p_{i:04d}.jpg")
            cv2.imwrite(tmp, img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            c.drawImage(tmp, 0, 0, width=page_width, height=page_height)
            c.showPage()
            page_count += 1

            if page_count % 50 == 0:
                log(f"  Page {page_count}...")

    if c is not None:
        c.save()
        sz = output_path.stat().st_size / (1024 * 1024)
        log(f"  PDF saved: {output_path} ({sz:.1f} MB, {page_count} pages)")
    else:
        log("  ERROR: No pages were processed")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 9: Build PDF from page images",
    )
    parser.add_argument("output_dir", help="Base output directory (e.g. output/audiq5)")
    parser.add_argument(
        "--source",
        type=str,
        default="pages",
        choices=["pages", "bw"],
        help="Source folder for images (default: pages)",
    )
    parser.add_argument(
        "--pdf-name",
        type=str,
        default="book.pdf",
        help="PDF output filename (default: book.pdf)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality for images in PDF (default: 90)",
    )
    args = parser.parse_args()

    log("=" * 60)
    log("PHASE 9: Build PDF")
    log("=" * 60)

    base = Path(args.output_dir)
    source_dir = base / args.source
    pages_json = base / "pages" / "pages.json"
    pdf_dir = ensure_dir(base / "pdf")

    # Load page ordering (always from pages/pages.json)
    if not pages_json.exists():
        log(f"ERROR: {pages_json} not found. Run Phase 6 first.")
        sys.exit(1)

    if not source_dir.exists():
        log(f"ERROR: Source directory {source_dir} not found.")
        if args.source == "bw":
            log("Run Phase 8 first: python scripts/p8_binarize.py")
        sys.exit(1)

    pages = json.loads(pages_json.read_text())
    log(f"Source: {source_dir}/ ({len(pages)} pages)")

    # Check overwrite
    pdf_path = pdf_dir / args.pdf_name
    if pdf_path.exists():
        if not check_overwrite(pdf_path):
            log("Skipped.")
            return

    # Build
    log("")
    t0 = time.time()
    build_pdf(pages, source_dir, pdf_path, args.jpeg_quality)
    elapsed = time.time() - t0

    log("")
    log("=" * 60)
    log("PHASE 9 COMPLETE")
    log(f"  PDF: {pdf_path}")
    log(f"  Time: {elapsed:.1f}s")
    log("=" * 60)


if __name__ == "__main__":
    main()
