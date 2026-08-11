"""
PDF assembly.

Builds a PDF from a list of page images using ReportLab.
"""

import cv2
import tempfile
import os
from pathlib import Path


def build_pdf(pages: list[dict], source_dir: Path, output_path: Path,
               jpeg_quality: int = 90, on_progress=None):
    """
    Assemble page images into a PDF.

    Args:
        pages: list of page metadata dicts with "filename" key
        source_dir: directory containing the page images
        output_path: where to save the PDF
        jpeg_quality: JPEG quality for images embedded in PDF
        on_progress: callback(current, total)
    """
    from reportlab.pdfgen import canvas as rl_canvas

    if not pages:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = None
    count = 0

    with tempfile.TemporaryDirectory() as tmp:
        for i, pg in enumerate(pages):
            ip = source_dir / pg["filename"]
            if not ip.exists():
                continue
            img = cv2.imread(str(ip))
            if img is None:
                continue

            ih, iw = img.shape[:2]
            pw = 595  # A4 width in points
            ph = pw * ih / iw

            if c is None:
                c = rl_canvas.Canvas(str(output_path), pagesize=(pw, ph))

            c._pagesize = (pw, ph)
            tp = os.path.join(tmp, f"p_{i:04d}.jpg")
            cv2.imwrite(tp, img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            c.drawImage(tp, 0, 0, width=pw, height=ph)
            c.showPage()
            count += 1

            if on_progress:
                on_progress(i + 1, len(pages))

    if c:
        c.save()

    return count


def build_binarized_pdf(pages: list[dict], source_dir: Path,
                         output_path: Path, block_size: int = 51,
                         offset: int = 10, jpeg_quality: int = 90,
                         on_progress=None):
    """Build a B&W PDF by binarizing each page on the fly."""
    from reportlab.pdfgen import canvas as rl_canvas

    if not pages:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = None
    count = 0

    if block_size % 2 == 0:
        block_size += 1

    with tempfile.TemporaryDirectory() as tmp:
        for i, pg in enumerate(pages):
            ip = source_dir / pg["filename"]
            if not ip.exists():
                continue
            img = cv2.imread(str(ip))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, offset
            )

            ih, iw = binary.shape[:2]
            pw = 595
            ph = pw * ih / iw

            if c is None:
                c = rl_canvas.Canvas(str(output_path), pagesize=(pw, ph))

            c._pagesize = (pw, ph)
            tp = os.path.join(tmp, f"p_{i:04d}.jpg")
            cv2.imwrite(tp, binary, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            c.drawImage(tp, 0, 0, width=pw, height=ph)
            c.showPage()
            count += 1

            if on_progress:
                on_progress(i + 1, len(pages))

    if c:
        c.save()

    return count
