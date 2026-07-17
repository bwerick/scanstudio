"""
Direct OCR Pipeline - Works directly on PDF without image extraction

Uses DocTR's built-in PDF support to process pages directly.
Falls back to image extraction only for EasyOCR (dirty OCR).

Usage:
    python ocr_pipeline_direct.py --pdf path/to/book.pdf --output-dir output/directory
"""

import os
import argparse
from pathlib import Path
import json
import easyocr
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import pymupdf
import torch


class DirectOCRPipeline:
    """OCR pipeline that works directly on PDF"""

    def __init__(self, pdf_path: str, output_dir: str):
        """
        Initialize OCR pipeline

        Args:
            pdf_path: Path to input PDF
            output_dir: Directory to save results
        """
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)

        # Create output directories
        self.dirty_ocr_dir = self.output_dir / "dirty_ocr"
        self.parsed_ocr_dir = self.output_dir / "parsed_ocr"
        self.temp_images_dir = self.output_dir / "temp_images"  # Only for EasyOCR

        for dir_path in [self.dirty_ocr_dir, self.parsed_ocr_dir, self.temp_images_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Select device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Using device: {self.device}")
        print(f"PDF: {self.pdf_path}")
        print(f"Output: {self.output_dir}")

    def generate_parsed_ocr_direct(self):
        """Generate parsed OCR using DocTR directly on PDF"""
        print(f"\n{'='*60}")
        print("Step 1: Generating Parsed OCR (DocTR) - Direct from PDF")
        print(f"{'='*60}")

        # Initialize DocTR model
        model = ocr_predictor(pretrained=True)

        # Process PDF directly
        print(f"Loading PDF...")
        doc = DocumentFile.from_pdf(str(self.pdf_path))

        print(f"Processing {len(doc)} pages...")
        result = model(doc)

        # Save results for each page
        for page_idx, page in enumerate(result.pages):
            page_name = f"page_{page_idx+1:04d}"

            # Save rendered text
            txt_path = self.parsed_ocr_dir / f"{page_name}.txt"
            page_text = page.render()
            with open(txt_path, 'w') as f:
                f.write(page_text)

            # Save structured output
            json_path = self.parsed_ocr_dir / f"{page_name}.json"
            page_dict = page.export()
            with open(json_path, 'w') as f:
                json.dump(page_dict, f, indent=2)

            if (page_idx + 1) % 10 == 0:
                print(f"  Processed {page_idx + 1}/{len(doc)} pages...")

        print(f"✓ Generated parsed OCR for {len(doc)} pages")
        return len(doc)

    def generate_dirty_ocr(self, total_pages: int):
        """
        Generate dirty OCR using EasyOCR
        Note: EasyOCR needs images, so we extract pages temporarily
        """
        print(f"\n{'='*60}")
        print("Step 2: Generating Dirty OCR (EasyOCR)")
        print(f"{'='*60}")
        print("Note: EasyOCR requires images, extracting pages temporarily...")

        # Extract pages as images (only for EasyOCR)
        doc = pymupdf.open(self.pdf_path)

        reader = easyocr.Reader(['en'], gpu=(self.device != 'cpu'))

        for page_num in range(total_pages):
            page = doc[page_num]

            # Extract page as image temporarily
            pix = page.get_pixmap(dpi=200)
            temp_img = self.temp_images_dir / f"temp_page_{page_num+1}.png"
            pix.save(str(temp_img))

            # Run EasyOCR
            result = reader.readtext(str(temp_img))

            # Extract text
            text = "\n".join([item[1] for item in result])

            # Save text
            page_name = f"page_{page_num+1:04d}"
            txt_path = self.dirty_ocr_dir / f"{page_name}.txt"
            with open(txt_path, 'w') as f:
                f.write(text)

            # Save structured result
            json_path = self.dirty_ocr_dir / f"{page_name}.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)

            # Clean up temp image
            temp_img.unlink()

            if (page_num + 1) % 10 == 0:
                print(f"  Processed {page_num + 1}/{total_pages} pages...")

        # Clean up temp directory
        self.temp_images_dir.rmdir()

        print(f"✓ Generated dirty OCR for {total_pages} pages")

    def run(self):
        """Run the complete pipeline"""
        print(f"\n{'='*60}")
        print("DIRECT OCR PIPELINE (No permanent image extraction)")
        print(f"{'='*60}")

        # Process with DocTR first (faster, works directly on PDF)
        total_pages = self.generate_parsed_ocr_direct()

        # Process with EasyOCR (needs temporary images)
        self.generate_dirty_ocr(total_pages)

        print(f"\n{'='*60}")
        print("✓ PIPELINE COMPLETE!")
        print(f"{'='*60}")
        print(f"\nResults saved to: {self.output_dir}")
        print(f"  - Dirty OCR: {self.dirty_ocr_dir}")
        print(f"  - Parsed OCR: {self.parsed_ocr_dir}")
        print(f"\nNext: Extract MCQs with:")
        print(f"  python testing/extract_mcqs_only.py --ocr-results {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Direct OCR pipeline (no image extraction)")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    args = parser.parse_args()

    pipeline = DirectOCRPipeline(args.pdf, args.output_dir)
    pipeline.run()


if __name__ == "__main__":
    main()
