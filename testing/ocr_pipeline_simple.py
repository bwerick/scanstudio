"""
Simple OCR Pipeline - Generates Dirty and Parsed OCR only

For each page:
1. Extract page as image from PDF
2. Run Dirty OCR (raw EasyOCR)
3. Run Parsed OCR (DocTR cleaned)

Skips ground truth PDF text extraction.

Usage:
    python ocr_pipeline_simple.py --pdf path/to/book.pdf --output-dir output/directory
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


class SimpleOCRPipeline:
    """Simple OCR pipeline for dirty and parsed OCR"""

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
        self.images_dir = self.output_dir / "images"
        self.dirty_ocr_dir = self.output_dir / "dirty_ocr"
        self.parsed_ocr_dir = self.output_dir / "parsed_ocr"

        for dir_path in [self.images_dir, self.dirty_ocr_dir, self.parsed_ocr_dir]:
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

    def extract_pages_as_images(self):
        """Extract all pages from PDF as PNG images"""
        print(f"\n{'='*60}")
        print("Step 1: Extracting pages as images")
        print(f"{'='*60}")

        doc = pymupdf.open(self.pdf_path)
        total_pages = len(doc)
        print(f"Total pages: {total_pages}")

        for page_num in range(total_pages):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)

            # Save as PNG
            img_path = self.images_dir / f"page_{page_num+1:04d}.png"
            pix.save(str(img_path))

            if (page_num + 1) % 50 == 0:
                print(f"  Extracted {page_num + 1}/{total_pages} pages...")

        print(f"✓ Extracted {total_pages} pages as images")

    def generate_dirty_ocr(self):
        """Generate dirty OCR using EasyOCR"""
        print(f"\n{'='*60}")
        print("Step 2: Generating Dirty OCR (EasyOCR)")
        print(f"{'='*60}")

        reader = easyocr.Reader(['en'], gpu=(self.device != 'cpu'))

        image_files = sorted(self.images_dir.glob("*.png"))
        print(f"Processing {len(image_files)} images...")

        for idx, img_path in enumerate(image_files):
            page_name = img_path.stem

            # Run EasyOCR
            result = reader.readtext(str(img_path))

            # Extract text
            text = "\n".join([item[1] for item in result])

            # Save text
            txt_path = self.dirty_ocr_dir / f"{page_name}.txt"
            with open(txt_path, 'w') as f:
                f.write(text)

            # Save structured result
            json_path = self.dirty_ocr_dir / f"{page_name}.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(image_files)} pages...")

        print(f"✓ Generated dirty OCR for {len(image_files)} pages")

    def generate_parsed_ocr(self):
        """Generate parsed OCR using DocTR"""
        print(f"\n{'='*60}")
        print("Step 3: Generating Parsed OCR (DocTR)")
        print(f"{'='*60}")

        # Initialize DocTR model
        model = ocr_predictor(pretrained=True)

        image_files = sorted(self.images_dir.glob("*.png"))
        print(f"Processing {len(image_files)} images...")

        for idx, img_path in enumerate(image_files):
            page_name = img_path.stem

            # Run DocTR
            doc = DocumentFile.from_images(str(img_path))
            result = model(doc)

            # Save rendered text
            txt_path = self.parsed_ocr_dir / f"{page_name}.txt"
            with open(txt_path, 'w') as f:
                f.write(result.render())

            # Save structured output
            json_path = self.parsed_ocr_dir / f"{page_name}.json"
            result_dict = result.export()
            with open(json_path, 'w') as f:
                json.dump(result_dict, f, indent=2)

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(image_files)} pages...")

        print(f"✓ Generated parsed OCR for {len(image_files)} pages")

    def run(self):
        """Run the complete pipeline"""
        print(f"\n{'='*60}")
        print("SIMPLE OCR PIPELINE")
        print(f"{'='*60}")

        self.extract_pages_as_images()
        self.generate_dirty_ocr()
        self.generate_parsed_ocr()

        print(f"\n{'='*60}")
        print("✓ PIPELINE COMPLETE!")
        print(f"{'='*60}")
        print(f"\nResults saved to: {self.output_dir}")
        print(f"  - Images: {self.images_dir}")
        print(f"  - Dirty OCR: {self.dirty_ocr_dir}")
        print(f"  - Parsed OCR: {self.parsed_ocr_dir}")
        print(f"\nNext: Extract MCQs with:")
        print(f"  python testing/extract_mcqs_only.py --ocr-results {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Simple OCR pipeline")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    args = parser.parse_args()

    pipeline = SimpleOCRPipeline(args.pdf, args.output_dir)
    pipeline.run()


if __name__ == "__main__":
    main()
