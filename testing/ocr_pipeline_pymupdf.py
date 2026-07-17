"""
Fast OCR Pipeline using PyMuPDF + DocTR

Dirty OCR: PyMuPDF's simple text extraction (fastest, lowest level)
Parsed OCR: DocTR advanced OCR (high quality)

Both work directly on PDF - no image extraction needed!

Usage:
    python ocr_pipeline_pymupdf.py --pdf path/to/book.pdf --output-dir output/directory
"""

import argparse
from pathlib import Path
import json
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import pymupdf


class FastOCRPipeline:
    """Fast OCR pipeline using PyMuPDF + DocTR"""

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

        for dir_path in [self.dirty_ocr_dir, self.parsed_ocr_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"PDF: {self.pdf_path}")
        print(f"Output: {self.output_dir}")

    def generate_dirty_ocr(self):
        """Generate dirty OCR using PyMuPDF's simple text extraction"""
        print(f"\n{'='*60}")
        print("Step 1: Generating Dirty OCR (PyMuPDF text extraction)")
        print(f"{'='*60}")

        doc = pymupdf.open(self.pdf_path)
        total_pages = len(doc)
        print(f"Processing {total_pages} pages...")

        for page_num in range(total_pages):
            page = doc[page_num]

            # Extract text directly (simplest method)
            text = page.get_text()

            # Save text
            page_name = f"page_{page_num+1:04d}"
            txt_path = self.dirty_ocr_dir / f"{page_name}.txt"
            with open(txt_path, 'w') as f:
                f.write(text)

            # Save metadata
            json_path = self.dirty_ocr_dir / f"{page_name}.json"
            with open(json_path, 'w') as f:
                json.dump({
                    "page_num": page_num + 1,
                    "method": "pymupdf_get_text",
                    "char_count": len(text)
                }, f, indent=2)

            if (page_num + 1) % 50 == 0:
                print(f"  Processed {page_num + 1}/{total_pages} pages...")

        print(f"✓ Generated dirty OCR for {total_pages} pages")
        return total_pages

    def generate_parsed_ocr(self):
        """Generate parsed OCR using DocTR directly on PDF"""
        print(f"\n{'='*60}")
        print("Step 2: Generating Parsed OCR (DocTR) - Direct from PDF")
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

    def run(self):
        """Run the complete pipeline"""
        print(f"\n{'='*60}")
        print("FAST OCR PIPELINE (PyMuPDF + DocTR)")
        print(f"{'='*60}")

        # Step 1: PyMuPDF text extraction (super fast)
        self.generate_dirty_ocr()

        # Step 2: DocTR OCR (slower but high quality)
        self.generate_parsed_ocr()

        print(f"\n{'='*60}")
        print("✓ PIPELINE COMPLETE!")
        print(f"{'='*60}")
        print(f"\nResults saved to: {self.output_dir}")
        print(f"  - Dirty OCR (PyMuPDF): {self.dirty_ocr_dir}")
        print(f"  - Parsed OCR (DocTR): {self.parsed_ocr_dir}")
        print(f"\nNext: Extract MCQs with:")
        print(f"  python testing/extract_mcqs_only.py --ocr-results {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fast OCR pipeline using PyMuPDF + DocTR")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    args = parser.parse_args()

    pipeline = FastOCRPipeline(args.pdf, args.output_dir)
    pipeline.run()


if __name__ == "__main__":
    main()
