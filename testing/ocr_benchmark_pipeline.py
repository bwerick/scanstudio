"""
OCR Quality Benchmark Pipeline

For each textbook, generates:
1. Dirty OCR (raw EasyOCR output)
2. Parsed OCR (DocTR cleaned output)
3. Ground Truth (PDF text extraction)

Then benchmarks MCQ extraction accuracy across 4 settings:
- Setting A: No textbook available (baseline)
- Setting B: Dirty OCR (raw)
- Setting C: Parsed OCR (DocTR)
- Setting D: Ground Truth PDF text (upper bound)

Usage:
    python testing/ocr_benchmark_pipeline.py --pdf "path/to/textbook.pdf" --output-dir "testing/results"
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import time

import numpy as np
from PIL import Image
import easyocr
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch
import platform

# For PDF processing
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("Warning: PyMuPDF not installed. Install with: pip install pymupdf")


class OCRBenchmarkPipeline:
    """Pipeline for benchmarking OCR quality on textbook MCQ extraction"""

    def __init__(self, pdf_path: str, output_dir: str):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.images_dir = self.output_dir / "images"
        self.dirty_ocr_dir = self.output_dir / "dirty_ocr"
        self.parsed_ocr_dir = self.output_dir / "parsed_ocr"
        self.ground_truth_dir = self.output_dir / "ground_truth"
        self.results_dir = self.output_dir / "results"

        for dir_path in [self.images_dir, self.dirty_ocr_dir,
                         self.parsed_ocr_dir, self.ground_truth_dir, self.results_dir]:
            dir_path.mkdir(exist_ok=True)

        # Determine device
        if platform.system() == "Darwin":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {self.device}")

        # Initialize OCR models (lazy loading)
        self.easyocr_reader = None
        self.doctr_model = None

    def _init_easyocr(self):
        """Lazy initialization of EasyOCR"""
        if self.easyocr_reader is None:
            print("Initializing EasyOCR...")
            self.easyocr_reader = easyocr.Reader(["en"], gpu=self.device == "cuda")

    def _init_doctr(self):
        """Lazy initialization of DocTR"""
        if self.doctr_model is None:
            print("Initializing DocTR...")
            self.doctr_model = ocr_predictor(
                det_arch="db_resnet50",
                reco_arch="crnn_vgg16_bn",
                pretrained=True
            ).to(self.device)

    def extract_pdf_pages(self, max_pages: int = None) -> int:
        """
        Extract pages from PDF as images

        Args:
            max_pages: Maximum number of pages to extract (None = all)

        Returns:
            Number of pages extracted
        """
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF required for PDF extraction. Install: pip install pymupdf")

        print(f"\n{'='*60}")
        print(f"STEP 1: Extracting pages from PDF")
        print(f"{'='*60}")

        doc = fitz.open(self.pdf_path)
        total_pages = len(doc)
        pages_to_extract = min(max_pages, total_pages) if max_pages else total_pages

        print(f"Total pages in PDF: {total_pages}")
        print(f"Extracting: {pages_to_extract} pages")

        for page_num in range(pages_to_extract):
            page = doc[page_num]

            # Extract as image (300 DPI for quality)
            mat = fitz.Matrix(300/72, 300/72)
            pix = page.get_pixmap(matrix=mat)

            # Save image
            img_path = self.images_dir / f"page_{page_num+1:04d}.png"
            pix.save(img_path)

            # Extract ground truth text
            text = page.get_text()
            gt_path = self.ground_truth_dir / f"page_{page_num+1:04d}.txt"
            gt_path.write_text(text)

            if (page_num + 1) % 10 == 0:
                print(f"  Processed {page_num + 1}/{pages_to_extract} pages...")

        doc.close()
        print(f"✓ Extracted {pages_to_extract} pages")
        return pages_to_extract

    def generate_dirty_ocr(self, page_nums: List[int] = None):
        """
        Generate dirty OCR using EasyOCR (raw output)

        Args:
            page_nums: List of page numbers to process (None = all)
        """
        print(f"\n{'='*60}")
        print(f"STEP 2: Generating Dirty OCR (EasyOCR)")
        print(f"{'='*60}")

        self._init_easyocr()

        image_files = sorted(self.images_dir.glob("*.png"))
        if page_nums:
            image_files = [f for i, f in enumerate(image_files) if i in page_nums]

        print(f"Processing {len(image_files)} pages...")

        for idx, img_path in enumerate(image_files):
            page_name = img_path.stem

            # Run EasyOCR
            result = self.easyocr_reader.readtext(str(img_path))

            # Save raw output with confidence scores
            output = []
            for bbox, text, prob in result:
                output.append({
                    "text": text,
                    "confidence": float(prob),
                    "bbox": [[float(x), float(y)] for x, y in bbox]
                })

            # Save as JSON (preserves structure)
            json_path = self.dirty_ocr_dir / f"{page_name}.json"
            with open(json_path, 'w') as f:
                json.dump(output, f, indent=2)

            # Also save as plain text
            txt_path = self.dirty_ocr_dir / f"{page_name}.txt"
            with open(txt_path, 'w') as f:
                for bbox, text, prob in result:
                    f.write(text + "\n")

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(image_files)} pages...")

        print(f"✓ Generated dirty OCR for {len(image_files)} pages")

    def generate_parsed_ocr(self, page_nums: List[int] = None):
        """
        Generate parsed OCR using DocTR (advanced output)

        Args:
            page_nums: List of page numbers to process (None = all)
        """
        print(f"\n{'='*60}")
        print(f"STEP 3: Generating Parsed OCR (DocTR)")
        print(f"{'='*60}")

        self._init_doctr()

        image_files = sorted(self.images_dir.glob("*.png"))
        if page_nums:
            image_files = [f for i, f in enumerate(image_files) if i in page_nums]

        print(f"Processing {len(image_files)} pages...")

        for idx, img_path in enumerate(image_files):
            page_name = img_path.stem

            # Run DocTR
            doc = DocumentFile.from_images(str(img_path))
            result = self.doctr_model(doc)

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

    def extract_mcqs(self, source_dir: Path, setting_name: str) -> List[Dict]:
        """
        Extract MCQs from text files in a directory

        Args:
            source_dir: Directory containing text files
            setting_name: Name of the setting (for logging)

        Returns:
            List of extracted MCQs
        """
        print(f"\n  Extracting MCQs from {setting_name}...")

        # Read all text files
        text_files = sorted(source_dir.glob("*.txt"))
        all_text = ""
        for txt_file in text_files:
            all_text += txt_file.read_text() + "\n\n"

        # Simple MCQ pattern matching
        # This is a basic implementation - can be enhanced with NLP
        mcqs = []

        # Pattern: Look for numbered questions followed by lettered choices
        lines = all_text.split('\n')
        current_question = None
        current_choices = []

        for line in lines:
            line = line.strip()

            # Check if line looks like a question
            if any(line.startswith(f"{i}.") or line.startswith(f"{i})")
                   for i in range(1, 200)):
                # Save previous question if exists
                if current_question and current_choices:
                    mcqs.append({
                        "question": current_question,
                        "choices": current_choices
                    })

                # Start new question
                current_question = line
                current_choices = []

            # Check if line looks like a choice (A. B. C. D. or a. b. c. d.)
            elif line and len(line) > 2 and (
                (line[0] in 'ABCDEabcde' and line[1] in '.)')
            ):
                current_choices.append(line)

        # Add last question
        if current_question and current_choices:
            mcqs.append({
                "question": current_question,
                "choices": current_choices
            })

        print(f"    Found {len(mcqs)} MCQs")
        return mcqs

    def benchmark_settings(self) -> Dict:
        """
        Benchmark MCQ extraction across all 4 settings

        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'='*60}")
        print(f"STEP 4: Benchmarking MCQ Extraction")
        print(f"{'='*60}")

        results = {}

        # Setting A: No textbook (baseline)
        print("\nSetting A: No textbook available")
        results['setting_a'] = {
            "name": "No Textbook",
            "mcqs_extracted": 0,
            "description": "Baseline - no source material"
        }

        # Setting B: Dirty OCR
        print("\nSetting B: Dirty OCR (EasyOCR)")
        mcqs_b = self.extract_mcqs(self.dirty_ocr_dir, "Dirty OCR")
        results['setting_b'] = {
            "name": "Dirty OCR (EasyOCR)",
            "mcqs_extracted": len(mcqs_b),
            "mcqs": mcqs_b[:5],  # Sample
            "description": "Raw EasyOCR output"
        }

        # Setting C: Parsed OCR
        print("\nSetting C: Parsed OCR (DocTR)")
        mcqs_c = self.extract_mcqs(self.parsed_ocr_dir, "Parsed OCR")
        results['setting_c'] = {
            "name": "Parsed OCR (DocTR)",
            "mcqs_extracted": len(mcqs_c),
            "mcqs": mcqs_c[:5],  # Sample
            "description": "DocTR advanced OCR"
        }

        # Setting D: Ground Truth
        print("\nSetting D: Ground Truth (PDF text)")
        mcqs_d = self.extract_mcqs(self.ground_truth_dir, "Ground Truth")
        results['setting_d'] = {
            "name": "Ground Truth PDF",
            "mcqs_extracted": len(mcqs_d),
            "mcqs": mcqs_d[:5],  # Sample
            "description": "Direct PDF text extraction"
        }

        # Save results
        results_path = self.results_dir / "benchmark_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Benchmark complete. Results saved to {results_path}")
        return results

    def run_full_pipeline(self, max_pages: int = None):
        """Run the complete benchmark pipeline"""
        start_time = time.time()

        print(f"\n{'#'*60}")
        print(f"OCR QUALITY BENCHMARK PIPELINE")
        print(f"{'#'*60}")
        print(f"PDF: {self.pdf_path}")
        print(f"Output: {self.output_dir}")

        # Step 1: Extract PDF pages
        num_pages = self.extract_pdf_pages(max_pages)

        # Step 2: Generate Dirty OCR
        self.generate_dirty_ocr()

        # Step 3: Generate Parsed OCR
        self.generate_parsed_ocr()

        # Step 4: Benchmark settings
        results = self.benchmark_settings()

        # Print summary
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Pages processed: {num_pages}")
        print(f"\nMCQ Extraction Results:")
        print(f"  Setting A (No Textbook):    {results['setting_a']['mcqs_extracted']} MCQs")
        print(f"  Setting B (Dirty OCR):      {results['setting_b']['mcqs_extracted']} MCQs")
        print(f"  Setting C (Parsed OCR):     {results['setting_c']['mcqs_extracted']} MCQs")
        print(f"  Setting D (Ground Truth):   {results['setting_d']['mcqs_extracted']} MCQs")
        print(f"\nTime elapsed: {elapsed:.2f} seconds")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="OCR Quality Benchmark Pipeline")
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to the PDF textbook"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="testing/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to process (default: all)"
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = OCRBenchmarkPipeline(args.pdf, args.output_dir)
    pipeline.run_full_pipeline(max_pages=args.max_pages)


if __name__ == "__main__":
    main()
