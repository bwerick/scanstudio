#!/usr/bin/env python3
"""
Unified MCQ Extraction Pipeline
Works on Mac, Legion, or any machine with GPU/CPU

Step 1: Extract clean OCR from PDF using DocTR
Step 2: Extract MCQs from clean OCR using GPT-4

Usage:
    export OPENAI_API_KEY="your-key"
    python run_mcq_pipeline.py --pdf "path/to/book.pdf" --output-dir "output/dir"

Options:
    --max-pages N     Process only first N pages (for testing)
    --max-mcqs N      Extract max N MCQs (for testing)
    --api-key KEY     OpenAI API key (or use OPENAI_API_KEY env var)

Examples:
    # Full book
    python run_mcq_pipeline.py \
        --pdf "output/az-principles-of-real-estate/Arizona Principles of Real Estate.pdf" \
        --output-dir "testing/az-results"

    # Test on 100 pages
    python run_mcq_pipeline.py \
        --pdf "output/az-principles-of-real-estate/Arizona Principles of Real Estate.pdf" \
        --output-dir "testing/az-test" \
        --max-pages 100 \
        --max-mcqs 50
"""

import os
import sys
import argparse
from pathlib import Path
import json
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from openai import OpenAI


class MCQPipeline:
    """Complete pipeline for OCR + MCQ extraction"""

    def __init__(self, pdf_path: str, output_dir: str, api_key: str = None):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.ocr_dir = self.output_dir / "clean_ocr"
        self.mcq_dir = self.output_dir / "mcqs"

        # Create directories
        self.ocr_dir.mkdir(parents=True, exist_ok=True)
        self.mcq_dir.mkdir(parents=True, exist_ok=True)

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

        print(f"PDF: {self.pdf_path}")
        print(f"Output: {self.output_dir}")

    def step1_extract_clean_ocr(self, max_pages: int = None):
        """Step 1: Extract clean OCR using DocTR"""
        print(f"\n{'='*60}")
        print("STEP 1: Extract Clean OCR (DocTR)")
        print(f"{'='*60}")

        # Load PDF
        print("Loading PDF...")
        doc = DocumentFile.from_pdf(str(self.pdf_path))

        # Limit pages if specified
        if max_pages:
            doc = doc[:max_pages]
            print(f"Processing first {max_pages} pages...")
        else:
            print(f"Processing all {len(doc)} pages...")

        # Initialize DocTR
        model = ocr_predictor(pretrained=True)

        # Process PDF
        result = model(doc)

        # Save each page
        for page_idx, page in enumerate(result.pages):
            page_name = f"page_{page_idx+1:04d}"

            # Save text
            txt_path = self.ocr_dir / f"{page_name}.txt"
            txt_path.write_text(page.render())

            # Save structured JSON
            json_path = self.ocr_dir / f"{page_name}.json"
            with open(json_path, 'w') as f:
                json.dump(page.export(), f, indent=2)

            if (page_idx + 1) % 10 == 0:
                print(f"  Processed {page_idx + 1}/{len(result.pages)} pages...")

        print(f"✓ Clean OCR complete: {len(result.pages)} pages")
        return len(result.pages)

    def step2_extract_mcqs(self, max_mcqs: int = None):
        """Step 2: Extract MCQs from clean OCR using GPT-4"""
        print(f"\n{'='*60}")
        print("STEP 2: Extract MCQs from Clean OCR")
        print(f"{'='*60}")

        # Load all OCR text
        text_files = sorted(self.ocr_dir.glob("*.txt"))
        all_text = ""
        for txt_file in text_files:
            content = txt_file.read_text()
            if content.strip():
                all_text += f"\n\n=== {txt_file.stem} ===\n{content}"

        if not all_text.strip():
            print("✗ No OCR text found!")
            return

        print(f"Total text length: {len(all_text)} characters")

        # Extract MCQs using GPT-4
        print("Extracting MCQs using GPT-4...")

        prompt = f"""You are extracting multiple-choice questions from a real estate textbook OCR text.

The text contains MCQ sections usually titled "REVIEWING YOUR UNDERSTANDING" or "CHECKING YOUR COMPREHENSION".

Extract ALL multiple-choice questions. For each MCQ:
1. Extract the exact question text
2. Extract all answer choices (A, B, C, D)
3. DO NOT include the correct answer (we'll get that separately)
4. Note which page it came from if possible

IMPORTANT: Only extract ACTUAL questions from the text - do NOT make up or generate questions.

Return a JSON array:
[
  {{
    "question_id": 1,
    "question": "Exact question text from book",
    "choices": {{
      "A": "First choice",
      "B": "Second choice",
      "C": "Third choice",
      "D": "Fourth choice"
    }},
    "source_page": "page_0036"
  }}
]

Here is the OCR text:

{all_text[:100000]}

Return ONLY the JSON array."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                max_tokens=16000,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            response_text = response.choices[0].message.content

            # Parse JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            mcqs = json.loads(response_text.strip())

            # Limit if requested
            if max_mcqs and len(mcqs) > max_mcqs:
                mcqs = mcqs[:max_mcqs]

            print(f"✓ Extracted {len(mcqs)} MCQs")

            # Save results
            output_file = self.mcq_dir / "extracted_mcqs.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "total_mcqs": len(mcqs),
                    "mcqs": mcqs,
                    "note": "Correct answers not included"
                }, f, indent=2)

            print(f"✓ Saved to: {output_file}")

            # Preview
            print("\nFirst 3 MCQs:")
            for mcq in mcqs[:3]:
                print(f"\n{mcq['question_id']}. {mcq['question']}")
                for letter, choice in mcq['choices'].items():
                    print(f"   {letter}. {choice}")

            return mcqs

        except Exception as e:
            print(f"✗ Error: {e}")
            return None

    def run(self, max_pages: int = None, max_mcqs: int = None):
        """Run complete pipeline"""
        print(f"\n{'='*60}")
        print("MCQ EXTRACTION PIPELINE")
        print(f"{'='*60}")

        # Step 1: OCR
        total_pages = self.step1_extract_clean_ocr(max_pages)

        # Step 2: MCQ Extraction
        mcqs = self.step2_extract_mcqs(max_mcqs)

        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE!")
        print(f"{'='*60}")
        print(f"Clean OCR: {self.ocr_dir}")
        print(f"MCQs: {self.mcq_dir}")


def main():
    parser = argparse.ArgumentParser(description="Complete MCQ extraction pipeline")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--max-pages", type=int, help="Max pages to process (optional)")
    parser.add_argument("--max-mcqs", type=int, help="Max MCQs to extract (optional)")
    parser.add_argument("--api-key", help="OpenAI API key (or use OPENAI_API_KEY env var)")

    args = parser.parse_args()

    # Check API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OpenAI API key not found!")
        print("Set OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)

    # Run pipeline
    pipeline = MCQPipeline(args.pdf, args.output_dir, api_key)
    pipeline.run(max_pages=args.max_pages, max_mcqs=args.max_mcqs)


if __name__ == "__main__":
    main()
