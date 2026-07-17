"""
Extract MCQs from OCR text without evaluation

This script:
1. Loads Dirty OCR and Parsed OCR text
2. Uses GPT-4 to extract MCQs (without correct answers)
3. Saves MCQs for later benchmarking when ground truth answers are provided

Usage:
    python extract_mcqs_only.py --ocr-results testing/az-full
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
from tqdm import tqdm


class MCQExtractor:
    """Extract MCQs from OCR text"""

    def __init__(self, ocr_results_dir: str, api_key: Optional[str] = None):
        """
        Initialize MCQ extractor

        Args:
            ocr_results_dir: Directory containing OCR results
            api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
        """
        self.results_dir = Path(ocr_results_dir)
        self.dirty_ocr_dir = self.results_dir / "dirty_ocr"
        self.parsed_ocr_dir = self.results_dir / "parsed_ocr"
        self.output_dir = self.results_dir / "mcq_extracted"
        self.output_dir.mkdir(exist_ok=True)

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )

        print(f"Initialized MCQ Extractor")
        print(f"  OCR Results: {self.results_dir}")
        print(f"  Output: {self.output_dir}")

    def load_all_text(self, source_dir: Path) -> str:
        """Load and concatenate all text files from a directory"""
        if not source_dir.exists():
            print(f"⚠️  Directory not found: {source_dir}")
            return ""

        text_files = sorted(source_dir.glob("*.txt"))
        all_text = ""
        for txt_file in text_files:
            content = txt_file.read_text()
            if content.strip():
                all_text += content + "\n\n"
        return all_text

    def extract_mcqs_with_llm(self, text: str, source_name: str) -> List[Dict]:
        """
        Extract MCQs from text using OpenAI GPT-4

        NOTE: Does NOT extract correct answers - just questions and choices
        Ground truth answers will be provided separately later

        Args:
            text: Source text to extract MCQs from
            source_name: Name of source (for logging)

        Returns:
            List of MCQ dictionaries with question and choices (no correct answer)
        """
        print(f"\nExtracting MCQs from {source_name} using OpenAI GPT-4...")

        if not text.strip():
            print(f"  ⚠️  No text found in {source_name}")
            return []

        prompt = f"""You are analyzing a real estate textbook to extract multiple-choice questions (MCQs).

Please extract ALL multiple-choice questions from the following text. For each MCQ, provide:
1. The question text
2. All answer choices (A, B, C, D, etc.)
3. Page/section reference if identifiable

DO NOT include the correct answer - we only need the questions and choices.

Return the results as a JSON array with this structure:
[
  {{
    "question_id": 1,
    "question": "Question text here?",
    "choices": {{
      "A": "First choice",
      "B": "Second choice",
      "C": "Third choice",
      "D": "Fourth choice"
    }},
    "source_page": "page number or chapter if identifiable"
  }},
  ...
]

Text to analyze (first 50,000 characters):

{text[:50000]}

Return ONLY the JSON array, no other text."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                max_tokens=16000,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            response_text = response.choices[0].message.content

            # Try to parse JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            mcqs = json.loads(response_text.strip())
            print(f"  ✓ Extracted {len(mcqs)} MCQs from {source_name}")
            return mcqs

        except Exception as e:
            print(f"  ✗ Error extracting MCQs: {str(e)}")
            return []

    def extract_all(self, max_mcqs: Optional[int] = None):
        """
        Extract MCQs from both Dirty and Parsed OCR

        Args:
            max_mcqs: Maximum number of MCQs to extract (for testing)
        """
        print(f"\n{'='*60}")
        print("EXTRACTING MCQs FROM OCR TEXT")
        print(f"{'='*60}")

        # Try parsed OCR first (better quality)
        print("\n[1/2] Extracting from Parsed OCR (DocTR)...")
        parsed_text = self.load_all_text(self.parsed_ocr_dir)

        if parsed_text.strip():
            mcqs = self.extract_mcqs_with_llm(parsed_text, "Parsed OCR")
        else:
            print("⚠️  Parsed OCR empty, trying Dirty OCR...")
            dirty_text = self.load_all_text(self.dirty_ocr_dir)
            mcqs = self.extract_mcqs_with_llm(dirty_text, "Dirty OCR")

        if not mcqs:
            print("\n✗ No MCQs found in OCR text!")
            return

        # Limit if requested
        if max_mcqs and len(mcqs) > max_mcqs:
            mcqs = mcqs[:max_mcqs]
            print(f"\nLimited to {max_mcqs} MCQs for testing")

        # Save extracted MCQs
        mcq_file = self.output_dir / "extracted_mcqs.json"
        with open(mcq_file, 'w') as f:
            json.dump({
                "total_mcqs": len(mcqs),
                "mcqs": mcqs,
                "note": "Correct answers not included - will be provided separately"
            }, f, indent=2)

        print(f"\n{'='*60}")
        print(f"✓ SUCCESS!")
        print(f"{'='*60}")
        print(f"Extracted {len(mcqs)} MCQs")
        print(f"Saved to: {mcq_file}")
        print(f"\nNext steps:")
        print(f"1. Review the extracted MCQs in {mcq_file}")
        print(f"2. Provide ground truth answers")
        print(f"3. Run benchmark with: python benchmark_with_answers.py")


def main():
    parser = argparse.ArgumentParser(description="Extract MCQs from OCR text")
    parser.add_argument("--ocr-results", required=True, help="Directory with OCR results")
    parser.add_argument("--api-key", help="OpenAI API key (or use OPENAI_API_KEY env var)")
    parser.add_argument("--max-mcqs", type=int, help="Maximum MCQs to extract (for testing)")

    args = parser.parse_args()

    # Initialize and extract
    extractor = MCQExtractor(args.ocr_results, args.api_key)
    extractor.extract_all(max_mcqs=args.max_mcqs)


if __name__ == "__main__":
    main()
