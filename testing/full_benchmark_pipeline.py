#!/usr/bin/env python3
"""
Exercise Benchmark Pipeline - OCR Quality Impact

Tests whether OCR-digitizing books improves an LLM's ability to answer exercises.

3 Conditions:
- Baseline: LLM answers with no context (general knowledge only)
- Raw OCR: LLM answers with raw embedded PDF text as context
- Corrected OCR: LLM answers with OLMoCR-corrected text as context

Pipeline:
1. Extract Raw OCR (PyMuPDF embedded text)
2. Correct OCR using OLMoCR
3. Extract exercises from book
4. Add reference answers (from user)
5. Benchmark: Answer exercises with each condition
6. Generate comparison report with LLM evaluation

Usage:
    export OPENAI_API_KEY="your-key"

    # Step 1: Extract OCR and exercises
    python full_benchmark_pipeline.py extract \
        --pdf "book.pdf" \
        --output-dir "results"

    # Step 2: Add reference answers to results/exercises/extracted_exercises.json
    # Edit the file and add "reference_answer": "..." to each exercise

    # Step 3: Run benchmark
    python full_benchmark_pipeline.py benchmark \
        --results-dir "results"
"""

import os
import sys
import argparse
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
import pymupdf
from tqdm import tqdm


class ExerciseBenchmarkPipeline:
    """Complete pipeline for OCR extraction and exercise benchmarking"""

    def __init__(self, output_dir: str, api_key: str = None):
        self.output_dir = Path(output_dir)
        self.raw_ocr_dir = self.output_dir / "raw_ocr"
        self.corrected_ocr_dir = self.output_dir / "corrected_ocr"
        self.exercise_dir = self.output_dir / "exercises"
        self.benchmark_dir = self.output_dir / "benchmark"

        # Create directories
        for d in [self.raw_ocr_dir, self.corrected_ocr_dir, self.exercise_dir, self.benchmark_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Initialize OpenAI
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def extract_ocr(self, pdf_path: str, max_pages: int = None):
        """Extract embedded PDF text and apply OLMoCR correction"""
        pdf_path = Path(pdf_path)

        print(f"\n{'='*60}")
        print("STEP 1: EXTRACT EMBEDDED TEXT FROM PDF")
        print("(No OCR - just extracting built-in text layer)")
        print(f"{'='*60}")

        doc = pymupdf.open(pdf_path)
        total_pages = min(len(doc), max_pages) if max_pages else len(doc)

        print(f"Processing {total_pages} pages...")
        for page_num in range(total_pages):
            page = doc[page_num]
            # Extract embedded text (NOT OCR - just the text layer in the PDF)
            text = page.get_text()

            page_name = f"page_{page_num+1:04d}"
            (self.raw_ocr_dir / f"{page_name}.txt").write_text(text)

            if (page_num + 1) % 50 == 0:
                print(f"  {page_num + 1}/{total_pages} pages...")

        print(f"✓ Embedded text extracted: {total_pages} pages")
        print(f"   Note: If PDF has no embedded text, provide text files manually in {self.raw_ocr_dir}/")

        print(f"\n{'='*60}")
        print("STEP 2: CORRECT OCR USING OLMoCR (allenai/olmOCR-2-7B-1025-FP8)")
        print(f"{'='*60}")

        # Create a temporary PDF with only the pages we need
        if max_pages:
            print(f"Creating temporary PDF with first {max_pages} pages...")
            temp_pdf = self.output_dir / "temp_input.pdf"
            temp_doc = pymupdf.open()
            for page_num in range(total_pages):
                temp_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            temp_doc.save(temp_pdf)
            temp_doc.close()
            pdf_to_process = temp_pdf
        else:
            pdf_to_process = pdf_path

        doc.close()

        # Run OLMoCR
        self.run_olmocr(pdf_to_process)

        # Clean up temp PDF
        if max_pages and temp_pdf.exists():
            temp_pdf.unlink()

        print(f"✓ Corrected OCR: {total_pages} pages")

    def run_olmocr(self, pdf_path: Path):
        """Run OLMoCR CLI to process PDF and extract corrected text"""
        # Create temporary workspace for OLMoCR
        with tempfile.TemporaryDirectory() as temp_workspace:
            temp_workspace_path = Path(temp_workspace)

            print(f"Running OLMoCR on {pdf_path.name}...")
            print("This may take several minutes depending on the number of pages...")

            try:
                # Run olmocr CLI
                cmd = [
                    "olmocr",
                    str(temp_workspace_path),
                    "--markdown",
                    "--pdfs", str(pdf_path)
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )

                print("✓ OLMoCR processing complete")

                # Parse OLMoCR output
                self.parse_olmocr_output(temp_workspace_path)

            except subprocess.CalledProcessError as e:
                print(f"✗ OLMoCR failed: {e}")
                print(f"stdout: {e.stdout}")
                print(f"stderr: {e.stderr}")
                raise
            except FileNotFoundError:
                print("✗ OLMoCR not found. Please install it first:")
                print("   pip install olmocr")
                print("   Or for GPU: pip install olmocr[gpu] --extra-index-url https://download.pytorch.org/whl/cu128")
                raise

    def parse_olmocr_output(self, workspace_path: Path):
        """Parse OLMoCR markdown output and save as individual page text files"""
        # OLMoCR saves output as markdown files
        md_files = list(workspace_path.glob("**/*.md"))

        if not md_files:
            print("Warning: No markdown output found from OLMoCR")
            return

        # Assuming single PDF, get the first markdown file
        md_file = md_files[0]
        content = md_file.read_text()

        # Split by page markers if present, or treat as single document
        # OLMoCR outputs full document markdown, we need to split by pages
        # For now, we'll try to parse the JSONL output instead
        jsonl_files = list(workspace_path.glob("**/*.jsonl"))

        if jsonl_files:
            jsonl_file = jsonl_files[0]
            with open(jsonl_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)

                    # Extract page-by-page content
                    if 'pages' in data:
                        for page in data['pages']:
                            page_num = page.get('page_num', 0)
                            text = page.get('response', {}).get('text', '')

                            page_name = f"page_{page_num:04d}"
                            (self.corrected_ocr_dir / f"{page_name}.txt").write_text(text)
        else:
            # Fallback: save entire markdown as single file and warn
            print("Warning: Could not find JSONL output, saving entire markdown")
            (self.corrected_ocr_dir / "page_0001.txt").write_text(content)

    def extract_exercises(self, max_exercises: int = None):
        """Extract exercises from corrected OCR"""
        print(f"\n{'='*60}")
        print("STEP 3: EXTRACT EXERCISES FROM CORRECTED OCR")
        print(f"{'='*60}")

        # Load corrected OCR text
        text_files = sorted(self.corrected_ocr_dir.glob("*.txt"))
        all_text = ""
        for txt_file in text_files:
            content = txt_file.read_text()
            if content.strip():
                all_text += f"\n\n=== {txt_file.stem} ===\n{content}"

        print(f"Text length: {len(all_text)} characters")

        prompt = f"""Extract ALL exercises and practice problems from this textbook.

Look for sections like:
- "REVIEWING YOUR UNDERSTANDING"
- "CHECKING YOUR COMPREHENSION"
- "PRACTICE PROBLEMS"
- "EXERCISES"
- "QUESTIONS"

For each exercise, extract:
1. The full question text (exact)
2. Chapter or section name (if available)
3. Source page

IMPORTANT:
- Extract ACTUAL questions from the text - do NOT generate fake questions
- Include both multiple-choice AND open-ended questions
- Preserve the complete question wording

Return JSON array:
[
  {{
    "exercise_id": 1,
    "question": "Exact question from book",
    "chapter": "Chapter 3" or null,
    "reference_answer": null,
    "source_page": "page_0036"
  }}
]

OCR Text:
{all_text[:100000]}

Return ONLY valid JSON."""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        response_text = response.choices[0].message.content
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        exercises = json.loads(response_text.strip())

        if max_exercises and len(exercises) > max_exercises:
            exercises = exercises[:max_exercises]

        print(f"✓ Extracted {len(exercises)} exercises")

        # Save
        exercise_file = self.exercise_dir / "extracted_exercises.json"
        with open(exercise_file, 'w') as f:
            json.dump({"total_exercises": len(exercises), "exercises": exercises}, f, indent=2)

        print(f"✓ Saved to: {exercise_file}")
        print(f"\nNEXT: Add reference answers to each exercise in {exercise_file}")
        print('      Add "reference_answer": "..." to each exercise')

        return exercises

    def load_all_text(self, source_dir: Path) -> str:
        """Load all text from directory"""
        text_files = sorted(source_dir.glob("*.txt"))
        return "\n\n".join(f.read_text() for f in text_files if f.read_text().strip())

    def answer_exercise(self, exercise: Dict, context: str, condition_name: str) -> Dict:
        """Answer single exercise with given context"""
        question = exercise["question"]

        if not context:
            prompt = f"""Answer this question using your general knowledge.

Question: {question}

Provide a clear, concise answer (2-5 sentences).

Return JSON: {{"answer": "Your answer here", "confidence": "low/medium/high"}}"""
        else:
            # Get relevant chapter context
            chapter = exercise.get("chapter", "")
            source_page = exercise.get("source_page", "")

            prompt = f"""Answer this question using ONLY the provided textbook content.

Textbook Content:
{context[:30000]}

Question: {question}

Provide a clear, concise answer based on the textbook (2-5 sentences).

Return JSON: {{"answer": "Your answer here", "confidence": "low/medium/high"}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            response_text = response.choices[0].message.content
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            return json.loads(response_text.strip())
        except Exception as e:
            return {"answer": "Error generating answer", "confidence": "error"}

    def evaluate_answer(self, question: str, predicted_answer: str, reference_answer: str) -> Dict:
        """Evaluate answer quality using LLM (0-10 scale)"""
        prompt = f"""Evaluate the quality of the predicted answer compared to the reference answer.

Question: {question}

Reference Answer: {reference_answer}

Predicted Answer: {predicted_answer}

Rate the predicted answer on a scale of 0-10 where:
- 0-2: Incorrect or completely off-topic
- 3-4: Partially correct but missing key points
- 5-6: Mostly correct but lacks detail or has minor errors
- 7-8: Correct with good detail
- 9-10: Excellent, comprehensive answer

Return JSON: {{"score": 7, "reasoning": "Explanation of score"}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            response_text = response.choices[0].message.content
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            return json.loads(response_text.strip())
        except:
            return {"score": 0, "reasoning": "Evaluation error"}

    def benchmark_condition(self, exercises: List[Dict], context: str, condition_name: str) -> Dict:
        """Benchmark one condition"""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {condition_name}")
        print(f"{'='*60}")

        results = []
        total_score = 0

        for exercise in tqdm(exercises, desc=condition_name):
            # Generate answer
            response = self.answer_exercise(exercise, context, condition_name)
            predicted_answer = response["answer"]

            # Evaluate answer
            reference_answer = exercise.get("reference_answer", "")
            if reference_answer:
                evaluation = self.evaluate_answer(
                    exercise["question"],
                    predicted_answer,
                    reference_answer
                )
                score = evaluation["score"]
                eval_reasoning = evaluation["reasoning"]
            else:
                score = 0
                eval_reasoning = "No reference answer provided"

            total_score += score

            results.append({
                "exercise_id": exercise["exercise_id"],
                "question": exercise["question"],
                "reference_answer": reference_answer,
                "predicted_answer": predicted_answer,
                "score": score,
                "confidence": response.get("confidence"),
                "evaluation_reasoning": eval_reasoning
            })

        avg_score = (total_score / len(exercises)) if exercises else 0

        return {
            "condition_name": condition_name,
            "total": len(exercises),
            "total_score": total_score,
            "avg_score": round(avg_score, 2),
            "results": results
        }

    def run_benchmark(self):
        """Run benchmark on all 3 conditions"""
        print(f"\n{'='*60}")
        print("BENCHMARK: 3 CONDITIONS")
        print(f"{'='*60}")

        # Load exercises
        exercise_file = self.exercise_dir / "extracted_exercises.json"
        if not exercise_file.exists():
            print(f"✗ Exercise file not found: {exercise_file}")
            return

        with open(exercise_file) as f:
            data = json.load(f)
            exercises = data["exercises"]

        # Check for reference answers
        if not exercises[0].get("reference_answer"):
            print("✗ No reference answers found in exercises!")
            print("   Add 'reference_answer' field to each exercise first")
            return

        print(f"Loaded {len(exercises)} exercises")

        # Load contexts
        raw_text = self.load_all_text(self.raw_ocr_dir)
        corrected_text = self.load_all_text(self.corrected_ocr_dir)

        # Benchmark each condition
        results = {}

        results['baseline'] = self.benchmark_condition(
            exercises, "", "Baseline: No Context")

        results['raw_ocr'] = self.benchmark_condition(
            exercises, raw_text, "Raw OCR Context")

        results['corrected_ocr'] = self.benchmark_condition(
            exercises, corrected_text, "Corrected OCR Context (OLMoCR)")

        # Save results
        results_file = self.benchmark_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        for key, res in results.items():
            print(f"{res['condition_name']}: {res['avg_score']:.2f}/10 (total: {res['total_score']}/{res['total']*10})")

        print(f"\n✓ OCR Correction Improvement: {results['corrected_ocr']['avg_score'] - results['raw_ocr']['avg_score']:.2f} points")
        print(f"✓ Full results: {results_file}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract OCR and exercises')
    extract_parser.add_argument('--pdf', required=True)
    extract_parser.add_argument('--output-dir', required=True)
    extract_parser.add_argument('--max-pages', type=int)
    extract_parser.add_argument('--max-exercises', type=int)

    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run benchmark')
    bench_parser.add_argument('--results-dir', required=True)

    # Both commands
    parser.add_argument('--api-key')

    args = parser.parse_args()

    if args.command == 'extract':
        pipeline = ExerciseBenchmarkPipeline(args.output_dir, args.api_key)
        pipeline.extract_ocr(args.pdf, args.max_pages)
        pipeline.extract_exercises(args.max_exercises)

    elif args.command == 'benchmark':
        pipeline = ExerciseBenchmarkPipeline(args.results_dir, args.api_key)
        pipeline.run_benchmark()


if __name__ == "__main__":
    main()
