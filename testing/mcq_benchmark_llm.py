"""
LLM-Based MCQ Extraction and Benchmarking

This pipeline:
1. Extracts MCQs from the textbook using OpenAI GPT-4
2. Benchmarks MCQ answering accuracy across 4 settings:
   - Setting A: No textbook (baseline)
   - Setting B: Dirty OCR (raw EasyOCR)
   - Setting C: Parsed OCR (DocTR)
   - Setting D: Ground Truth PDF

Usage:
    export OPENAI_API_KEY="your-key"
    python mcq_benchmark_llm.py --ocr-results testing/az-principles-results
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
from tqdm import tqdm


class MCQBenchmark:
    """LLM-based MCQ extraction and benchmarking"""

    def __init__(self, ocr_results_dir: str, api_key: Optional[str] = None):
        """
        Initialize benchmark system

        Args:
            ocr_results_dir: Directory containing OCR results
            api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
        """
        self.results_dir = Path(ocr_results_dir)
        self.dirty_ocr_dir = self.results_dir / "dirty_ocr"
        self.parsed_ocr_dir = self.results_dir / "parsed_ocr"
        self.ground_truth_dir = self.results_dir / "ground_truth"
        self.output_dir = self.results_dir / "mcq_results"
        self.output_dir.mkdir(exist_ok=True)

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )

        print(f"Initialized MCQ Benchmark")
        print(f"  OCR Results: {self.results_dir}")
        print(f"  Output: {self.output_dir}")

    def load_all_text(self, source_dir: Path) -> str:
        """Load and concatenate all text files from a directory"""
        text_files = sorted(source_dir.glob("*.txt"))
        all_text = ""
        for txt_file in text_files:
            content = txt_file.read_text()
            if content.strip():  # Only add non-empty files
                all_text += content + "\n\n"
        return all_text

    def extract_mcqs_with_llm(self, text: str, source_name: str) -> List[Dict]:
        """
        Extract MCQs from text using OpenAI GPT-4

        Args:
            text: Source text to extract MCQs from
            source_name: Name of source (for logging)

        Returns:
            List of MCQ dictionaries with question, choices, and correct answer
        """
        print(f"\nExtracting MCQs from {source_name} using OpenAI GPT-4...")

        if not text.strip():
            print(f"  ⚠️  No text found in {source_name}")
            return []

        prompt = f"""You are analyzing a real estate textbook to extract multiple-choice questions (MCQs).

Please extract ALL multiple-choice questions from the following text. For each MCQ, provide:
1. The question text
2. All answer choices (A, B, C, D, etc.)
3. The correct answer(s)

Return the results as a JSON array with this structure:
[
  {{
    "question": "Question text here?",
    "choices": {{
      "A": "First choice",
      "B": "Second choice",
      "C": "Third choice",
      "D": "Fourth choice"
    }},
    "correct_answer": "A",
    "source_page": "page number if identifiable"
  }},
  ...
]

Text to analyze:

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
            # Remove markdown code blocks if present
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

    def answer_mcq_with_context(self, mcq: Dict, context: str, setting_name: str) -> Dict:
        """
        Answer a single MCQ using the provided context

        Args:
            mcq: MCQ dictionary with question and choices
            context: Textbook content to use as reference
            setting_name: Name of setting (for logging)

        Returns:
            Dictionary with question, predicted answer, confidence, and reasoning
        """
        question = mcq["question"]
        choices = mcq["choices"]

        # Format choices
        choices_text = "\n".join([f"{key}. {value}" for key, value in choices.items()])

        if not context.strip():
            # Setting A: No context
            prompt = f"""Answer this multiple-choice question based on your general knowledge:

Question: {question}

{choices_text}

Provide your answer in this JSON format:
{{
  "answer": "A",
  "confidence": "low/medium/high",
  "reasoning": "Brief explanation"
}}"""
        else:
            # Settings B, C, D: With context
            prompt = f"""Answer this multiple-choice question using ONLY the information from the provided textbook excerpt. If the answer cannot be determined from the excerpt, make your best guess but indicate low confidence.

Textbook excerpt:
{context[:30000]}

Question: {question}

{choices_text}

Provide your answer in this JSON format:
{{
  "answer": "A",
  "confidence": "low/medium/high",
  "reasoning": "Brief explanation citing the textbook"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            response_text = response.choices[0].message.content

            # Parse JSON response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            result = json.loads(response_text.strip())
            return result

        except Exception as e:
            print(f"    ✗ Error answering MCQ: {str(e)}")
            return {
                "answer": "A",
                "confidence": "error",
                "reasoning": f"Error: {str(e)}"
            }

    def benchmark_setting(self, mcqs: List[Dict], context: str, setting_name: str) -> Dict:
        """
        Benchmark MCQ answering for one setting

        Args:
            mcqs: List of MCQs with correct answers
            context: Textbook content for this setting (empty for Setting A)
            setting_name: Name of the setting

        Returns:
            Dictionary with accuracy metrics
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking {setting_name}")
        print(f"{'='*60}")
        print(f"Total MCQs: {len(mcqs)}")
        print(f"Context length: {len(context)} characters")

        results = []
        correct = 0
        total = 0

        for idx, mcq in enumerate(tqdm(mcqs, desc=f"Answering MCQs ({setting_name})")):
            response = self.answer_mcq_with_context(mcq, context, setting_name)

            predicted = response["answer"]
            actual = mcq.get("correct_answer", "")
            is_correct = predicted.upper() == actual.upper()

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "question": mcq["question"],
                "choices": mcq["choices"],
                "correct_answer": actual,
                "predicted_answer": predicted,
                "is_correct": is_correct,
                "confidence": response.get("confidence", "unknown"),
                "reasoning": response.get("reasoning", "")
            })

        accuracy = (correct / total * 100) if total > 0 else 0

        return {
            "setting_name": setting_name,
            "total_mcqs": total,
            "correct": correct,
            "accuracy": accuracy,
            "results": results
        }

    def run_full_benchmark(self, sample_size: Optional[int] = None):
        """
        Run complete benchmark across all 4 settings

        Args:
            sample_size: Limit number of MCQs to test (for quick testing)
        """
        print(f"\n{'='*60}")
        print("STEP 1: Extract MCQs from Ground Truth")
        print(f"{'='*60}")

        # Load ground truth text
        ground_truth_text = self.load_all_text(self.ground_truth_dir)

        # If ground truth is empty, use parsed OCR as source
        if not ground_truth_text.strip():
            print("⚠️  Ground truth is empty, using Parsed OCR as MCQ source...")
            ground_truth_text = self.load_all_text(self.parsed_ocr_dir)

        # Extract MCQs using LLM
        all_mcqs = self.extract_mcqs_with_llm(ground_truth_text, "Ground Truth")

        if not all_mcqs:
            print("✗ No MCQs found! Cannot proceed with benchmark.")
            return

        # Sample if requested
        if sample_size and len(all_mcqs) > sample_size:
            all_mcqs = all_mcqs[:sample_size]
            print(f"Using sample of {sample_size} MCQs for testing")

        # Save extracted MCQs
        mcq_file = self.output_dir / "extracted_mcqs.json"
        with open(mcq_file, 'w') as f:
            json.dump(all_mcqs, f, indent=2)
        print(f"✓ Saved extracted MCQs to {mcq_file}")

        print(f"\n{'='*60}")
        print("STEP 2: Benchmark All Settings")
        print(f"{'='*60}")

        # Load contexts for each setting
        dirty_ocr_text = self.load_all_text(self.dirty_ocr_dir)
        parsed_ocr_text = self.load_all_text(self.parsed_ocr_dir)

        # Benchmark each setting
        benchmark_results = {}

        # Setting A: No textbook
        benchmark_results['setting_a'] = self.benchmark_setting(
            all_mcqs, "", "Setting A: No Textbook (Baseline)"
        )

        # Setting B: Dirty OCR
        benchmark_results['setting_b'] = self.benchmark_setting(
            all_mcqs, dirty_ocr_text, "Setting B: Dirty OCR (EasyOCR)"
        )

        # Setting C: Parsed OCR
        benchmark_results['setting_c'] = self.benchmark_setting(
            all_mcqs, parsed_ocr_text, "Setting C: Parsed OCR (DocTR)"
        )

        # Setting D: Ground truth
        benchmark_results['setting_d'] = self.benchmark_setting(
            all_mcqs, ground_truth_text, "Setting D: Ground Truth PDF"
        )

        # Save results
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)

        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")

        for setting_key, results in benchmark_results.items():
            print(f"\n{results['setting_name']}:")
            print(f"  Accuracy: {results['accuracy']:.2f}% ({results['correct']}/{results['total_mcqs']})")

        print(f"\n✓ Full results saved to {results_file}")

        # Generate comparison report
        self.generate_report(benchmark_results)

    def generate_report(self, benchmark_results: Dict):
        """Generate a markdown report of results"""
        report_file = self.output_dir / "benchmark_report.md"

        with open(report_file, 'w') as f:
            f.write("# OCR Quality Benchmark Report\n\n")
            f.write("## Summary\n\n")
            f.write("| Setting | Description | Accuracy | Correct/Total |\n")
            f.write("|---------|-------------|----------|---------------|\n")

            for setting_key, results in benchmark_results.items():
                f.write(f"| {setting_key.upper()} | {results['setting_name']} | "
                       f"{results['accuracy']:.2f}% | {results['correct']}/{results['total_mcqs']} |\n")

            f.write("\n## Analysis\n\n")

            # Calculate improvements
            baseline_acc = benchmark_results['setting_a']['accuracy']
            dirty_acc = benchmark_results['setting_b']['accuracy']
            parsed_acc = benchmark_results['setting_c']['accuracy']
            gt_acc = benchmark_results['setting_d']['accuracy']

            f.write(f"- **Baseline (no textbook)**: {baseline_acc:.2f}%\n")
            f.write(f"- **Dirty OCR improvement**: +{dirty_acc - baseline_acc:.2f}%\n")
            f.write(f"- **Parsed OCR improvement**: +{parsed_acc - baseline_acc:.2f}%\n")
            f.write(f"- **Ground truth improvement**: +{gt_acc - baseline_acc:.2f}%\n")
            f.write(f"\n**OCR quality impact**: Parsed OCR is {parsed_acc - dirty_acc:.2f}% better than Dirty OCR\n")

        print(f"✓ Report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(description="LLM-based MCQ benchmarking")
    parser.add_argument("--ocr-results", required=True, help="Directory with OCR results")
    parser.add_argument("--api-key", help="OpenAI API key (or use OPENAI_API_KEY env var)")
    parser.add_argument("--sample-size", type=int, help="Number of MCQs to test (for quick testing)")

    args = parser.parse_args()

    # Initialize and run benchmark
    benchmark = MCQBenchmark(args.ocr_results, args.api_key)
    benchmark.run_full_benchmark(sample_size=args.sample_size)


if __name__ == "__main__":
    main()
