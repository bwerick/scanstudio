# Benchmark Pipeline Changes

## Summary

The benchmarking pipeline has been completely redesigned to test whether OCR-digitizing books improves an LLM's ability to answer exercises from those books.

## What Changed

### Previous Approach (MCQs, 4 Settings)
- **Focus**: MCQ answering accuracy
- **Settings**:
  - Setting A: No textbook (baseline)
  - Setting B: Dirty OCR (PyMuPDF)
  - Setting C: Clean OCR (DocTR)
  - Setting D: Ground truth
- **Evaluation**: Binary correct/incorrect
- **OCR Engines**: PyMuPDF + DocTR

### New Approach (Exercises, 3 Conditions)
- **Focus**: Exercise answering quality
- **Conditions**:
  - **Baseline**: LLM answers with no context
  - **Raw Text**: LLM answers with embedded PDF text (no OCR - just PDF text layer)
  - **Corrected OCR**: LLM answers with OLMoCR-processed text (actual OCR with correction)
- **Evaluation**: LLM scoring (0-10 scale) against reference answers
- **Text Sources**: Embedded PDF text (raw) vs **OLMoCR v0.4.0** (allenai/olmOCR-2-7B-1025-FP8)

## Key Improvements

1. **More Realistic Testing**: Uses actual exercises (not just MCQs) which better reflects real-world use cases
2. **Direct OCR Comparison**: Raw OCR vs Corrected OCR directly tests the value of OCR correction
3. **Nuanced Evaluation**: 0-10 scoring allows for partial credit and better differentiation
4. **Simplified Pipeline**: 3 conditions instead of 4, removing the need for ground truth text

## Directory Structure Changes

### Before
```
benchmark-results/
├── dirty_ocr/
├── clean_ocr/
├── mcqs/
└── benchmark/
```

### After
```
benchmark-results/
├── raw_ocr/
├── corrected_ocr/
├── exercises/
└── benchmark/
```

## Installation & Requirements

### Install OLMoCR

```bash
# For CPU or remote inference (lightweight, ~2GB)
pip install olmocr

# For local GPU inference (requires NVIDIA GPU with 12GB+ RAM)
pip install olmocr[gpu] --extra-index-url https://download.pytorch.org/whl/cu128

# Other dependencies
pip install pymupdf openai tqdm
```

### System Requirements

- **Raw OCR**: No GPU needed (PyMuPDF)
- **Corrected OCR**: Requires one of:
  - NVIDIA GPU (12GB+ VRAM, 30GB disk space)
  - Remote OLMoCR server endpoint
- **Exercise Extraction & Evaluation**: OpenAI API key

## Usage

### Step 1: Extract OCR and Exercises
```bash
export OPENAI_API_KEY="your-key"

python testing/full_benchmark_pipeline.py extract \
  --pdf "path/to/book.pdf" \
  --output-dir "testing/benchmark-results"
```

### Step 2: Add Reference Answers
Edit `benchmark-results/exercises/extracted_exercises.json` and add reference answers:
```json
{
  "exercises": [
    {
      "exercise_id": 1,
      "question": "Explain the role of a designated broker...",
      "reference_answer": "A designated broker is responsible for...",
      "chapter": "Chapter 3",
      "source_page": "page_0036"
    }
  ]
}
```

### Step 3: Run Benchmark
```bash
python testing/full_benchmark_pipeline.py benchmark \
  --results-dir "testing/benchmark-results"
```

## Results Format

```json
{
  "baseline": {
    "condition_name": "Baseline: No Context",
    "avg_score": 3.2,
    "total": 20
  },
  "raw_ocr": {
    "condition_name": "Raw OCR Context",
    "avg_score": 5.8,
    "total": 20
  },
  "corrected_ocr": {
    "condition_name": "Corrected OCR Context (OLMoCR)",
    "avg_score": 7.4,
    "total": 20
  }
}
```

## Research Question

**Does OCR-digitizing books (with OLMoCR) improve an LLM's ability to answer exercises compared to using embedded PDF text?**

The pipeline directly compares:
- **Raw Embedded Text** (PDF's built-in text layer - no OCR)
- **OLMoCR Text** (actual OCR with AI-powered correction)

The difference in average scores shows whether OCR-digitizing adds value over just using embedded PDF text for downstream LLM tasks.
