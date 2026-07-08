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
  - **Raw OCR**: LLM answers with raw embedded PDF text
  - **Corrected OCR**: LLM answers with OLMoCR-corrected text
- **Evaluation**: LLM scoring (0-10 scale) against reference answers
- **OCR Engines**: PyMuPDF + LLM-based correction (OLMoCR approach)

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

**Does OCR correction improve an LLM's ability to answer exercises from books?**

The pipeline directly compares:
- **Raw OCR** (baseline OCR quality)
- **Corrected OCR** (OLMoCR-improved quality)

The difference in average scores shows the impact of OCR correction on downstream LLM performance.
