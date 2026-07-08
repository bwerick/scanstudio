# Exercise Benchmark Pipeline - OCR Quality Impact

## Overview

This pipeline tests whether OCR-digitizing books improves an LLM's ability to answer exercises from those books. We compare 3 conditions per exercise:

- **Baseline**: LLM answers with no context (general knowledge only)
- **Raw OCR**: LLM answers with raw OCR text of the relevant chapter as context (embedded PDF text)
- **Corrected OCR**: LLM answers with LLM-corrected OCR text of the relevant chapter as context (OLMoCR)

## Full Workflow

### Step 1: Extract OCR and Exercises

```bash
export OPENAI_API_KEY="your-key"

python testing/full_benchmark_pipeline.py extract \
  --pdf "output/az-principles-of-real-estate/Arizona Principles of Real Estate.pdf" \
  --output-dir "testing/benchmark-results"
```

**What this does:**
1. Extracts **Raw OCR** (PyMuPDF embedded text) → `benchmark-results/raw_ocr/`
2. Extracts **Corrected OCR** (OLMoCR) → `benchmark-results/corrected_ocr/`
3. Extracts **Exercises** from book → `benchmark-results/exercises/extracted_exercises.json`

**Time:** ~30-90 min on Legion GPU for full book (OLMoCR is more intensive)

### Step 2: Add Reference Answers

Edit `benchmark-results/exercises/extracted_exercises.json` and add reference answers for evaluation:

```json
{
  "exercises": [
    {
      "exercise_id": 1,
      "question": "Explain the role of a designated broker in a real estate brokerage.",
      "reference_answer": "A designated broker is responsible for...",  ← ADD THIS
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

**What this does:**
1. Loads exercises with reference answers
2. Answers each exercise using 3 different conditions:
   - **Baseline**: No context (LLM general knowledge)
   - **Raw OCR**: Raw embedded text from PDF as context
   - **Corrected OCR**: OLMoCR-corrected text as context
3. Uses LLM to evaluate answer quality against reference answers
4. Generates comparison report

**Time:** ~10-30 min depending on number of exercises

### Step 4: View Results

Results saved to `benchmark-results/benchmark/benchmark_results.json`:

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

Scores are 0-10 based on LLM evaluation of answer quality vs reference answers.

## Quick Test (50 pages)

```bash
# Step 1: Extract (faster)
python testing/full_benchmark_pipeline.py extract \
  --pdf "output/az-principles-of-real-estate/Arizona Principles of Real Estate.pdf" \
  --output-dir "testing/test-run" \
  --max-pages 50 \
  --max-exercises 10

# Step 2: Add reference answers manually

# Step 3: Benchmark
python testing/full_benchmark_pipeline.py benchmark \
  --results-dir "testing/test-run"
```

## Running on Legion

```bash
# Transfer files
scp -r testing/ "output/az-principles-of-real-estate/" user@legion-ip:~/scanstudio/

# SSH
ssh user@legion-ip
cd ~/scanstudio

# Install deps (one-time)
pip install python-doctr openai torch torchvision pymupdf tqdm

# Run
export OPENAI_API_KEY="your-key"
python testing/full_benchmark_pipeline.py extract \
  --pdf "output/az-principles-of-real-estate/Arizona Principles of Real Estate.pdf" \
  --output-dir "results"

# Add answers, then:
python testing/full_benchmark_pipeline.py benchmark --results-dir "results"
```

## Output Structure

```
benchmark-results/
├── raw_ocr/            # PyMuPDF embedded text extraction
│   ├── page_0001.txt
│   └── ...
├── corrected_ocr/      # OLMoCR corrected text
│   ├── page_0001.txt
│   └── ...
├── exercises/          # Extracted exercises
│   └── extracted_exercises.json
└── benchmark/          # Benchmark results
    └── benchmark_results.json
```

## Key Findings

The benchmark will show:
- **Baseline performance** (LLM general knowledge)
- **Impact of raw OCR** (does embedded PDF text help?)
- **Impact of corrected OCR** (does OLMoCR correction improve answers?)
- **OCR correction benefit** (difference between raw and corrected)

This quantifies whether OCR-digitizing books improves an LLM's ability to answer exercises from those books!
