# MCQ Extraction Pipeline

Single unified pipeline that extracts MCQs from PDF textbooks.

## Quick Start

```bash
# 1. Set API key
export OPENAI_API_KEY="your-openai-key"

# 2. Run pipeline
python testing/run_mcq_pipeline.py \
  --pdf "output/az-principles-of-real-estate/Arizona Principles of Real Estate.pdf" \
  --output-dir "testing/results"
```

## Installation

```bash
pip install python-doctr openai torch torchvision
```

## What It Does

1. **Clean OCR Extraction** (DocTR)
   - Extracts high-quality text from PDF
   - Auto-detects GPU (CUDA/MPS) or falls back to CPU
   - Saves to `output-dir/clean_ocr/`

2. **MCQ Extraction** (GPT-4)
   - Finds and extracts multiple-choice questions
   - Saves to `output-dir/mcqs/extracted_mcqs.json`

## Options

```bash
--max-pages 100    # Process only first 100 pages (for testing)
--max-mcqs 50      # Extract max 50 MCQs (for testing)
--api-key KEY      # Provide API key via command line
```

## Output Format

Results in `output-dir/mcqs/extracted_mcqs.json`:

```json
{
  "total_mcqs": 20,
  "mcqs": [
    {
      "question_id": 1,
      "question": "Each brokerage is operated by a/an:",
      "choices": {
        "A": "designated broker",
        "B": "DBW&F (Doing business with and for)",
        "C": "associate broker",
        "D": "independent broker"
      },
      "source_page": "page_0036"
    }
  ]
}
```

## Performance

- **Mac (M2 MPS)**: ~3-10 pages/min
- **Legion (NVIDIA GPU)**: ~10-30 pages/min
- **CPU only**: ~1-3 pages/min

## Examples

### Test on 50 pages:
```bash
python testing/run_mcq_pipeline.py \
  --pdf "book.pdf" \
  --output-dir "test-run" \
  --max-pages 50
```

### Full book, max 100 MCQs:
```bash
python testing/run_mcq_pipeline.py \
  --pdf "book.pdf" \
  --output-dir "full-run" \
  --max-mcqs 100
```
