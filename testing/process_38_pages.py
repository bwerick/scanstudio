#!/usr/bin/env python3
"""Process first 38 pages only"""

from pathlib import Path
import json
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import pymupdf

pdf_path = 'output/az-principles-of-real-estate/Arizona Principles of Real Estate.pdf'
output_dir = Path('testing/az-full')
max_pages = 38

# Create output directories
dirty_ocr_dir = output_dir / 'dirty_ocr'
parsed_ocr_dir = output_dir / 'parsed_ocr'
dirty_ocr_dir.mkdir(parents=True, exist_ok=True)
parsed_ocr_dir.mkdir(parents=True, exist_ok=True)

print(f'Processing first {max_pages} pages...')

# Step 1: PyMuPDF (Dirty OCR)
print('\n=== Step 1: Dirty OCR (PyMuPDF) ===')
doc = pymupdf.open(pdf_path)
for page_num in range(min(max_pages, len(doc))):
    page = doc[page_num]
    text = page.get_text()
    page_name = f'page_{page_num+1:04d}'
    (dirty_ocr_dir / f'{page_name}.txt').write_text(text)
    if (page_num + 1) % 10 == 0:
        print(f'  {page_num + 1}/{max_pages} pages...')
print(f'✓ Dirty OCR complete')

# Step 2: DocTR (Parsed OCR)
print('\n=== Step 2: Parsed OCR (DocTR) ===')
model = ocr_predictor(pretrained=True)
doc_file = DocumentFile.from_pdf(pdf_path)
# Process only first max_pages
result = model(doc_file[:max_pages])
for page_idx, page in enumerate(result.pages):
    page_name = f'page_{page_idx+1:04d}'
    (parsed_ocr_dir / f'{page_name}.txt').write_text(page.render())
    if (page_idx + 1) % 10 == 0:
        print(f'  {page_idx + 1}/{max_pages} pages...')
print(f'✓ Parsed OCR complete')
print(f'\n✓ Done! Processed {max_pages} pages.')
print(f'Results in: {output_dir}')
