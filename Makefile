.PHONY: install frameextraction parseOCR

install:
    pip install -r requirements.txt

frameextraction:
    python frameextraction.py

parse_OCR: frameextraction
    python frame_ocr.py

