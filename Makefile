.PHONY: install frameextraction parseOCR

install_opencv:
	apt-get update && apt-get install -y python3-opencv

install:
	pip install -r requirements.txt

frameextraction:
	python frameextraction.py

parse_OCR: frameextraction
	python frame_ocr.py

