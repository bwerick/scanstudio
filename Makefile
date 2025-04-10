#  _____ __    _____ _____ _____ __    _____ _____ 
# |   __|  |  |     | __  |  _  |  |  |  _  |   | |
# |   __|  |__|  |  |    -|   __|  |__|     | | | |
# |__|  |_____|_____|__|__|__|  |_____|__|__|_|___|
#         

.PHONY: install frameextraction parseOCR

install_opencv:
	apt-get update && apt-get install -y python3-opencv

install:
	pip install -r requirements.txt

frameextraction:
	python frameextraction.py

parse_OCR: frameextraction
	python frame_ocr.py

easy_OCR: frameextraction
	python easyframe_ocr.py

clean:
	rm -rf test_frames