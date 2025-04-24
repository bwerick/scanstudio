#  _____ __    _____ _____ _____ __    _____ _____ 
# |   __|  |  |     | __  |  _  |  |  |  _  |   | |
# |   __|  |__|  |  |    -|   __|  |__|     | | | |
# |__|  |_____|_____|__|__|__|  |_____|__|__|_|___|
#         

VIDEOS ?= $(wildcard Videos/*.mp4 Videos/*.mov)
OUTPUT_DIR = test_frames

# Extract the base names of videos: video1.mp4 → video1
BASENAMES = $(notdir $(basename $(VIDEOS)))

# Construct output dirs: test_frames/video1/, etc.
FRAME_DIRS = $(addsuffix /,$(addprefix $(OUTPUT_DIR)/,$(BASENAMES)))


.PHONY: all
all: $(FRAME_DIRS)

$(OUTPUT_DIR)/%/: Videos/%.mov
	@echo "Extracting frames from $< to $@"
	python frameextraction.py --kwargs out_path=$(OUTPUT_DIR) video=$<

$(OUTPUT_DIR)/%/: Videos/%.mp4
	@echo "Extracting frames from $< to $@"
	python frameextraction.py --kwargs out_path=$(OUTPUT_DIR) video=$<

.PHONY: install
install: 
	pip install -r requirements.txt

.PHONY: install_opencv
install_opencv:
	apt-get update && apt-get install -y python3-opencv

frameextraction:
	python frameextraction.py --kwargs out_path=$(OUTPUT_DIR)

parse_OCR: frameextraction
	python frame_ocr.py

easy_OCR: frameextraction
	python easyframe_ocr.py

clean:
	rm -rf $(OUTPUT_DIR)

debug:
	@echo VIDEOS: $(VIDEOS)
	@echo BASENAMES: $(BASENAMES)
	@echo FRAME_DIRS: $(FRAME_DIRS)
