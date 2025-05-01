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
FRAME_IMAGES = $(wildcard $(OUTPUT_DIR)/*/*.jpg)

.PHONY: all
all: frameextraction

$(OUTPUT_DIR)/%/: Videos/%.mov
	@echo "Extracting frames from $< to $@"
	python frameextraction.py --kwargs out_path=$(OUTPUT_DIR) video=$<

$(OUTPUT_DIR)/%/: Videos/%.mp4
	@echo "Extracting frames from $< to $@"
	python frameextraction.py --kwargs out_path=$(OUTPUT_DIR) video=$<


MARKDOWNS = $(FRAME_IMAGES:.jpg=.md)
%.md: %.jpg
	@echo "Generating Markdown for $<"
	convert $< -strip -interlace Plane -quality 100% intermediate.jpg
	marker_single intermediate.jpg $@
	@rm -f intermediate.jpg

.PHONY: install
install: 
	pip install -r requirements.txt

.PHONY: install_opencv
install_opencv:
	apt-get update && apt-get install -y python3-opencv

.PHONY: frameextraction
frameextraction: $(FRAME_DIRS)

.PHONY: generate_md
generate_md: $(FRAME_DIRS) $(MARKDOWNS)

parse_OCR: frameextraction
	python frame_ocr.py

easy_OCR: frameextraction
	python easyframe_ocr.py

clean:
	rm -rf $(OUTPUT_DIR)