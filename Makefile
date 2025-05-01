#  _____ __    _____ _____ _____ __    _____ _____ 
# |   __|  |  |     | __  |  _  |  |  |  _  |   | |
# |   __|  |__|  |  |    -|   __|  |__|     | | | |
# |__|  |_____|_____|__|__|__|  |_____|__|__|_|___|
#         

VIDEOS ?= $(wildcard Videos/*.mp4 Videos/*.mov)
OUTPUT_DIR = test_frames

# Extract the base names of videos: video1.mp4 → video1
BASENAMES = $(notdir $(basename $(VIDEOS)))

.PHONY: all
all: $(FRAME_DIRS) $(FRAME_TEXTS)

# Construct output dirs: test_frames/video1/, etc.
FRAME_DIRS = $(addsuffix /,$(addprefix $(OUTPUT_DIR)/,$(BASENAMES)))
FRAME_IMAGES = $(wildcard $(OUTPUT_DIR)/*/*.jpg)

$(OUTPUT_DIR)/%/: Videos/%.mov
	@echo "Extracting frames from $< to $@"
	python frameextraction.py --kwargs out_path=$(OUTPUT_DIR) video=$<

$(OUTPUT_DIR)/%/: Videos/%.mp4
	@echo "Extracting frames from $< to $@"
	python frameextraction.py --kwargs out_path=$(OUTPUT_DIR) video=$<

.PHONY: frameextraction
frameextraction: $(FRAME_DIRS)

# Marker-pdf extraction 
MARKDOWNS = $(FRAME_IMAGES:.jpg=.md)
%.md: %.jpg
	@echo "Generating Markdown for $<"
	convert $< -strip -interlace Plane -quality 100% intermediate.jpg
	marker_single intermediate.jpg $@
	@rm -f intermediate.jpg

.PHONY: generate_md
generate_md: $(FRAME_DIRS) $(MARKDOWNS)

# OCR with doctr
FRAME_TEXTS = $(FRAME_IMAGES:.jpg=.txt)
%.txt: %.jpg
	@echo "Generating OCR text for $<"
	python doctr_ocr.py --kwargs frame=$< out_path=$@

.PHONY: doctr_ocr
doctr_ocr: $(FRAME_TEXTS)

# OCR with easyOCR
easy_OCR: frameextraction
	python easyframe_ocr.py


.PHONY: install
install: 
	pip install -r requirements.txt


.PHONY: install_opencv
install_opencv:
	apt-get update && apt-get install -y python3-opencv


.PHONY: clean
clean:
	rm -rf $(OUTPUT_DIR)

