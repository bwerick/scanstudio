#  _____ __    _____ _____ _____ __    _____ _____ 
# |   __|  |  |     | __  |  _  |  |  |  _  |   | |
# |   __|  |__|  |  |    -|   __|  |__|     | | | |
# |__|  |_____|_____|__|__|__|  |_____|__|__|_|___|
#         

VIDEOS ?= $(wildcard Videos/*.mp4 Videos/*.mov)
OUTPUT_DIR = test_frames

# Extract the base names of videos: video1.mp4 → video1
BASENAMES = $(notdir $(basename $(VIDEOS)))

DIRS := $(wildcard test_frames/*)             
PDFS := $(patsubst test_frames/%,%.pdf,$(DIRS))



.PHONY: all pdfs
all: keyframes
pdfs: $(PDFS)


../sam_vit_h_4b8939.pth: 
	@echo "Downloading SAM model weights"
	curl -L -o ../sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

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
%.jpg: $(FRAME_DIRS)

$(OUTPUT_DIR)/%/keyframes/: $(OUTPUT_DIR)/%
	@echo "Extracting keyframes from $< to $@"
	python keyframe_extraction.py --frames-root $(OUTPUT_DIR)


KEY_FRAMES_DIRS = $(addsuffix keyframes/,$(FRAME_DIRS))

.PHONY: keyframes
keyframes: $(FRAME_DIRS) $(KEY_FRAMES_DIRS)

$(OUTPUT_DIR)/%/left/: $(OUTPUT_DIR)/%/keyframes
	@echo "Cropping left side of keyframes in $< to $@"
	python batch_image_cropper.py --kwargs book=$* side=left

$(OUTPUT_DIR)/%/right/: $(OUTPUT_DIR)/%/keyframes
	@echo "Cropping right side of keyframes in $< to $@"
	python batch_image_cropper.py --kwargs book=$* side=right


$(OUTPUT_DIR)/%/cropped/: $(OUTPUT_DIR)/%/left $(OUTPUT_DIR)/%/right
	@echo "Merging cropped images in $@"
	mkdir -p $@
	mv $(word 1,$^)/* $@
	mv $(word 2,$^)/* $@

# add target to run streamlit_keyframes
.PHONY: streamlit_keyframes
streamlit_keyframes: keyframes streamlit_keyframes.py
	@echo "Running Streamlit app for keyframes"
	streamlit run streamlit_keyframes.py

# Marker-pdf extraction 
MARKDOWNS = $(FRAME_IMAGES:.jpg=.md)
%.md: %.jpg
	@echo "Generating Markdown for $<"
	convert $< -strip -interlace Plane -quality 100% intermediate.jpg
	marker_single intermediate.jpg $@
	@rm -f intermediate.jpg

.PHONY: generate_md
generate_md: $(MARKDOWNS)

# OCR with doctr
DOCTR_TXTS = $(FRAME_IMAGES:.jpg=_doctr.txt)
%_doctr.txt: %.jpg
	@echo "Generating doctr OCR text for $<"
	python ocr.py --kwargs frame=$< out_path=$@ ocr_mode=doctr

.PHONY: doctr_ocr
doctr_ocr: $(DOCTR_TXTS)

# OCR with easyOCR
EASYOCR_TXTS = $(FRAME_IMAGES:.jpg=_easyocr.txt)
%_easyocr.txt: %.jpg
	@echo "Generating easyOCR text for $<"
	python ocr.py --kwargs frame=$< out_path=$@ ocr_mode=easyocr

.PHONY: easy_ocr
easy_ocr: $(EASYOCR_TXTS)

.PHONY: install
install: 
	pip install -r requirements.txt


.PHONY: install_opencv
install_opencv:
	@if [ "$$(uname)" = "Darwin" ]; then \
		echo "Installing OpenCV using Homebrew on macOS"; \
		brew install opencv; \
	else \
		echo "Installing OpenCV using apt-get on Linux"; \
		apt-get update && apt-get install -y python3-opencv; \
	fi


.PHONY: clean
clean:
	rm -rf $(OUTPUT_DIR)

# create .venv
.venv:
	python3 -m venv .venv


# ---------------------------------------------------------------------------
# Pattern rule: “stem”.pdf ← test_frames/“stem”/*.jpg
# $*  → stem (directory name without path)
# $@ → target PDF
# ---------------------------------------------------------------------------
%.pdf: $(OUTPUT_DIR)/%/
	@echo "Building $@ from images in $<"
	# Collect .jpg and .png (empty if none); then call ImageMagick
	@imgs="$(wildcard $</cropped/*.jpg) $(wildcard $</cropped/*.png)"; \
		test -n "$$imgs" || { echo "No images found in $</"; exit 1; }; \
		magick $$imgs -resize '512x>' -quality 95 -interlace Plane $@
