# ============================
# Clean pipeline Makefile
# ============================

VIDEO_DIR   := Videos
OUTPUT_DIR  := test_frames

# Discover videos; derive BOOK names from filenames.
VIDEOS := $(wildcard $(VIDEO_DIR)/*.mp4 $(VIDEO_DIR)/*.mov)
BOOKS  := $(notdir $(basename $(VIDEOS)))

# Stamp targets per book
FRAMES_STAMPS    := $(addprefix $(OUTPUT_DIR)/,$(addsuffix /.frames.stamp,$(BOOKS)))
KEYFRAMES_STAMPS := $(addprefix $(OUTPUT_DIR)/,$(addsuffix /keyframes/.keyframes.stamp,$(BOOKS)))
LEFT_STAMPS      := $(addprefix $(OUTPUT_DIR)/,$(addsuffix /left/.left.stamp,$(BOOKS)))
RIGHT_STAMPS     := $(addprefix $(OUTPUT_DIR)/,$(addsuffix /right/.right.stamp,$(BOOKS)))
CROPPED_STAMPS   := $(addprefix $(OUTPUT_DIR)/,$(addsuffix /cropped/.cropped.stamp,$(BOOKS)))

PDFS := $(addsuffix .pdf,$(BOOKS))

# Safer make defaults
.DELETE_ON_ERROR:
.SUFFIXES:

# Convenience: print vars with `make print-BOOKS` etc.
.PHONY: print-%
print-%:
	@echo '$*=$($*)'

# Default target
.PHONY: all
all: keyframes

# ----------------------------
# Ensure base output directory exists
# ----------------------------
$(OUTPUT_DIR):
	@mkdir -p $@

# ----------------------------
# Model weights (unchanged)
# ----------------------------
../sam_vit_h_4b8939.pth:
	@echo "Downloading SAM model weights"
	curl -L -o $@ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# ----------------------------
# Frames extraction (single rule for mp4/mov)
# Uses secondary expansion to pick whichever file exists.
# ----------------------------
.SECONDEXPANSION:

$(OUTPUT_DIR)/%/.frames.stamp: frameextraction.py | $(OUTPUT_DIR)
	$(eval VIDEO_FILE := $(firstword $(wildcard $(VIDEO_DIR)/$*.mp4 $(VIDEO_DIR)/$*.mov)))
	@test -n "$(VIDEO_FILE)" || { echo "ERROR: no video found for '$*' in $(VIDEO_DIR)/"; exit 1; }
	@mkdir -p "$(OUTPUT_DIR)/$*"
	@echo "Extracting frames for $* from $(VIDEO_FILE)"
	python frameextraction.py --kwargs out_path=$(OUTPUT_DIR) video=$(VIDEO_FILE)
	@ls -1 "$(OUTPUT_DIR)/$*"/*.jpg >/dev/null 2>&1 || { \
		echo "ERROR: no frames (*.jpg) produced in $(OUTPUT_DIR)/$*/"; \
		exit 1; \
	}
	@touch $@

.PHONY: frames
frames: $(FRAMES_STAMPS)

# ----------------------------
# Keyframes extraction
# Note: if your python script processes all books at once, that's fine;
# we still validate per-book outputs so stamps remain truthful.
# ----------------------------
$(OUTPUT_DIR)/%/keyframes/.keyframes.stamp: $(OUTPUT_DIR)/%/.frames.stamp keyframe_extraction.py
	@mkdir -p "$(OUTPUT_DIR)/$*/keyframes"
	@echo "Extracting keyframes for $*"
	python keyframe_extraction.py --frames-root $(OUTPUT_DIR)
	@ls -1 "$(OUTPUT_DIR)/$*/keyframes"/*.jpg >/dev/null 2>&1 || { \
		echo "ERROR: no keyframes (*.jpg) produced in $(OUTPUT_DIR)/$*/keyframes/"; \
		exit 1; \
	}
	@touch $@

.PHONY: keyframes
keyframes: $(KEYFRAMES_STAMPS)

# ----------------------------
# Left / Right pages
# ----------------------------
$(OUTPUT_DIR)/%/left/.left.stamp: $(OUTPUT_DIR)/%/keyframes/.keyframes.stamp batch_image_cropper.py
	@mkdir -p "$(OUTPUT_DIR)/$*/left"
	@echo "Cropping LEFT pages for $*"
	python batch_image_cropper.py --kwargs book=$* side=left
	@ls -1 "$(OUTPUT_DIR)/$*/left"/*.jpg >/dev/null 2>&1 || { \
		echo "ERROR: no LEFT crops (*.jpg) produced in $(OUTPUT_DIR)/$*/left/"; \
		exit 1; \
	}
	@touch $@

$(OUTPUT_DIR)/%/right/.right.stamp: $(OUTPUT_DIR)/%/keyframes/.keyframes.stamp batch_image_cropper.py
	@mkdir -p "$(OUTPUT_DIR)/$*/right"
	@echo "Cropping RIGHT pages for $*"
	python batch_image_cropper.py --kwargs book=$* side=right
	@ls -1 "$(OUTPUT_DIR)/$*/right"/*.jpg >/dev/null 2>&1 || { \
		echo "ERROR: no RIGHT crops (*.jpg) produced in $(OUTPUT_DIR)/$*/right/"; \
		exit 1; \
	}
	@touch $@

.PHONY: left-pages right-pages
left-pages:  $(LEFT_STAMPS)
right-pages: $(RIGHT_STAMPS)

# ----------------------------
# Cropped staging (merge left+right into cropped/)
# This moves files. If you prefer COPY instead of MOVE, tell me.
# ----------------------------
$(OUTPUT_DIR)/%/cropped/.cropped.stamp: $(OUTPUT_DIR)/%/left/.left.stamp $(OUTPUT_DIR)/%/right/.right.stamp
	@mkdir -p "$(OUTPUT_DIR)/$*/cropped"
	@echo "Staging cropped images for $*"
	@rm -rf "$(OUTPUT_DIR)/$*/cropped/"*
	@mv "$(OUTPUT_DIR)/$*/left/"*  "$(OUTPUT_DIR)/$*/cropped/" 2>/dev/null || true
	@mv "$(OUTPUT_DIR)/$*/right/"* "$(OUTPUT_DIR)/$*/cropped/" 2>/dev/null || true
	@ls -1 "$(OUTPUT_DIR)/$*/cropped"/*.jpg >/dev/null 2>&1 || { \
		echo "ERROR: no staged crops (*.jpg) in $(OUTPUT_DIR)/$*/cropped/"; \
		exit 1; \
	}
	@touch $@

.PHONY: cropped
cropped: $(CROPPED_STAMPS)

# ----------------------------
# Final PDFs: make <book>.pdf
# ----------------------------
%.pdf: $(OUTPUT_DIR)/%/cropped/.cropped.stamp
	@echo "Building $@ from $(OUTPUT_DIR)/$*/cropped/"
	@set -e; \
	cd "$(OUTPUT_DIR)/$*/cropped"; \
	shopt -s nullglob; \
	images=( *.jpg *.png ); \
	[ "$${#images[@]}" -gt 0 ] || { echo "ERROR: no images in $(OUTPUT_DIR)/$*/cropped/"; exit 1; }; \
	magick "$${images[@]}" -resize '512x>' -quality 95 -interlace Plane "$(abspath $@)"

.PHONY: pdfs
pdfs: $(PDFS)

# ----------------------------
# Streamlit helper
# ----------------------------
.PHONY: streamlit_keyframes
streamlit_keyframes: keyframes streamlit_keyframes.py
	@echo "Running Streamlit app for keyframes"
	streamlit run streamlit_keyframes.py

# ----------------------------
# Install / clean
# ----------------------------
.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: clean
clean:
	rm -rf "$(OUTPUT_DIR)" *.pdf

# ----------------------------
# Force rebuild helpers
# ----------------------------
.PHONY: force-frames force-keyframes force-left force-right force-cropped
force-frames:
	@rm -f $(FRAMES_STAMPS)

force-keyframes:
	@rm -f $(KEYFRAMES_STAMPS)

force-left:
	@rm -f $(LEFT_STAMPS)

force-right:
	@rm -f $(RIGHT_STAMPS)

force-cropped:
	@rm -f $(CROPPED_STAMPS)
