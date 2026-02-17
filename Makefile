# ============================
# Clean pipeline Makefile
# ============================

VIDEO_DIR   := Videos
OUTPUT_DIR  := test_frames

# Recipes use bash features (arrays, shopt, pipefail).
SHELL := /bin/bash

# NOTE ABOUT WHITESPACE IN FILENAMES
# ---------------------------------
# GNU Make cannot safely keep filenames that contain spaces inside variables like
# $(wildcard ...) or lists such as BOOKS := a b c.
#
# If you have videos with spaces (e.g., "African Founders.mp4"), this Makefile
# avoids putting those names into make "word" lists. Instead, targets like
# `make frames` and `make keyframes` iterate over files using a null-delimited
# shell loop.
#
# For per-book operations (crop/pdf), use BOOK="..." variables, e.g.:
#   make frames-one BOOK="African Founders"
#   make keyframes-one BOOK="African Founders"
#   make left BOOK="African Founders"
#   make pdf BOOK="African Founders"

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
# Helpers
# ----------------------------
.PHONY: list-videos list-books
list-videos:
	@find "$(VIDEO_DIR)" -maxdepth 1 \( -type f -o -type l \) \( -iname '*.mp4' -o -iname '*.mov' \) -print

list-books:
	@find "$(VIDEO_DIR)" -maxdepth 1 \( -type f -o -type l \) \( -iname '*.mp4' -o -iname '*.mov' \) -print0 | \
		while IFS= read -r -d '' p; do \
			b=$$(basename "$$p"); \
			echo "$${b%.*}"; \
		done

.PHONY: guard-book
guard-book:
	@test -n "$(BOOK)" || { echo "ERROR: set BOOK=\"...\""; exit 2; }

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


# ----------------------------
# Frames extraction (whitespace-safe, all videos)
# ----------------------------
$(OUTPUT_DIR)/.frames_all.stamp: frameextraction.py | $(OUTPUT_DIR)
	@set -euo pipefail; \
	find "$(VIDEO_DIR)" -maxdepth 1 \( -type f -o -type l \) \( -iname '*.mp4' -o -iname '*.mov' \) -print -quit | grep -q . || { \
		echo "ERROR: no videos found in $(VIDEO_DIR)/"; \
		exit 1; \
	}; \
	find "$(VIDEO_DIR)" -maxdepth 1 \( -type f -o -type l \) \( -iname '*.mp4' -o -iname '*.mov' \) -print0 | \
	while IFS= read -r -d '' video; do \
		stem=$$(basename "$$video"); stem=$${stem%.*}; \
		echo "Extracting frames for $$stem from $$video"; \
		python frameextraction.py --kwargs out_path="$(OUTPUT_DIR)" video="$$video"; \
		ls -1 "$(OUTPUT_DIR)/$$stem"/*.jpg >/dev/null 2>&1 || { \
			echo "ERROR: no frames (*.jpg) produced in $(OUTPUT_DIR)/$$stem/"; \
			exit 1; \
		}; \
		touch "$(OUTPUT_DIR)/$$stem/.frames.stamp"; \
	done
	@touch $@

.PHONY: frames
frames: $(OUTPUT_DIR)/.frames_all.stamp

# Single-video frames (whitespace-safe)
.PHONY: frames-one
frames-one: guard-book frameextraction.py | $(OUTPUT_DIR)
	@set -euo pipefail; \
	video=""; \
	for ext in mp4 mov MP4 MOV; do \
		candidate="$(VIDEO_DIR)/$(BOOK).$$ext"; \
		if [ -f "$$candidate" ]; then video="$$candidate"; break; fi; \
	done; \
	[ -n "$$video" ] || { echo "ERROR: no video found for '$(BOOK)' in $(VIDEO_DIR)/"; exit 1; }; \
	echo "Extracting frames for $(BOOK) from $$video"; \
	python frameextraction.py --kwargs out_path="$(OUTPUT_DIR)" video="$$video"; \
	ls -1 "$(OUTPUT_DIR)/$(BOOK)"/*.jpg >/dev/null 2>&1 || { \
		echo "ERROR: no frames (*.jpg) produced in $(OUTPUT_DIR)/$(BOOK)/"; \
		exit 1; \
	}; \
	touch "$(OUTPUT_DIR)/$(BOOK)/.frames.stamp"


# ----------------------------
# Keyframes extraction
# Note: if your python script processes all books at once, that's fine;
# we still validate per-book outputs so stamps remain truthful.
# ----------------------------

# ----------------------------
# Keyframes extraction (whitespace-safe, all books)
# ----------------------------
$(OUTPUT_DIR)/.keyframes_all.stamp: $(OUTPUT_DIR)/.frames_all.stamp keyframe_extraction.py
	@set -euo pipefail; \
	echo "Extracting keyframes for all frame directories under $(OUTPUT_DIR)"; \
	python keyframe_extraction.py --frames-root "$(OUTPUT_DIR)"; \
	find "$(OUTPUT_DIR)" -mindepth 1 -maxdepth 1 -type d -print0 | \
	while IFS= read -r -d '' d; do \
		[ -d "$$d/keyframes" ] || continue; \
		ls -1 "$$d/keyframes"/*.jpg >/dev/null 2>&1 || { \
			echo "ERROR: no keyframes (*.jpg) produced in $$d/keyframes/"; \
			exit 1; \
		}; \
		touch "$$d/keyframes/.keyframes.stamp"; \
	done
	@touch $@

.PHONY: keyframes
keyframes: $(OUTPUT_DIR)/.keyframes_all.stamp

# Single-book keyframes (whitespace-safe)
.PHONY: keyframes-one
keyframes-one: guard-book keyframe_extraction.py
	@set -euo pipefail; \
	[ -d "$(OUTPUT_DIR)/$(BOOK)" ] || { echo "ERROR: frames directory not found: $(OUTPUT_DIR)/$(BOOK)"; exit 1; }; \
	echo "Extracting keyframes for $(BOOK)"; \
	python keyframe_extraction.py --frames-dir "$(OUTPUT_DIR)/$(BOOK)"; \
	ls -1 "$(OUTPUT_DIR)/$(BOOK)/keyframes"/*.jpg >/dev/null 2>&1 || { \
		echo "ERROR: no keyframes (*.jpg) produced in $(OUTPUT_DIR)/$(BOOK)/keyframes/"; \
		exit 1; \
	}; \
	touch "$(OUTPUT_DIR)/$(BOOK)/keyframes/.keyframes.stamp"


# ----------------------------
# Left / Right pages
# ----------------------------

# ----------------------------
# Left / Right pages (use BOOK="..." for whitespace-safe book names)
# ----------------------------
.PHONY: left right
left: guard-book batch_image_cropper.py
	@set -euo pipefail; \
	[ -d "$(OUTPUT_DIR)/$(BOOK)/keyframes" ] || { echo "ERROR: missing keyframes dir: $(OUTPUT_DIR)/$(BOOK)/keyframes"; exit 1; }; \
	mkdir -p "$(OUTPUT_DIR)/$(BOOK)/left"; \
	echo "Cropping LEFT pages for $(BOOK)"; \
	python batch_image_cropper.py --kwargs "book=$(BOOK)" side=left; \
	ls -1 "$(OUTPUT_DIR)/$(BOOK)/left"/*.jpg >/dev/null 2>&1 || { \
		echo "ERROR: no LEFT crops (*.jpg) produced in $(OUTPUT_DIR)/$(BOOK)/left/"; \
		exit 1; \
	}; \
	touch "$(OUTPUT_DIR)/$(BOOK)/left/.left.stamp"

right: guard-book batch_image_cropper.py
	@set -euo pipefail; \
	[ -d "$(OUTPUT_DIR)/$(BOOK)/keyframes" ] || { echo "ERROR: missing keyframes dir: $(OUTPUT_DIR)/$(BOOK)/keyframes"; exit 1; }; \
	mkdir -p "$(OUTPUT_DIR)/$(BOOK)/right"; \
	echo "Cropping RIGHT pages for $(BOOK)"; \
	python batch_image_cropper.py --kwargs "book=$(BOOK)" side=right; \
	ls -1 "$(OUTPUT_DIR)/$(BOOK)/right"/*.jpg >/dev/null 2>&1 || { \
		echo "ERROR: no RIGHT crops (*.jpg) produced in $(OUTPUT_DIR)/$(BOOK)/right/"; \
		exit 1; \
	}; \
	touch "$(OUTPUT_DIR)/$(BOOK)/right/.right.stamp"

# ----------------------------
# Cropped staging (merge left+right into cropped/)
# This moves files. If you prefer COPY instead of MOVE, tell me.
# ----------------------------

# ----------------------------
# Cropped staging (use BOOK="..." for whitespace-safe book names)
# ----------------------------
.PHONY: cropped
cropped: guard-book
	@set -euo pipefail; \
	[ -d "$(OUTPUT_DIR)/$(BOOK)/left" ] || { echo "ERROR: missing left dir: $(OUTPUT_DIR)/$(BOOK)/left"; exit 1; }; \
	[ -d "$(OUTPUT_DIR)/$(BOOK)/right" ] || { echo "ERROR: missing right dir: $(OUTPUT_DIR)/$(BOOK)/right"; exit 1; }; \
	mkdir -p "$(OUTPUT_DIR)/$(BOOK)/cropped"; \
	echo "Staging cropped images for $(BOOK)"; \
	rm -rf "$(OUTPUT_DIR)/$(BOOK)/cropped/"*; \
	mv "$(OUTPUT_DIR)/$(BOOK)/left/"*  "$(OUTPUT_DIR)/$(BOOK)/cropped/" 2>/dev/null || true; \
	mv "$(OUTPUT_DIR)/$(BOOK)/right/"* "$(OUTPUT_DIR)/$(BOOK)/cropped/" 2>/dev/null || true; \
	ls -1 "$(OUTPUT_DIR)/$(BOOK)/cropped"/*.jpg >/dev/null 2>&1 || { \
		echo "ERROR: no staged crops (*.jpg) in $(OUTPUT_DIR)/$(BOOK)/cropped/"; \
		exit 1; \
	}; \
	touch "$(OUTPUT_DIR)/$(BOOK)/cropped/.cropped.stamp"

# ----------------------------
# Final PDFs: make <book>.pdf
# ----------------------------

# ----------------------------
# Final PDF (use BOOK="..." for whitespace-safe book names)
# ----------------------------
.PHONY: pdf
pdf: guard-book
	@set -euo pipefail; \
	stamp="$(OUTPUT_DIR)/$(BOOK)/cropped/.cropped.stamp"; \
	if [ ! -f "$$stamp" ]; then \
		echo "Cropped stamp not found; running: make cropped BOOK=\"$(BOOK)\""; \
		$(MAKE) cropped BOOK="$(BOOK)"; \
	fi; \
	out="$(BOOK).pdf"; \
	out_abs="$(CURDIR)/$$out"; \
	echo "Building $$out from $(OUTPUT_DIR)/$(BOOK)/cropped/"; \
	cd "$(OUTPUT_DIR)/$(BOOK)/cropped"; \
	shopt -s nullglob; \
	images=( *.jpg *.png ); \
	[ "$${#images[@]}" -gt 0 ] || { echo "ERROR: no images in $(OUTPUT_DIR)/$(BOOK)/cropped/"; exit 1; }; \
	magick "$${images[@]}" -resize '512x>' -quality 95 -interlace Plane "$$out_abs"

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

.PHONY: force-frames force-keyframes
force-frames:
	@rm -f "$(OUTPUT_DIR)/.frames_all.stamp"
	@find "$(OUTPUT_DIR)" -mindepth 2 -maxdepth 2 -name .frames.stamp -print0 2>/dev/null | xargs -0 rm -f || true

force-keyframes:
	@rm -f "$(OUTPUT_DIR)/.keyframes_all.stamp"
	@find "$(OUTPUT_DIR)" -mindepth 3 -maxdepth 3 -path '*/keyframes/.keyframes.stamp' -print0 2>/dev/null | xargs -0 rm -f || true
