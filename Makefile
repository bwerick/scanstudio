# ScanStudio Pipeline Makefile
#
# Usage (live):
#   make live NAME=mybook
#   make finish VIDEO=recordings/mybook.mp4 [MODE=double]
#
# Usage (batch):
#   make all VIDEO=recordings/mybook.mp4

ifeq ($(filter install help live clean tkinter probe-camera,$(MAKECMDGOALS)),)
ifndef VIDEO
$(error VIDEO is required. Usage: make all VIDEO=recordings/mybook.mp4)
endif
endif

# NAME derives from VIDEO, but can be set directly (e.g. for clean after a live run).
NAME     ?= $(basename $(notdir $(VIDEO)))
OUTDIR   := output/$(NAME)
SCRIPTS  := scripts

# Phase output markers (new directory structure)
MOTION    := $(OUTDIR)/data/motion_signal.npy
PEAKS     := $(OUTDIR)/data/peaks.npy
KEYFRAMES := $(OUTDIR)/json/keyframes.json
PAGES     := $(OUTDIR)/json/pages.json
BW_META   := $(OUTDIR)/json/bw_metadata.json
PDF       := $(OUTDIR)/pdf/$(NAME).pdf
PDF_BW    := $(OUTDIR)/pdf/$(NAME)_bw.pdf

# Default parameters
SAFETY_MARGIN ?= 0.005
BLOCK_SIZE    ?= 51
BW_OFFSET     ?= 10
BW_METHOD     ?= sauvola
BW_UPSCALE    ?= 2
BW_K          ?= 0.2
MODE          ?= double
# 'auto' picks whichever camera delivers the requested 4K mode (USB indices
# shift on reconnect). Set CAMERA=<n> to force one; `make probe-camera` lists them.
CAMERA        ?= auto
SETTLE        ?= 2.0
TURN          ?= 5.0
SETTLE_TIME   ?= 0.1
PREVIEW_HEIGHT ?= 1080

.PHONY: all bw live finish motion peaks keyframes review crop split page-review binarize pdf pdf-bw clean install tkinter probe-camera help

help:
	@echo "ScanStudio Pipeline"
	@echo ""
	@echo "Usage: make <target> VIDEO=recordings/mybook.mp4"
	@echo ""
	@echo "  all           Full pipeline (pauses at review)"
	@echo "  bw            Binarize + B&W PDF"
	@echo "  live          P0: Live webcam capture (make live NAME=mybook)"
	@echo "  probe-camera  List camera indices and which one delivers 4K"
	@echo "  finish        P4-P9 back half (run after 'live')"
	@echo ""
	@echo "  motion        P1: Motion signal"
	@echo "  peaks         P2: Detect peaks"
	@echo "  keyframes     P3: Select keyframes"
	@echo "  review        P4: Review keyframes (GUI, reentrant)"
	@echo "  crop          P5: Crop keyframes"
	@echo "  split         P6: Split into pages"
	@echo "  page-review   P7: Page quality review (GUI)"
	@echo "  binarize      P8: Binarize to B&W"
	@echo "  pdf           P9: Build PDF"
	@echo "  pdf-bw        P9: Build B&W PDF"
	@echo ""
	@echo "  clean         Delete output/<NAME>/ (VIDEO= or NAME=; keeps recording)"
	@echo ""
	@echo "  SAFETY_MARGIN=$(SAFETY_MARGIN)  BLOCK_SIZE=$(BLOCK_SIZE)  BW_OFFSET=$(BW_OFFSET)"
	@echo "  BW_METHOD=$(BW_METHOD) (sauvola|adaptive)  BW_UPSCALE=$(BW_UPSCALE)  BW_K=$(BW_K) (higher=thinner)"
	@echo "  MODE=$(MODE)  (double=book spreads, single=loose docs)"
	@echo "  live: CAMERA=$(CAMERA)  SETTLE=$(SETTLE)  TURN=$(TURN)  SETTLE_TIME=$(SETTLE_TIME)  PREVIEW_HEIGHT=$(PREVIEW_HEIGHT)"

all: motion peaks keyframes finish
	@echo "Pipeline complete: $(PDF)"

# Back half (P4-P9): review, crop, split, page-review, build PDF.
# Use after 'live' (or run individually). Pauses at P4 and P7.
finish: review crop split page-review pdf
	@echo "Pipeline complete: $(PDF)"

# Live capture (P0): record the webcam and auto-select keyframes in real time.
# Replaces P1-P3; produces the recording + the same artifacts they would.
#   make live NAME=mybook [CAMERA=1]
#   make finish VIDEO=recordings/mybook.mp4
live:
ifndef NAME
	$(error NAME is required. Usage: make live NAME=mybook)
endif
	@mkdir -p recordings
	python $(SCRIPTS)/p0_live_capture.py output/$(NAME) recordings/$(NAME).mp4 \
		--camera $(CAMERA) --settle-threshold $(SETTLE) --turn-threshold $(TURN) \
		--settle-time $(SETTLE_TIME) --preview-height $(PREVIEW_HEIGHT)
	@echo "Live capture done. Continue with: make finish VIDEO=recordings/$(NAME).mp4"

bw: binarize pdf-bw
	@echo "B&W pipeline complete: $(PDF_BW)"

motion: $(MOTION)
$(MOTION):
	python $(SCRIPTS)/p1_motion_signal.py $(VIDEO)

peaks: $(PEAKS)
$(PEAKS): $(MOTION)
	python $(SCRIPTS)/p2_detect_peaks.py $(OUTDIR)

keyframes: $(KEYFRAMES)
$(KEYFRAMES): $(PEAKS)
	python $(SCRIPTS)/p3_select_keyframes.py $(OUTDIR) $(VIDEO)

review: $(KEYFRAMES)
	python $(SCRIPTS)/p4_review_keyframes.py $(OUTDIR) $(VIDEO) --mode $(MODE)

crop: $(KEYFRAMES)
	python $(SCRIPTS)/p5_crop.py $(OUTDIR) --mode $(MODE) --safety-margin $(SAFETY_MARGIN)

split: $(PAGES)
$(PAGES): $(KEYFRAMES)
	python $(SCRIPTS)/p6_split_pages.py $(OUTDIR) --mode $(MODE)

page-review: $(PAGES)
	python $(SCRIPTS)/p7_review_pages.py $(OUTDIR)

binarize: $(BW_META)
$(BW_META): $(PAGES)
	python $(SCRIPTS)/p8_binarize.py $(OUTDIR) --method $(BW_METHOD) --block-size $(BLOCK_SIZE) --offset $(BW_OFFSET) --upscale $(BW_UPSCALE) --sauvola-k $(BW_K)

pdf: $(PDF)
$(PDF): $(PAGES)
	python $(SCRIPTS)/p9_build_pdf.py $(OUTDIR)

pdf-bw: $(PDF_BW)
$(PDF_BW): $(BW_META)
	python $(SCRIPTS)/p9_build_pdf.py $(OUTDIR) --source bw --pdf-name $(NAME)_bw.pdf

probe-camera:
	python $(SCRIPTS)/probe_camera.py

install: tkinter
	pip install -r requirements.txt

# tkinter is a system package (not pip-installable). The review GUIs (P4/P7)
# need it. On macOS install the matching Homebrew package for the active Python.
tkinter:
	@python -c "import tkinter" 2>/dev/null && echo "tkinter OK" || { \
		echo "tkinter missing."; \
		if [ "$$(uname)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then \
			ver=$$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"); \
			echo "Installing python-tk@$$ver via Homebrew..."; \
			brew install python-tk@$$ver; \
		else \
			echo "Install Tk for your Python, e.g. apt install python3-tk (Debian/Ubuntu)."; \
			exit 1; \
		fi; \
	}

clean:
ifeq ($(strip $(NAME)),)
	$(error VIDEO or NAME required. Usage: make clean VIDEO=recordings/mybook.mp4  (or NAME=mybook))
endif
	@echo "Removing $(OUTDIR)/"
	rm -rf $(OUTDIR)
	@echo "Recording kept: recordings/$(NAME).mp4 (remove manually to fully undo the run)"