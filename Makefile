# ScanStudio Pipeline Makefile
#
# Usage:
#   make all VIDEO=recordings/mybook.mp4
#   make review VIDEO=recordings/mybook.mp4
#   make pdf VIDEO=recordings/mybook.mp4

ifeq ($(filter install help live,$(MAKECMDGOALS)),)
ifndef VIDEO
$(error VIDEO is required. Usage: make all VIDEO=recordings/mybook.mp4)
endif
endif

NAME     := $(basename $(notdir $(VIDEO)))
OUTDIR   := output/$(NAME)
SCRIPTS  := scripts

# Phase output markers (new directory structure)
MOTION    := $(OUTDIR)/data/motion_signal.npy
PEAKS     := $(OUTDIR)/data/peaks.npy
KEYFRAMES := $(OUTDIR)/json/keyframes.json
PAGES     := $(OUTDIR)/json/pages.json
BW_META   := $(OUTDIR)/json/bw_metadata.json
PDF       := $(OUTDIR)/pdf/book.pdf
PDF_BW    := $(OUTDIR)/pdf/book_bw.pdf

# Default parameters
SAFETY_MARGIN ?= 0.005
BLOCK_SIZE    ?= 51
BW_OFFSET     ?= 10
MODE          ?= double
CAMERA        ?= 0
SETTLE        ?= 2.0
TURN          ?= 5.0
SETTLE_TIME   ?= 0.4

.PHONY: all bw live finish motion peaks keyframes review crop split page-review binarize pdf pdf-bw clean install help

help:
	@echo "ScanStudio Pipeline"
	@echo ""
	@echo "Usage: make <target> VIDEO=recordings/mybook.mp4"
	@echo ""
	@echo "  all           Full pipeline (pauses at review)"
	@echo "  bw            Binarize + B&W PDF"
	@echo "  live          P0: Live webcam capture (make live NAME=mybook)"
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
	@echo "  clean         Delete all outputs"
	@echo ""
	@echo "  SAFETY_MARGIN=$(SAFETY_MARGIN)  BLOCK_SIZE=$(BLOCK_SIZE)  BW_OFFSET=$(BW_OFFSET)"
	@echo "  MODE=$(MODE)  (double=book spreads, single=loose docs)"
	@echo "  live: CAMERA=$(CAMERA)  SETTLE=$(SETTLE)  TURN=$(TURN)  SETTLE_TIME=$(SETTLE_TIME)"

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
		--settle-time $(SETTLE_TIME)
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
	python $(SCRIPTS)/p4_review_keyframes.py $(OUTDIR) $(VIDEO)

crop: $(KEYFRAMES)
	python $(SCRIPTS)/p5_crop.py $(OUTDIR) --mode $(MODE) --safety-margin $(SAFETY_MARGIN)

split: $(PAGES)
$(PAGES): $(KEYFRAMES)
	python $(SCRIPTS)/p6_split_pages.py $(OUTDIR) --mode $(MODE)

page-review: $(PAGES)
	python $(SCRIPTS)/p7_review_pages.py $(OUTDIR)

binarize: $(BW_META)
$(BW_META): $(PAGES)
	python $(SCRIPTS)/p8_binarize.py $(OUTDIR) --block-size $(BLOCK_SIZE) --offset $(BW_OFFSET)

pdf: $(PDF)
$(PDF): $(PAGES)
	python $(SCRIPTS)/p9_build_pdf.py $(OUTDIR)

pdf-bw: $(PDF_BW)
$(PDF_BW): $(BW_META)
	python $(SCRIPTS)/p9_build_pdf.py $(OUTDIR) --source bw --pdf-name book_bw.pdf

install:
	pip install -r requirements.txt

clean:
	@echo "Removing $(OUTDIR)/"
	rm -rf $(OUTDIR)