# ScanStudio Pipeline Makefile
#
# Usage:
#   make all VIDEO=recordings/mybook.mp4
#   make review VIDEO=recordings/mybook.mp4
#   make pdf VIDEO=recordings/mybook.mp4

ifndef VIDEO
$(error VIDEO is required. Usage: make all VIDEO=recordings/mybook.mp4)
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

.PHONY: all bw motion peaks keyframes review crop split page-review binarize pdf pdf-bw clean help

help:
	@echo "ScanStudio Pipeline"
	@echo ""
	@echo "Usage: make <target> VIDEO=recordings/mybook.mp4"
	@echo ""
	@echo "  all           Full pipeline (pauses at review)"
	@echo "  bw            Binarize + B&W PDF"
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

all: motion peaks keyframes review crop split page-review pdf
	@echo "Pipeline complete: $(PDF)"

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

clean:
	@echo "Removing $(OUTDIR)/"
	rm -rf $(OUTDIR)