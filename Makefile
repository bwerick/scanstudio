# ScanStudio Pipeline Makefile
#
# Usage:
#   make all VIDEO=recordings/mybook.mp4
#   make review VIDEO=recordings/mybook.mp4
#   make pdf VIDEO=recordings/mybook.mp4
#   make bw VIDEO=recordings/mybook.mp4
#   make clean VIDEO=recordings/mybook.mp4
#
# The VIDEO argument is required for all targets.
# The output directory is derived from the video filename:
#   recordings/mybook.mp4 → output/mybook/

# ── Check VIDEO argument ──────────────────────────────────────
ifndef VIDEO
$(error VIDEO is required. Usage: make all VIDEO=recordings/mybook.mp4)
endif

# ── Derived paths ─────────────────────────────────────────────
NAME     := $(basename $(notdir $(VIDEO)))
OUTDIR   := output/$(NAME)
SCRIPTS  := scripts

# ── Phase output markers ──────────────────────────────────────
MOTION    := $(OUTDIR)/motion/motion_signal.npy
PEAKS     := $(OUTDIR)/peaks/peaks.npy
KEYFRAMES := $(OUTDIR)/keyframes/keyframes.json
REVIEW    := $(OUTDIR)/review/final_keyframes.json
PAGES     := $(OUTDIR)/pages/pages.json
BW        := $(OUTDIR)/bw/bw_metadata.json
PDF       := $(OUTDIR)/pdf/book.pdf
PDF_BW    := $(OUTDIR)/pdf/book_bw.pdf

# ── Default parameters (override on command line) ─────────────
SAFETY_MARGIN ?= 0.005
BLOCK_SIZE    ?= 51
BW_OFFSET     ?= 10
JPEG_QUALITY  ?= 92

# ══════════════════════════════════════════════════════════════
# Targets
# ══════════════════════════════════════════════════════════════

.PHONY: all bw motion peaks keyframes review rereview split page-review binarize pdf pdf-bw clean help

help:
	@echo "ScanStudio Pipeline"
	@echo ""
	@echo "Usage: make <target> VIDEO=recordings/mybook.mp4"
	@echo ""
	@echo "Full pipeline:"
	@echo "  all           Run full pipeline (pauses at review steps)"
	@echo "  bw            Binarize pages and build B&W PDF"
	@echo ""
	@echo "Individual phases:"
	@echo "  motion        Phase 1: Compute motion signal"
	@echo "  peaks         Phase 2: Detect page turn peaks"
	@echo "  keyframes     Phase 3: Select keyframes"
	@echo "  review        Phase 4: Review keyframes (GUI)"
	@echo "  rereview      Phase 5: Re-review keyframes (GUI)"
	@echo "  split         Phase 6: Split spreads into pages"
	@echo "  page-review   Phase 7: Review page quality (GUI)"
	@echo "  binarize      Phase 8: Binarize pages to B&W"
	@echo "  pdf           Phase 9: Build PDF from color pages"
	@echo "  pdf-bw        Phase 9: Build PDF from B&W pages"
	@echo ""
	@echo "Utilities:"
	@echo "  clean         Delete all outputs for this video"
	@echo "  help          Show this message"
	@echo ""
	@echo "Parameters (override with VAR=value):"
	@echo "  SAFETY_MARGIN  Crop safety margin (default: $(SAFETY_MARGIN))"
	@echo "  BLOCK_SIZE     Binarize block size (default: $(BLOCK_SIZE))"
	@echo "  BW_OFFSET      Binarize offset (default: $(BW_OFFSET))"
	@echo "  JPEG_QUALITY   JPEG quality (default: $(JPEG_QUALITY))"
	@echo ""
	@echo "Current video: $(VIDEO)"
	@echo "Output dir:    $(OUTDIR)"

# ── Full pipeline ─────────────────────────────────────────────

all: motion peaks keyframes review split page-review pdf
	@echo ""
	@echo "════════════════════════════════════════════════════"
	@echo "  Pipeline complete: $(PDF)"
	@echo "════════════════════════════════════════════════════"

bw: binarize pdf-bw
	@echo ""
	@echo "════════════════════════════════════════════════════"
	@echo "  B&W pipeline complete: $(PDF_BW)"
	@echo "════════════════════════════════════════════════════"

# ── Phase 1: Motion signal ────────────────────────────────────

motion: $(MOTION)

$(MOTION):
	python $(SCRIPTS)/p1_motion_signal.py $(VIDEO)

# ── Phase 2: Detect peaks ─────────────────────────────────────

peaks: $(PEAKS)

$(PEAKS): $(MOTION)
	python $(SCRIPTS)/p2_detect_peaks.py $(OUTDIR)

# ── Phase 3: Select keyframes ─────────────────────────────────

keyframes: $(KEYFRAMES)

$(KEYFRAMES): $(PEAKS)
	python $(SCRIPTS)/p3_select_keyframes.py $(OUTDIR) $(VIDEO)

# ── Phase 4: Review keyframes (interactive) ───────────────────

review: $(REVIEW)

$(REVIEW): $(KEYFRAMES)
	python $(SCRIPTS)/p4_review_keyframes.py $(OUTDIR) $(VIDEO)

# ── Phase 5: Re-review ────────────────────────────────────────

rereview:
	python $(SCRIPTS)/p5_prep_rereview.py $(OUTDIR) $(VIDEO)

# ── Phase 6: Split pages ──────────────────────────────────────

split: $(PAGES)

$(PAGES): $(REVIEW)
	python $(SCRIPTS)/p6_split_pages.py $(OUTDIR) --safety-margin $(SAFETY_MARGIN)

# ── Phase 7: Page quality review (interactive) ────────────────

page-review: $(PAGES)
	python $(SCRIPTS)/p7_review_pages.py $(OUTDIR)

# ── Phase 8: Binarize ─────────────────────────────────────────

binarize: $(BW)

$(BW): $(PAGES)
	python $(SCRIPTS)/p8_binarize.py $(OUTDIR) --block-size $(BLOCK_SIZE) --offset $(BW_OFFSET)

# ── Phase 9: Build PDF ────────────────────────────────────────

pdf: $(PDF)

$(PDF): $(PAGES)
	python $(SCRIPTS)/p9_build_pdf.py $(OUTDIR)

pdf-bw: $(PDF_BW)

$(PDF_BW): $(BW)
	python $(SCRIPTS)/p9_build_pdf.py $(OUTDIR) --source bw --pdf-name book_bw.pdf

# ── Clean ─────────────────────────────────────────────────────

clean:
	@echo "Removing $(OUTDIR)/"
	rm -rf $(OUTDIR)
	@echo "Done."