# ScanStudio Pipeline Makefile
#
# Usage (live):
#   make live NAME=mybook
#   make finish VIDEO=recordings/mybook.mp4 [MODE=double]
#
# Usage (batch):
#   make all VIDEO=recordings/mybook.mp4

ifeq ($(filter install install-legacy help live live-web clean tkinter ffmpeg probe-camera test,$(MAKECMDGOALS)),)
ifeq ($(strip $(VIDEO)),)
$(error VIDEO is required. Usage: make all VIDEO=recordings/mybook.mp4)
endif
endif

# NAME derives from VIDEO, but can be set directly (e.g. for clean after a live run).
NAME     ?= $(basename $(notdir $(VIDEO)))
OUTDIR   := output/$(NAME)
SCRIPTS  := scripts

# Python interpreter. Defaults to the project venv; override with PYTHON=python3.
PYTHON   ?= .venv/bin/python

# When using the default interpreter, `make install` creates the venv on demand
# (see the .venv/bin/python rule). An explicit PYTHON=... override is left alone.
ifeq ($(PYTHON),.venv/bin/python)
VENV := .venv/bin/python
endif

# Fail early with a clear message if the interpreter is missing, instead of a
# cryptic "No such file or directory" from every recipe. Skipped for targets
# that don't need Python (help, clean) and for install (which bootstraps it).
ifeq ($(filter install install-legacy help clean ffmpeg,$(MAKECMDGOALS)),)
ifeq ($(shell command -v $(PYTHON) >/dev/null 2>&1 && echo ok),)
$(error Python interpreter '$(PYTHON)' not found. Run 'make install' to create the venv, or override PYTHON=python3)
endif
endif

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
# 'auto' also writes one PDF per document when review marked document starts
# (P4's Doc Start / P7's First Page). The combined <NAME>.pdf is always written.
SPLIT_DOCS    ?= auto
# 'auto' picks whichever camera delivers the requested 4K mode (USB indices
# shift on reconnect). Set CAMERA=<n> to force one; `make probe-camera` lists them.
CAMERA        ?= auto
SETTLE        ?= 2.0
TURN          ?= 5.0
SETTLE_TIME   ?= 0.3
PREVIEW_HEIGHT ?= 720
# Port every ScanStudio web app serves on (http://localhost:$(PORT)) —
# capture and both reviews share it, since the phases run serially. One
# ChromeOS port-forwarding rule covers the whole pipeline.
PORT          ?= 8412
# VERBOSE=1 adds live-web's pipeline diagnostics (per-message camera/socket
# stats) to the log. Default keeps it to captures and warnings.
VERBOSE       ?=
VERBOSE_FLAG  := $(if $(strip $(VERBOSE)),--verbose,)

.PHONY: all bw live live-web finish finish-web motion peaks keyframes review review-web crop split page-review page-review-web binarize pdf pdf-bw clean install install-legacy tkinter ffmpeg probe-camera test help

help:
	@echo "ScanStudio Pipeline"
	@echo ""
	@echo "Usage: make <target> VIDEO=recordings/mybook.mp4"
	@echo ""
	@echo "  all           Full pipeline (pauses at review)"
	@echo "  bw            Binarize + B&W PDF"
	@echo "  live          P0: Live webcam capture (make live NAME=mybook)"
	@echo "  live-web      P0: Live capture via Chrome's camera (ChromeOS-friendly)"
	@echo "  probe-camera  List camera indices and which one delivers 4K"
	@echo "  finish        P4-P9 back half (run after 'live')"
	@echo "  finish-web    P4-P9 with the reviews in Chrome (ChromeOS-friendly)"
	@echo ""
	@echo "  motion        P1: Motion signal"
	@echo "  peaks         P2: Detect peaks"
	@echo "  keyframes     P3: Select keyframes"
	@echo "  review        P4: Review keyframes (GUI, reentrant)"
	@echo "  review-web    P4 in Chrome — ChromeOS-friendly, <video> insert scrubber"
	@echo "  crop          P5: Crop keyframes"
	@echo "  split         P6: Split into pages"
	@echo "  page-review   P7: Page review — drop/adjust/mark documents (GUI)"
	@echo "  page-review-web  P7 in Chrome (ChromeOS-friendly)"
	@echo "  binarize      P8: Binarize to B&W"
	@echo "  pdf           P9: Build PDF"
	@echo "  pdf-bw        P9: Build B&W PDF"
	@echo ""
	@echo "  clean         Delete output/<NAME>/ (VIDEO= or NAME=; keeps recording)"
	@echo "  test          Headless tests (no camera or display needed)"
	@echo ""
	@echo "  install         Pipeline dependencies (requirements.txt)"
	@echo "  install-legacy  Torch etc. for the root legacy scripts (several GB)"
	@echo ""
	@echo "  SAFETY_MARGIN=$(SAFETY_MARGIN)  BLOCK_SIZE=$(BLOCK_SIZE)  BW_OFFSET=$(BW_OFFSET)"
	@echo "  BW_METHOD=$(BW_METHOD) (sauvola|adaptive)  BW_UPSCALE=$(BW_UPSCALE)  BW_K=$(BW_K) (higher=thinner)"
	@echo "  MODE=$(MODE)  (double=book spreads, single=loose docs)"
	@echo "  SPLIT_DOCS=$(SPLIT_DOCS)  (auto=one PDF per document too, never=combined only)"
	@echo "  live: CAMERA=$(CAMERA)  SETTLE=$(SETTLE)  TURN=$(TURN)  SETTLE_TIME=$(SETTLE_TIME)  PREVIEW_HEIGHT=$(PREVIEW_HEIGHT)"
	@echo "  web apps (live-web, review-web, page-review-web): PORT=$(PORT), shared"

all: motion peaks keyframes finish
	@echo "Pipeline complete: $(PDF)"

# Back half (P4-P9): review, crop, split, page-review, build PDF.
# Use after 'live' (or run individually). Pauses at P4 and P7.
finish: review crop split page-review pdf
	@echo "Pipeline complete: $(PDF)"

# The same back half with the browser reviews: pauses at P4 and P7 until
# Finish (Q) is pressed in the tab, then the chain proceeds — the web
# analogue of closing the Tk window. Ctrl+C in the terminal aborts the chain.
finish-web: review-web crop split page-review-web pdf
	@echo "Pipeline complete: $(PDF)"

# Live capture (P0): record the webcam and auto-select keyframes in real time.
# Replaces P1-P3; produces the recording + the same artifacts they would.
#   make live NAME=mybook [CAMERA=1]
#   make finish VIDEO=recordings/mybook.mp4
live:
ifeq ($(strip $(NAME)),)
	$(error NAME is required. Usage: make live NAME=mybook)
endif
	@mkdir -p recordings
	$(PYTHON) $(SCRIPTS)/p0_live_capture.py output/$(NAME) recordings/$(NAME).mp4 \
		--camera $(CAMERA) --settle-threshold $(SETTLE) --turn-threshold $(TURN) \
		--settle-time $(SETTLE_TIME) --preview-height $(PREVIEW_HEIGHT)
	@echo "Live capture done. Continue with: make finish VIDEO=recordings/$(NAME).mp4"

# Live capture (P0) with the browser as the camera. Same artifacts as 'live';
# the only road to the camera on ChromeOS, where the container has no
# /dev/video*. Open http://localhost:$(PORT) in Chrome once it starts
# (ChromeOS: forward the port first — Settings > Linux > Port forwarding).
live-web:
ifeq ($(strip $(NAME)),)
	$(error NAME is required. Usage: make live-web NAME=mybook)
endif
	@mkdir -p recordings
	$(PYTHON) $(SCRIPTS)/p0_web_capture.py output/$(NAME) recordings/$(NAME).mp4 \
		--port $(PORT) --settle-threshold $(SETTLE) --turn-threshold $(TURN) \
		--settle-time $(SETTLE_TIME) $(VERBOSE_FLAG)
	@echo "Live capture done. Continue with: make finish-web VIDEO=recordings/$(NAME).mp4"

bw: binarize pdf-bw
	@echo "B&W pipeline complete: $(PDF_BW)"

motion: $(MOTION)
$(MOTION):
	$(PYTHON) $(SCRIPTS)/p1_motion_signal.py $(VIDEO)

peaks: $(PEAKS)
$(PEAKS): $(MOTION)
	$(PYTHON) $(SCRIPTS)/p2_detect_peaks.py $(OUTDIR)

keyframes: $(KEYFRAMES)
$(KEYFRAMES): $(PEAKS)
	$(PYTHON) $(SCRIPTS)/p3_select_keyframes.py $(OUTDIR) $(VIDEO)

review: $(KEYFRAMES)
	$(PYTHON) $(SCRIPTS)/p4_review_keyframes.py $(OUTDIR) $(VIDEO) --mode $(MODE)

# The same review served to Chrome (scripts/p4_web_review.py): identical
# state and save format, browser rendering. The insert scrubber is a native
# <video> over the recording — the ChromeOS path, where Tk clips off small
# screens and software-decodes 4K per scrub step.
review-web: $(KEYFRAMES)
	$(PYTHON) $(SCRIPTS)/p4_web_review.py $(OUTDIR) $(VIDEO) --mode $(MODE) --port $(PORT)

crop: $(KEYFRAMES)
	$(PYTHON) $(SCRIPTS)/p5_crop.py $(OUTDIR) --mode $(MODE) --safety-margin $(SAFETY_MARGIN)

split: $(PAGES)
$(PAGES): $(KEYFRAMES)
	$(PYTHON) $(SCRIPTS)/p6_split_pages.py $(OUTDIR) --mode $(MODE)

page-review: $(PAGES)
	$(PYTHON) $(SCRIPTS)/p7_review_pages.py $(OUTDIR)

page-review-web: $(PAGES)
	$(PYTHON) $(SCRIPTS)/p7_web_review.py $(OUTDIR) --port $(PORT)

binarize: $(BW_META)
$(BW_META): $(PAGES)
	$(PYTHON) $(SCRIPTS)/p8_binarize.py $(OUTDIR) --method $(BW_METHOD) --block-size $(BLOCK_SIZE) --offset $(BW_OFFSET) --upscale $(BW_UPSCALE) --sauvola-k $(BW_K)

pdf: $(PDF)
$(PDF): $(PAGES)
	$(PYTHON) $(SCRIPTS)/p9_build_pdf.py $(OUTDIR) --split-docs $(SPLIT_DOCS)

pdf-bw: $(PDF_BW)
$(PDF_BW): $(BW_META)
	$(PYTHON) $(SCRIPTS)/p9_build_pdf.py $(OUTDIR) --source bw --pdf-name $(NAME)_bw.pdf --split-docs $(SPLIT_DOCS)

probe-camera:
	$(PYTHON) $(SCRIPTS)/probe_camera.py

# Headless tests — no camera, no display, no recording needed. Each file runs
# standalone under plain python, so this needs nothing beyond requirements.txt
# (pytest works too, if you have it: `pytest tests/`).
test:
	@for t in tests/test_*.py; do echo "$$t"; $(PYTHON) $$t || exit 1; done

install: $(VENV) tkinter ffmpeg
	$(PYTHON) -m pip install -r requirements.txt
	@$(PYTHON) -c "import cv2" 2>/dev/null && echo "opencv OK" || { \
		echo "opencv installed but won't import."; \
		command -v apt-get >/dev/null 2>&1 && \
			echo "Minimal Linux images lack the libraries its wheel links against:"; \
		command -v apt-get >/dev/null 2>&1 && \
			echo "  sudo apt-get install -y libgl1 libglib2.0-0"; \
		exit 1; \
	}

# ffmpeg does the fast CFR normalize at the end of `make live-web`; without it
# OpenCV re-encodes 4K in software, which works but is painfully slow. Unlike
# tkinter it IS installed with sudo on apt systems (the main audience is
# ChromeOS Crostini, where sudo is passwordless), but since live-web has the
# OpenCV fallback, nothing here ever fails `make install`.
ffmpeg:
	@command -v ffmpeg >/dev/null 2>&1 && echo "ffmpeg OK" || { \
		if [ "$$(uname)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then \
			echo "Installing ffmpeg via Homebrew..."; \
			brew install ffmpeg || echo "WARNING: ffmpeg install failed; live-web falls back to OpenCV"; \
		elif command -v apt-get >/dev/null 2>&1; then \
			echo "Installing ffmpeg via apt..."; \
			sudo apt-get install -y ffmpeg || echo "WARNING: ffmpeg install failed; live-web falls back to OpenCV"; \
		else \
			echo "NOTE: no ffmpeg — install it with your package manager for fast"; \
			echo "live-web finishing (the OpenCV fallback works but is slow at 4K)"; \
		fi; \
	}

# Torch and friends for the legacy scripts at the repo root — several GB, and
# nothing in P0-P9 imports them, so this is deliberately separate from
# `install` rather than part of it.
install-legacy: $(VENV)
	$(PYTHON) -m pip install -r requirements-legacy.txt

# Create the project venv on demand. macOS's bundled Python is often old and
# linked against LibreSSL, which urllib3 2.x does not support; Homebrew Python
# provides a current interpreter linked against OpenSSL. On Debian/Ubuntu the
# venv module is a separate package, so a bare failure there is usually a
# missing python3-venv rather than a broken Python.
.venv/bin/python:
	@set -e; \
	bootstrap_python="$$(command -v python3)"; \
	if [ "$$(uname)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then \
		echo "Installing or updating Homebrew Python..."; \
		brew install python; \
		bootstrap_python="$$(brew --prefix python)/bin/python3"; \
	fi; \
	"$$bootstrap_python" -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ is required'"; \
	echo "Creating .venv with $$bootstrap_python..."; \
	"$$bootstrap_python" -m venv .venv || { \
		echo "Could not create the venv."; \
		command -v apt-get >/dev/null 2>&1 && \
			echo "On Debian/Ubuntu: sudo apt-get install -y python3-venv"; \
		exit 1; \
	}

# tkinter is a system package (not pip-installable). The review GUIs (P4/P7)
# need it. macOS: install the matching Homebrew package for the active Python.
# Linux: print the right package for the distro's package manager — installing
# it needs root, which is the operator's call, not the Makefile's. A venv reads
# tkinter from the base Python's stdlib, so it works right after without
# recreating .venv.
tkinter:
	@$(PYTHON) -c "import tkinter" 2>/dev/null && echo "tkinter OK" || { \
		echo "tkinter missing."; \
		ver=$$($(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"); \
		if [ "$$(uname)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then \
			echo "Installing python-tk@$$ver via Homebrew..."; \
			brew install python-tk@$$ver; \
		elif command -v apt-get >/dev/null 2>&1; then \
			echo "Install it with:  sudo apt-get install -y python3-tk"; exit 1; \
		elif command -v dnf >/dev/null 2>&1; then \
			echo "Install it with:  sudo dnf install -y python3-tkinter"; exit 1; \
		elif command -v pacman >/dev/null 2>&1; then \
			echo "Install it with:  sudo pacman -S --needed tk"; exit 1; \
		elif command -v zypper >/dev/null 2>&1; then \
			echo "Install it with:  sudo zypper install -y python3-tk"; exit 1; \
		else \
			echo "Install the Tk bindings for Python $$ver with your package manager."; \
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