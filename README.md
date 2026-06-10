# Scan Studio

A data preparation pipeline for converting book scanning videos into high-quality PDFs. Point it at a video of you scanning a book, run `make all`, and get a PDF of every page.

For downstream text extraction and dataframe queries from generated PDFs, use the supporting [Document Parser](https://github.com/rlnsanz/document_parser).

## Quick Start

```bash
make install
make all VIDEO=recordings/mybook.mp4
```

The pipeline pauses twice for interactive review (P4 and P7). After `make all` completes, the PDF is at `output/mybook/pdf/book.pdf`.

## Overview

Scan Studio processes a single video file through nine numbered phases:

| Phase | Name | Type |
|-------|------|------|
| P1 | Motion Signal | Automated |
| P2 | Detect Peaks | Automated |
| P3 | Select Keyframes | Automated |
| P4 | Review Keyframes | **Interactive** |
| P5 | Crop | Automated |
| P6 | Split Pages | Automated |
| P7 | Page Quality Review | **Interactive** |
| P8 | Binarize | Automated (optional) |
| P9 | Build PDF | Automated |

Two modes are supported via `MODE=`:
- `double` (default) — book spreads filmed two pages at a time
- `single` — loose documents filmed one page at a time

## System Requirements

**macOS only.** The interactive review GUIs (P4, P7) use the macOS Command key (`⌘S` to save) and are not tested on other platforms.

- **macOS 13 Ventura or later** recommended
- **Python 3.10+** — the pipeline uses union type syntax (`str | None`) introduced in 3.10
- **RAM** — 16 GB recommended; 8 GB workable for shorter or lower-resolution videos. P3 holds full-resolution keyframes in memory during extraction.
- **Storage** — plan for 3–10 GB per book depending on video resolution and length (raw video + one full-res image per page in `images/` and `pages/`)
- **Apple Silicon (M1 or later)** — not required for the core pipeline, but significantly faster for the torch-based legacy scripts (`featurize.py`, `ocr.py`, `yolo.py`)

**tkinter note:** P4 and P7 use tkinter for their GUIs. If you installed Python via Homebrew and tkinter is missing, run:
```bash
brew install python-tk
```

## Prerequisites

- Python 3.10+
- No external command-line tools required (pure Python pipeline)

## Installation

```bash
make install
```

This installs all packages from `requirements.txt`.

## Directory Structure

Input video can be anywhere; the conventional location is `recordings/`.

All output is written to `output/<video_name>/`:

```
output/<name>/
├── images/     # full-resolution keyframe images (modified in-place by crop)
├── pages/      # individual split pages ready for PDF
├── bw/         # binarized B&W pages (created by make bw)
├── plots/      # diagnostic plots (motion signal, peak detection)
├── data/       # raw signal arrays (.npy)
├── json/       # metadata, keyframe list, review logs
├── reports/    # markdown and text reports
└── pdf/        # book.pdf and book_bw.pdf
```

## Commands

Every target requires `VIDEO=path/to/file.mp4`.

| Command | Description |
|---------|-------------|
| `make all VIDEO=...` | Full pipeline — runs P1–P7 and P9, pauses at P4 and P7 |
| `make bw VIDEO=...` | Binarize + B&W PDF (run after `make all`) |
| `make motion VIDEO=...` | P1: Compute motion signal |
| `make peaks VIDEO=...` | P2: Detect page-turn peaks |
| `make keyframes VIDEO=...` | P3: Extract keyframe images |
| `make review VIDEO=...` | P4: Review keyframes (GUI, reentrant) |
| `make crop VIDEO=...` | P5: Crop keyframes |
| `make split VIDEO=...` | P6: Split into individual pages |
| `make page-review VIDEO=...` | P7: Page quality review (GUI) |
| `make binarize VIDEO=...` | P8: Binarize to B&W |
| `make pdf VIDEO=...` | P9: Build color PDF |
| `make pdf-bw VIDEO=...` | P9: Build B&W PDF |
| `make clean VIDEO=...` | Delete all outputs for this video |
| `make install` | Install Python dependencies |
| `make help VIDEO=...` | Show all targets and parameters |

## Pipeline Details

### P1 — Motion Signal

```bash
make motion VIDEO=recordings/mybook.mp4
```

Reads every frame of the video at reduced resolution and computes per-frame pixel differences to build a motion signal. Saves the raw and smoothed signal to `data/` and a diagnostic plot to `plots/motion_plot.png`.

### P2 — Detect Peaks

```bash
make peaks VIDEO=recordings/mybook.mp4
```

Finds peaks in the motion signal that correspond to page-turn events. Saves detected peak indices to `data/peaks.npy` and a labeled plot to `plots/`.

### P3 — Select Keyframes

```bash
make keyframes VIDEO=recordings/mybook.mp4
```

For each detected spread, picks the lowest-motion frame (sharpness as tiebreaker). Extracts full-resolution images from the video into `images/` and writes `json/keyframes.json`.

### P4 — Review Keyframes (interactive)

```bash
make review VIDEO=recordings/mybook.mp4
```

OpenCV GUI for reviewing and correcting the keyframe selection. This phase is reentrant — run it as many times as needed before proceeding.

**Keys:**

| Key | Action |
|-----|--------|
| `→` / `D` | Next frame |
| `←` / `A` | Previous frame |
| `1` | Keep |
| `2` | Delete — Duplicate |
| `3` | Delete — Occlusion |
| `4` | Delete — Other |
| `5` | Mark as Cover |
| `6` | Mark as Doc Start |
| `I` | Insert frame (opens video scrubber) |
| `L` | Set crop guides (`←/→` move, `Tab` switch, `Enter` confirm, `Esc` cancel) |
| `Shift+L` | Per-frame crop override |
| `N` | Open note field |
| `⌘S` | Save |

### P5 — Crop

```bash
make crop VIDEO=recordings/mybook.mp4
```

Crops the book/page out of the surrounding frame. Modifies `images/` in-place. Re-run P3 to restore originals.

- `double` mode: applies crop bounds from P4 + Otsu background detection
- `single` mode: uses GrabCut to segment the page from the table surface; handles rotation and works with any page color

### P6 — Split Pages

```bash
make split VIDEO=recordings/mybook.mp4
```

- `double` mode: splits each keyframe at the center spine into left and right pages → `pages/`
- `single` mode: copies cropped images directly → `pages/`

### P7 — Page Quality Review (interactive)

```bash
make page-review VIDEO=recordings/mybook.mp4
```

Tkinter GUI for flagging page quality. Results saved to `json/page_review.json`.

| Key | Flag |
|-----|------|
| `1` | Great |
| `2` | Acceptable |
| `3` | Poor |
| `4` | Crop issue |

### P8 — Binarize (optional)

```bash
make binarize VIDEO=recordings/mybook.mp4
```

Applies adaptive thresholding to produce clean black-and-white images → `bw/`. Controlled by `BLOCK_SIZE` and `BW_OFFSET` (see Configuration).

### P9 — Build PDF

```bash
make pdf VIDEO=recordings/mybook.mp4       # color PDF from pages/
make pdf-bw VIDEO=recordings/mybook.mp4   # B&W PDF from bw/
```

Assembles pages in order into a PDF using reportlab. Output: `pdf/book.pdf` or `pdf/book_bw.pdf`.

## Configuration

Override parameters on the command line:

```bash
make all VIDEO=recordings/mybook.mp4 MODE=single SAFETY_MARGIN=0.01
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VIDEO` | *(required)* | Path to input video file |
| `MODE` | `double` | `double` for book spreads, `single` for loose documents |
| `SAFETY_MARGIN` | `0.005` | Crop safety margin as a fraction of image dimension |
| `BLOCK_SIZE` | `51` | Adaptive threshold block size for binarization (must be odd) |
| `BW_OFFSET` | `10` | Threshold offset for binarization |

## Example Walkthrough

```bash
# 1. Run the full automated + interactive pipeline
make all VIDEO=recordings/african_founders.mp4

# At P4: review keyframes in the GUI, label bad frames, insert missing ones, save with ⌘S
# At P7: flag page quality, close the window when done

# 2. Optionally produce a B&W version
make bw VIDEO=recordings/african_founders.mp4

# Output:
#   output/african_founders/pdf/book.pdf
#   output/african_founders/pdf/book_bw.pdf
```

## Troubleshooting

**No peaks detected** — The video may have low-contrast page turns. Check `plots/motion_plot.png` to inspect the signal. Adjust peak detection parameters in `scripts/p2_detect_peaks.py`.

**Crop removes too much / too little** — Adjust `SAFETY_MARGIN`, or use the `L` key in P4 review to set crop guides manually.

**Binarization looks wrong** — Try different `BLOCK_SIZE` (larger = coarser regions) and `BW_OFFSET` (higher = more aggressive thresholding).

**PDF page order is wrong** — Page ordering follows the `pages.json` metadata. Check `json/keyframes.json` for frame numbering issues.

## License

Apache 2.0
