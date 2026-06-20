# Scan Studio

A data preparation pipeline for converting book scanning videos into high-quality PDFs. Scan a book one of two ways:

- **Live capture** — point your webcam at the book like a robot head, run `make live`, and it records and selects a keyframe for each spread in real time as you turn pages.
- **From a recording** — already have a video of you scanning? Run `make all` and get a PDF of every page.

Both paths converge on the same review → crop → split → PDF back half.

For downstream text extraction and dataframe queries from generated PDFs, use the supporting [Document Parser](https://github.com/rlnsanz/document_parser).

## Quick Start

Live capture (webcam):

```bash
make install
make live NAME=mybook
make finish VIDEO=recordings/mybook.mp4
```

From an existing recording:

```bash
make install
make all VIDEO=recordings/mybook.mp4
```

Either way, the pipeline pauses twice for interactive review (P4 and P7), and the finished PDF lands at `output/mybook/pdf/mybook.pdf`.

## Overview

A scan flows through numbered phases. The front end produces a recording plus keyframe images and metadata; the back half (P4–P9) reviews, crops, splits, and assembles the PDF. **P0 (live capture) is an alternative front end to P1–P3** — it produces the exact same artifacts in real time, so everything downstream is identical regardless of which path you use.

| Phase | Name | Type |
|-------|------|------|
| P0 | Live Capture | **Interactive** (alternative to P1–P3) |
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
- **Storage** — plan for 10+ GB per book at 4K (the default capture resolution; the raw recording dominates), less at 1080p, plus one image per page in `images/` and `pages/`
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
└── pdf/        # <name>.pdf and <name>_bw.pdf
```

## Commands

Most targets require `VIDEO=path/to/file.mp4`. The exception is `make live`, which takes `NAME=` instead (the recording doesn't exist yet) and creates `recordings/<NAME>.mp4`.

| Command | Description |
|---------|-------------|
| `make live NAME=...` | P0: Live webcam capture — records + selects keyframes, then run `make finish VIDEO=recordings/<NAME>.mp4` |
| `make finish VIDEO=...` | Back half (P4–P9): review, crop, split, page-review, PDF — run after `make live` |
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
| `make probe-camera` | List camera indices and which one delivers 4K |
| `make install` | Install Python dependencies |
| `make help` | Show all targets and parameters |

## Pipeline Details

### P0 — Live Capture (interactive)

```bash
make live NAME=mybook
make live NAME=mybook CAMERA=1 SETTLE=1.5 TURN=4.0
```

Opens a live webcam window and uses it like a robot head: it records the feed to `recordings/<NAME>.mp4` while an online state machine watches motion and **auto-captures the sharpest frame each time the book settles after a page turn**. The on-screen overlay shows a live motion bar with the settle/turn thresholds marked, the current state (`WAITING` / `SETTLED` / `TURNING`), and a running capture count.

This replaces P1–P3: on quit it writes the recording plus `images/`, `json/keyframes.json`, `json/metadata.json`, and the signal arrays in `data/` — the same artifacts P1–P3 produce. Continue straight into review:

```bash
make finish VIDEO=recordings/mybook.mp4
```

**Keys:**

| Key | Action |
|-----|--------|
| `Q` / `Esc` | Quit and save |
| `U` | Undo last capture |
| `C` | Force-capture the current frame now |
| `Space` | Pause / resume auto-capture |
| `M` | Toggle capture sound mute |

**Tuning:** webcam motion magnitudes differ from pre-recorded clips, so you may need to adjust the thresholds (see Configuration). Watch the motion bar relative to the threshold ticks: if turns aren't detected, lower `TURN`; if it captures while you're still moving, raise `SETTLE` or `SETTLE_TIME`.

> **Camera selection:** `make live` requests 4K and `CAMERA=auto` (the default) picks whichever connected camera actually delivers it — USB indices shuffle on reconnect, so run `make probe-camera` to see what each reports, or set `CAMERA=<index>` to force one. Mount the camera on a fixed stand so framing stays stable across the session.

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

Tkinter GUI for reviewing and correcting the keyframe selection. This phase is reentrant — run it as many times as needed before proceeding.

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
| `C` | Toggle center line |
| `G` | Adjust geometry — **double:** split/gutter line (`←`/`→` gutter, `[` `]` rotate); **single:** crop box (arrows move, `⇧`+arrows resize, `[` `]` tilt). `Enter` save, `Esc` cancel, `Backspace` reset |
| `⌘S` | Save |

`G` adapts to `MODE=` (the `review` target passes it through automatically):

- **double** — previews the spread split + deskew and lets you correct the gutter and rotation per spread; corrections propagate forward to later spreads.
- **single** — the gutter overlay is hidden (each frame is already one page). `G` opens a **crop editor**: an adjustable rotated rectangle over the raw frame. Use it when the GrabCut auto-crop (P5) clips real text or wanders as page sizes vary (e.g. receipts). Confirming stores the box as 4 corners on the keyframe, and P5 warps exactly that box instead of auto-detecting. A confirmed crop is drawn as a green box during review.

### P5 — Crop

```bash
make crop VIDEO=recordings/mybook.mp4
```

Crops the book/page out of the surrounding frame. Modifies `images/` in-place. Re-run P3 to restore originals.

- `double` mode: applies crop bounds from P4 + Otsu background detection
- `single` mode: warps a P4 manual crop box (`crop_quad`) if present; otherwise uses GrabCut to segment the page from the table surface (handles rotation, works with any page color)

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

| Key | Action |
|-----|------|
| `1` | Flag: Great |
| `2` | Flag: Acceptable |
| `3` | Flag: Poor |
| `4` | Flag: Crop issue |
| `0` | Clear flag |
| `X` | Toggle drop page |
| `⌘S` | Save |

### P8 — Binarize (optional)

```bash
make binarize VIDEO=recordings/mybook.mp4
```

Produces clean black-and-white images → `bw/` (written as lossless PNG). Defaults to Sauvola local thresholding; set `BW_METHOD=adaptive` for the older Gaussian adaptive threshold. The grayscale is upscaled (`BW_UPSCALE`) first to anti-alias letter edges. Tune with `BW_METHOD`, `BW_UPSCALE`, `BW_K` (Sauvola; higher = thinner strokes), `BLOCK_SIZE`, and `BW_OFFSET` (adaptive only) — see Configuration.

### P9 — Build PDF

```bash
make pdf VIDEO=recordings/mybook.mp4       # color PDF from pages/
make pdf-bw VIDEO=recordings/mybook.mp4   # B&W PDF from bw/
```

Assembles pages in order into a PDF using reportlab. Output: `pdf/<name>.pdf` or `pdf/<name>_bw.pdf`.

## Configuration

Override parameters on the command line:

```bash
make all VIDEO=recordings/mybook.mp4 MODE=single SAFETY_MARGIN=0.01
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VIDEO` | *(required)* | Path to input video file (all targets except `live`) |
| `NAME` | *(required for `live`)* | Project name; `make live` records to `recordings/<NAME>.mp4` |
| `MODE` | `double` | `double` for book spreads, `single` for loose documents |
| `SAFETY_MARGIN` | `0.005` | Crop safety margin as a fraction of image dimension |
| `BW_METHOD` | `sauvola` | Binarization method: `sauvola` or `adaptive` |
| `BW_UPSCALE` | `2` | Grayscale upscale factor before thresholding (anti-aliases edges) |
| `BW_K` | `0.2` | Sauvola threshold factor (higher = thinner strokes) |
| `BLOCK_SIZE` | `51` | Threshold window size for binarization (must be odd) |
| `BW_OFFSET` | `10` | Threshold offset (`adaptive` method only) |
| `CAMERA` | `auto` | Webcam for `make live` — `auto` picks the camera that delivers 4K, or set an index (`make probe-camera` lists them) |
| `SETTLE` | `2.0` | Live: motion below this counts as "still" (book settled) |
| `TURN` | `5.0` | Live: motion above this counts as a page turn in progress |
| `SETTLE_TIME` | `0.4` | Live: seconds of stillness required before a capture fires |

## Example Walkthrough

```bash
# 1. Run the full automated + interactive pipeline
make all VIDEO=recordings/african_founders.mp4

# At P4: review keyframes in the GUI, label bad frames, insert missing ones, save with ⌘S
# At P7: flag page quality, close the window when done

# 2. Optionally produce a B&W version
make bw VIDEO=recordings/african_founders.mp4

# Output:
#   output/african_founders/pdf/african_founders.pdf
#   output/african_founders/pdf/african_founders_bw.pdf
```

## Troubleshooting

**No peaks detected** — The video may have low-contrast page turns. Check `plots/motion_plot.png` to inspect the signal. Adjust peak detection parameters in `scripts/p2_detect_peaks.py`.

**Crop removes too much / too little** — Adjust `SAFETY_MARGIN`. For an off-center spine, fix the split with the `G` gutter line in P4 review.

**Binarization looks wrong** — Try `BW_METHOD=adaptive`, or tune `BW_K` (Sauvola stroke weight; higher = thinner), `BLOCK_SIZE` (larger = coarser regions), and `BW_OFFSET` (adaptive only).

**PDF page order is wrong** — Page ordering follows the `pages.json` metadata. Check `json/keyframes.json` for frame numbering issues.

## License

Apache 2.0
