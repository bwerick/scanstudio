# Scan Studio

A data preparation pipeline for converting book scanning videos into high-quality PDFs. Scan a book one of two ways:

- **Live capture** — point your webcam at the book like a robot head, run `make live`, and it records and selects a keyframe for each spread in real time as you turn pages.
- **From a recording** — already have a video of you scanning? Run `make all` and get a PDF of every page.

Both paths converge on the same review → crop → split → PDF back half.


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
| P7 | Page Review | **Interactive** |
| P8 | Binarize | Automated (optional) |
| P9 | Build PDF | Automated |

Two modes are supported via `MODE=`:
- `double` (default) — book spreads filmed two pages at a time
- `single` — loose documents filmed one page at a time

## System Requirements

**macOS and Linux.** The pipeline is pure Python/OpenCV; the three things that
genuinely differ between the two — the camera backend, the save key, and the
capture chime — are resolved at runtime in `scripts/utils.py`, so the same
commands work on both. The save accelerator follows the platform: **`⌘S` on
macOS, `Ctrl+S` on Linux** (every other key is the same).

- **Python 3.10+** — the pipeline uses union type syntax (`str | None`) introduced in 3.10
- **macOS 13 Ventura or later**, or a **Linux desktop with X11/Wayland** — P4 and P7 open GUI windows, and P0 opens a live preview, so a headless box can only run the automated phases (P1–P3, P5, P6, P8, P9)
- **RAM** — 16 GB recommended; 8 GB workable for shorter or lower-resolution videos. P3 holds full-resolution keyframes in memory during extraction.
- **Storage** — plan for 10+ GB per book at 4K (the default capture resolution; the raw recording dominates), less at 1080p, plus one image per page in `images/` and `pages/`
- **GPU** — not required for the core pipeline, but the torch-based legacy scripts (`featurize.py`, `ocr.py`, `yolo.py`) are significantly faster on Apple Silicon (MPS) or a CUDA card on Linux; they fall back to CPU otherwise

**tkinter note:** P4 and P7 use tkinter for their GUIs. It is a system package, not
pip-installable, so `make install` checks for it and prints the right command for
your platform if it's missing:

```bash
brew install python-tk            # macOS (Homebrew Python)
sudo apt-get install python3-tk   # Debian / Ubuntu
sudo dnf install python3-tkinter  # Fedora / RHEL
sudo pacman -S tk                 # Arch
```

A venv reads tkinter from the base Python's stdlib, so installing it afterwards
works without recreating `.venv`.

## Prerequisites

- Python 3.10+
- No external command-line tools required (pure Python pipeline)
- **Linux only:** on a minimal image (server, container, WSL) the OpenCV wheel
  also needs `libgl1` and `libglib2.0-0`, and `python3-venv` must be present for
  `make install` to create `.venv`:
  ```bash
  sudo apt-get install -y python3-venv python3-tk libgl1 libglib2.0-0
  ```
  Your user must be in the `video` group to open a webcam for `make live`
  (`sudo usermod -aG video $USER`, then log out and back in).

## Installation

```bash
make install
```

This installs the pipeline's dependencies from `requirements.txt`, then verifies
tkinter and OpenCV actually import — the two things that fail for platform
reasons rather than pip reasons.

The torch-based legacy scripts at the repo root (`ocr.py`, `yolo.py`,
`pageselection.py`, the streamlit apps) are **not** part of that: their
dependencies are several GB, and no pipeline phase imports them. Install them
only if you're running those scripts:

```bash
make install-legacy
```

On Linux, pip resolves torch to the CUDA build by default; for the much smaller
CPU-only wheel, install it first with
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Directory Structure

Input video can be anywhere; the conventional location is `recordings/`.

All output is written to `output/<video_name>/`:

```
output/<name>/
├── images/     # full-resolution keyframe images (modified in-place by crop)
├── pages/      # individual split pages ready for PDF
├── pages_orig/ # pristine copies of pages P7 re-rendered (rotate/translate)
├── bw/         # binarized B&W pages (created by make bw)
├── plots/      # diagnostic plots (motion signal, peak detection)
├── data/       # raw signal arrays (.npy)
├── json/       # metadata, keyframe list, review logs
├── reports/    # markdown and text reports
└── pdf/        # <name>.pdf, <name>_bw.pdf, and one PDF per document
```

## Commands

Most targets require `VIDEO=path/to/file.mp4`. The exception is `make live`, which takes `NAME=` instead (the recording doesn't exist yet) and creates `recordings/<NAME>.mp4`.

| Command | Description |
|---------|-------------|
| `make live NAME=...` | P0: Live webcam capture — records + selects keyframes, then run `make finish VIDEO=recordings/<NAME>.mp4` |
| `make live-web NAME=...` | P0 with Chrome as the camera — same artifacts; the only camera path on ChromeOS |
| `make finish VIDEO=...` | Back half (P4–P9): review, crop, split, page-review, PDF — run after `make live` |
| `make finish-web VIDEO=...` | Same back half with the reviews in Chrome — press **Finish (Q)** in each tab to let the chain advance |
| `make all VIDEO=...` | Full pipeline — runs P1–P7 and P9, pauses at P4 and P7 |
| `make bw VIDEO=...` | Binarize + B&W PDF (run after `make all`) |
| `make motion VIDEO=...` | P1: Compute motion signal |
| `make peaks VIDEO=...` | P2: Detect page-turn peaks |
| `make keyframes VIDEO=...` | P3: Extract keyframe images |
| `make review VIDEO=...` | P4: Review keyframes (GUI, reentrant) |
| `make review-web VIDEO=...` | P4 in Chrome — same review, ChromeOS-friendly, `<video>` insert scrubber |
| `make crop VIDEO=...` | P5: Crop keyframes |
| `make split VIDEO=...` | P6: Split into individual pages |
| `make page-review VIDEO=...` | P7: Page review — drop pages, adjust geometry, mark documents (GUI) |
| `make page-review-web VIDEO=...` | P7 in Chrome (ChromeOS-friendly) |
| `make binarize VIDEO=...` | P8: Binarize to B&W |
| `make pdf VIDEO=...` | P9: Build color PDF |
| `make pdf-bw VIDEO=...` | P9: Build B&W PDF |
| `make clean VIDEO=...` | Delete all outputs for this video |
| `make probe-camera` | List camera indices and which one delivers 4K |
| `make install` | Install the pipeline's Python dependencies |
| `make install-legacy` | Install torch etc. for the root legacy scripts (several GB) |
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

> **Camera selection:** `make live` requests 4K and `CAMERA=auto` (the default) picks whichever connected camera actually delivers it — indices shuffle on reconnect (and on Linux one physical camera usually claims several `/dev/video*` nodes, only one of which captures), so run `make probe-camera` to see what each reports, or set `CAMERA=<index>` to force one. `CAMERA=<n>` is `/dev/video<n>` on Linux. Mount the camera on a fixed stand so framing stays stable across the session.
>
> The capture backend is chosen per platform: AVFoundation on macOS, V4L2 on Linux — the only ones that expose a webcam's high-res modes. On Linux the format is set to MJPG before the resolution, since most UVC cameras offer 4K only as MJPG and default to raw YUYV.

### P0 — Live Capture in the browser (`make live-web`)

Same phase, different front end: **Chrome owns the camera** via `getUserMedia`/`MediaRecorder`, and streams small analysis frames to a local server (`scripts/p0_web_capture.py`) that runs the *same* detector (`scripts/live_state.py`) as `make live` — two front ends, one brain, identical artifacts, so `make finish` works unchanged.

```bash
make live-web NAME=mybook       # then open http://localhost:8412 in Chrome
```

Built for **ChromeOS**: the Crostini container has no `uvcvideo` kernel module and therefore no `/dev/video*` regardless of the USB-sharing toggles, so the browser is the only road to the camera there. Before opening the page, forward the port once: *Settings → Linux → Port forwarding → Add 8412* (`localhost` is a secure context, `penguin.linux.test` is not). The page uses `MediaStreamTrackProcessor`, so it needs Chrome/Edge 94+, and the same keys as `make live` work in the tab (`Q` finishes).

Two differences from the native path: the recording is encoded by the browser (hardware H.264/VP9) and normalized to constant frame rate at the end — with `ffmpeg` if installed (`make install` sets it up; without it the OpenCV fallback works but is slow at 4K) — because P4 scrubs the recording by frame index; and keep the tab visible while scanning, since a hidden tab stops frame analysis (the page warns if this happens).

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

Prefer the browser on ChromeOS (or any small screen): `make review-web VIDEO=...` serves the same review to Chrome at `http://localhost:8412` — identical state, keys, and save format, with the insert scrubber as a native `<video>` (hardware decode, instant seeking, vs. a software 4K decode per keypress in Tk). All ScanStudio web apps share port 8412 (they run serially), so the one ChromeOS forwarding rule from `live-web` already covers this.

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
| `6` | Mark as Doc Start — this spread begins a new document (P6 passes it to its first page; P7's `F` refines it; P9 gives it its own PDF) |
| `I` | Insert frame (opens video scrubber) |
| `C` | Toggle center line |
| `E` | Jump to the next watchdog-flagged frame (double mode; cycles) |
| `G` | Adjust geometry: a draggable crop box over the raw frame — drag a corner/edge to resize, `[` `]` tilt, `⇧`+arrows resize. **double**: drag anywhere inside the box to place the gutter line (`←`/`→` nudge it), `⇧`+drag inside to move the box (`↑`/`↓` too). **single**: drag inside to move the box. `Enter` save, `Esc` cancel, `Backspace` reset |
| `⌘S` / `Ctrl+S` | Save |

`G` adapts to `MODE=` (the `review` target passes it through automatically):

- **double** — a crop box around the spread plus the gutter (split) line inside it. Frames you never touch use the session's **consensus box**: the page mask is voted across a sample of frames (a hand or mid-turn page in any one frame is outvoted) and its minimum-area rectangle — snug fit and tilt in one shot — applies to the whole session, cached in `json/consensus_geometry.json`. Corrections propagate forward to later spreads until the next correction: the box and its tilt verbatim (measured on real sessions, the book doesn't move between corrections beyond noise), the gutter as a tracking prior. Confirming a touched box stores it as 4 corners (`crop_quad`), and P5 warps exactly that box. A background **watchdog** re-measures every frame's page boundary against the box in effect and flags frames where it shifted persistently (a bumped book) or couldn't be measured — the top bar counts alerts and `E` cycles through them, so a review pass means checking the short list, not every spread.
- **single** — the gutter overlay is hidden (each frame is already one page). The same crop box editor covers cases where the GrabCut auto-crop (P5) clips real text or wanders as page sizes vary (e.g. receipts). Confirming stores the box as 4 corners on the keyframe, and P5 warps exactly that box instead of auto-detecting; unlike double mode the box does **not** propagate. A confirmed crop is drawn as a green box during review.

### P5 — Crop

```bash
make crop VIDEO=recordings/mybook.mp4
```

Crops the book/page out of the surrounding frame. Modifies `images/` in-place. Re-run P3 to restore originals.

- `double` mode: warps the P4 manual crop box (`crop_quad`) in effect — this frame's own or one propagated from an earlier correction; otherwise warps the session's consensus box (one box voted across frames, so the output stays steady); crop bounds + per-frame page-mask detection only when no consensus could be voted
- `single` mode: warps a P4 manual crop box (`crop_quad`) if present (no propagation); otherwise uses GrabCut to segment the page from the table surface (handles rotation, works with any page color)

### P6 — Split Pages

```bash
make split VIDEO=recordings/mybook.mp4
```

- `double` mode: splits each keyframe at the center spine into left and right pages → `pages/`
- `single` mode: copies cropped images directly → `pages/`
- a spread flagged **Doc Start** in P4 passes its flag to the first page it produced (`is_doc_start` in `pages.json`), where P7 can move it to the exact page and P9 turns it into a separate PDF

### P7 — Page Review (interactive)

```bash
make page-review VIDEO=recordings/mybook.mp4
```

Tkinter GUI for dropping bad pages, nudging a page's geometry, and marking where each document starts. Review state is saved to `json/page_review.json`; Save also applies drops, re-renders adjusted pages, and stamps document starts into `json/pages.json`.

Browser version: `make page-review-web VIDEO=...` (Chrome at `http://localhost:8412` — the shared ScanStudio port — same keys and outputs; the ChromeOS-friendly path).

| Key | Action |
|-----|------|
| `X` | Toggle drop page |
| `F` | Toggle **First Page** — this page starts a new document; the note box then names it |
| `G` | Geometry: arrows translate the page inside its frame, `⇧`+arrows go 5× further, `[` `]` tilt ±0.25°, `{` `}` tilt ±1.25°, `Enter` keep, `Esc` cancel, `Backspace` reset to as-scanned |
| `⌘S` / `Ctrl+S` | Save |

**Geometry is non-destructive.** The first time a page is nudged, its untouched JPEG is stashed in `pages_orig/`; every Save re-renders `pages/<file>` from that pristine copy (rotation about the centre, white fill, same dimensions). Adjusting a page twice therefore costs one re-encode rather than two, `Backspace` restores exactly what P6 produced, and P8/P9 read `pages/` as always. P6 clears `pages_orig/` when it regenerates `pages/`, since those copies would no longer be the right baseline.

**One scan → many PDFs.** A recording is often several documents (chapters, articles, a run of receipts). P4's `6` Doc Start marks a spread; P6 lands it on the first page of that spread; `F` here moves it to the exact page and names it. P9 then writes one PDF per document alongside the combined whole-scan PDF — see below.

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

When review marked document starts, P9 *also* writes one PDF per document, named after the combined file and numbered in order, with the document's title as a slug:

```
pdf/mybook.pdf                     # the whole scan, always written
pdf/mybook_01.pdf                  # pages before the first mark (untitled)
pdf/mybook_02_chapter-one.pdf
pdf/mybook_bw_02_chapter-one.pdf   # same split for make bw
json/documents.json                # segments, titles, page ranges, PDF names
```

The combined PDF is always produced, so `make pdf`, `make bw` and anything else expecting `<name>.pdf` are unaffected. Set `SPLIT_DOCS=never` for the combined file only.

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
| `SPLIT_DOCS` | `auto` | `auto` also writes one PDF per document when review marked document starts; `never` writes only the combined PDF |
| `SAFETY_MARGIN` | `0.005` | Crop safety margin as a fraction of image dimension |
| `BW_METHOD` | `sauvola` | Binarization method: `sauvola` or `adaptive` |
| `BW_UPSCALE` | `2` | Grayscale upscale factor before thresholding (anti-aliases edges) |
| `BW_K` | `0.2` | Sauvola threshold factor (higher = thinner strokes) |
| `BLOCK_SIZE` | `51` | Threshold window size for binarization (must be odd) |
| `BW_OFFSET` | `10` | Threshold offset (`adaptive` method only) |
| `CAMERA` | `auto` | Webcam for `make live` — `auto` picks the camera that delivers 4K, or set an index (`make probe-camera` lists them) |
| `SETTLE` | `2.0` | Live: motion below this counts as "still" (book settled) |
| `TURN` | `5.0` | Live: motion above this counts as a page turn in progress |
| `SETTLE_TIME` | `0.3` | Live: seconds of stillness required before a capture fires |

## Example Walkthrough

```bash
# 1. Run the full automated + interactive pipeline
make all VIDEO=recordings/african_founders.mp4

# At P4: review keyframes in the GUI, label bad frames, insert missing ones, save with ⌘S (Ctrl+S on Linux)
# At P7: drop bad pages, nudge geometry, mark where each document starts, save with ⌘S (Ctrl+S on Linux)

# 2. Optionally produce a B&W version
make bw VIDEO=recordings/african_founders.mp4

# Output:
#   output/african_founders/pdf/african_founders.pdf
#   output/african_founders/pdf/african_founders_bw.pdf
```

## Troubleshooting

**No peaks detected** — The video may have low-contrast page turns. Check `plots/motion_plot.png` to inspect the signal. Adjust peak detection parameters in `scripts/p2_detect_peaks.py`.

**Crop removes too much / too little** — Press `G` in P4 review and fix the box: drag its corners/edges, or `⇧`+drag inside to move it; it propagates to later spreads until your next correction. For an off-center spine, drag anywhere inside the box to place the gutter line. If the consensus box looks stale after re-recording, delete `json/consensus_geometry.json` to re-vote it. (`SAFETY_MARGIN` only pads the legacy per-frame auto crop, used when no consensus exists.)

**Binarization looks wrong** — Try `BW_METHOD=adaptive`, or tune `BW_K` (Sauvola stroke weight; higher = thinner), `BLOCK_SIZE` (larger = coarser regions), and `BW_OFFSET` (adaptive only).

**PDF page order is wrong** — Page ordering follows the `pages.json` metadata. Check `json/keyframes.json` for frame numbering issues.

**`ImportError: libGL.so.1` (Linux)** — The OpenCV wheel links against system libraries a minimal image doesn't ship: `sudo apt-get install -y libgl1 libglib2.0-0`.

**No camera found on Linux** — Check the device exists (`ls /dev/video*`) and that you can read it (`make probe-camera`). "Permission denied" means your user isn't in the `video` group: `sudo usermod -aG video $USER`, then log out and back in. If a camera opens but never reaches 4K, run `v4l2-ctl --list-formats-ext -d /dev/videoN` (from `v4l-utils`) to see which modes it really offers — many webcams expose 4K only under MJPG, which `make live` already requests.

**`make live` is silent on capture (Linux)** — The chime falls back to the terminal bell when no sound player is found. Install one of `pulseaudio-utils` (`paplay`), `alsa-utils` (`aplay`), or `libcanberra-gtk3-module` (`canberra-gtk-play`); the sound theme comes from `sound-theme-freedesktop`.

## License

Apache 2.0
