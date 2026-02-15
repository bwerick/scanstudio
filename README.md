# Scan Studio

A data preparation pipeline for converting book scanning videos into high-quality PDFs. Place your videos in the `Videos/` directory, run `make`, and get PDFs of individual book pages.

## Overview

Scan Studio processes book scanning videos through multiple stages:
1. **Frame Extraction** - Extracts all frames from video files
2. **Keyframe Detection** - Identifies unique pages (removes duplicates)
3. **Page Cropping** - Splits two-page spreads into left and right pages
4. **PDF Generation** - Creates a final PDF from the cropped pages

## Prerequisites

- Python 3.x
- ImageMagick (for PDF generation)
- FFmpeg (for video processing)
- Required Python packages (see [Installation](#installation))

## Installation

Install Python dependencies:

```bash
make install
```

This will install all packages from `requirements.txt`.

## Directory Structure

```
scanstudio/
├── Videos/              # Place your .mp4 or .mov video files here
├── test_frames/         # Output directory (auto-generated)
│   └── <book_name>/
│       ├── *.jpg        # Extracted frames
│       ├── keyframes/   # Deduplicated keyframes
│       ├── left/        # Left page crops
│       ├── right/       # Right page crops
│       └── cropped/     # Staged pages for PDF
├── frameextraction.py
├── keyframe_extraction.py
├── batch_image_cropper.py
├── streamlit_keyframes.py
└── Makefile
```

## Quick Start

1. Place your book scanning videos in the `Videos/` directory
2. Run the default pipeline:
   ```bash
   make
   ```
3. Find processed keyframes in `test_frames/<book_name>/keyframes/`

## Usage

### Basic Commands

| Command | Description |
|---------|-------------|
| `make` | Run the default pipeline (extracts frames and keyframes) |
| `make frames` | Extract frames from all videos |
| `make keyframes` | Extract keyframes (deduplicated pages) |
| `make frames-one BOOK="..."` | Extract frames for a single book (video stem) |
| `make keyframes-one BOOK="..."` | Extract keyframes for a single book |
| `make left BOOK="..."` | Crop left pages for a single book (interactive) |
| `make right BOOK="..."` | Crop right pages for a single book (interactive) |
| `make cropped BOOK="..."` | Stage cropped pages for a single book |
| `make pdf BOOK="..."` | Generate a PDF for a single book |

### Processing Individual Books

To process a specific book (including names with spaces), pass it via `BOOK`:

```bash
make frames-one BOOK="African Founders"
make keyframes-one BOOK="African Founders"
make left BOOK="African Founders"
make right BOOK="African Founders"
make cropped BOOK="African Founders"
make pdf BOOK="African Founders"
```


### Utility Commands

| Command | Description |
|---------|-------------|
| `make install` | Install Python dependencies |
| `make clean` | Remove all generated files and PDFs |
| `make list-books` | List discovered book names (video stems) |
| `make list-videos` | List discovered video files |


## Pipeline Details

### 1. Frame Extraction

```bash
make frames
```

- Processes all `.mp4` and `.mov` files in `Videos/`
- Outputs frames as `.jpg` files to `test_frames/<book_name>/`
- Uses `frameextraction.py`

### 2. Keyframe Detection

```bash
make keyframes
```

- Analyzes extracted frames to identify unique pages
- Removes duplicate/similar frames (e.g., when camera is idle)
- Outputs to `test_frames/<book_name>/keyframes/`
- Uses `keyframe_extraction.py`

### 3. Page Cropping

```bash
make left BOOK="<book_name>"
make right BOOK="<book_name>"
```

- Splits two-page spreads into individual pages
- Left pages go to `test_frames/<book_name>/left/`
- Right pages go to `test_frames/<book_name>/right/`
- Uses `batch_image_cropper.py`

### 4. Page Staging

```bash
make cropped BOOK="<book_name>"
```

- Moves all left and right pages into `cropped/` directory
- Prepares images for PDF generation
- **Note:** This moves (not copies) files from left/right directories

### 5. PDF Generation

```bash
make pdf BOOK="<book_name>"
```

- Combines all cropped pages into a single PDF
- Resizes images to max width of 512px (maintains aspect ratio)
- Uses 95% JPEG quality with progressive encoding
- Output: `<book_name>.pdf` in the project root

## Configuration

Key variables in the Makefile:

```makefile
VIDEO_DIR   := Videos        # Input video directory
OUTPUT_DIR  := test_frames   # Output directory for frames
```

To change these, edit the Makefile or override when calling make:

```bash
make VIDEO_DIR=my_videos OUTPUT_DIR=output frames
```

## Dependencies

The pipeline uses the following Python scripts:

- **frameextraction.py** - Extracts frames from video files
- **keyframe_extraction.py** - Detects unique keyframes
- **batch_image_cropper.py** - Crops left/right pages from spreads
- **streamlit_keyframes.py** - Interactive viewer for keyframes

## Model Weights

The pipeline may download SAM (Segment Anything Model) weights if needed:

```bash
../sam_vit_h_4b8939.pth
```

This will be automatically downloaded from Facebook AI Research if required.

## Troubleshooting

### No frames extracted
- Verify video files are in `Videos/` directory
- Check video format (must be `.mp4` or `.mov`)
- Ensure FFmpeg is installed

### No keyframes detected
- Verify frames were extracted successfully
- Check `test_frames/<book_name>/` for `.jpg` files

### PDF generation fails
- Ensure ImageMagick is installed: `magick --version`
- Verify cropped images exist in `test_frames/<book_name>/cropped/`

### Debugging

Print internal make variables:

```bash
make list-books        # List discovered book names (video stems)
make list-videos       # List discovered video files
```
## License
Apache 2.0 License