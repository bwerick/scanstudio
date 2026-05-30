"""
Shared utilities for the ScanStudio pipeline.

Provides logging, overwrite prompts, and path helpers used by all scripts.

Directory structure per video:
  output/<video_name>/
  ├── images/    # keyframe images (4K), modified in-place
  ├── pages/     # split/cropped individual pages
  ├── plots/     # all diagnostic plots
  ├── data/      # .npy signal data
  ├── json/      # all metadata, configs, logs
  ├── reports/   # .md and .txt reports
  └── pdf/       # final PDFs
"""

import sys
import time
from pathlib import Path


def log(msg: str):
    """Print a timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def derive_output_dir(video_path: str, output_dir_override: str | None = None) -> Path:
    """
    Derive the per-video output directory from the video filename.
    recordings/foo.mp4 → output/foo/
    """
    if output_dir_override:
        return Path(output_dir_override)
    video_name = Path(video_path).stem
    return Path.cwd() / "output" / video_name


class ProjectPaths:
    """Standardized paths for a video project's output."""

    def __init__(self, output_dir: str | Path):
        self.base = Path(output_dir)
        self.images = self.base / "images"
        self.pages = self.base / "pages"
        self.plots = self.base / "plots"
        self.data = self.base / "data"
        self.json = self.base / "json"
        self.reports = self.base / "reports"
        self.pdf = self.base / "pdf"

    def ensure_all(self):
        """Create all subdirectories."""
        for d in [self.images, self.pages, self.plots, self.data,
                  self.json, self.reports, self.pdf]:
            d.mkdir(parents=True, exist_ok=True)
        return self

    def ensure(self, *dirs: str):
        """Create specific subdirectories by name."""
        for name in dirs:
            getattr(self, name).mkdir(parents=True, exist_ok=True)
        return self


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_overwrite(path: Path) -> bool:
    """Prompt to confirm overwrite if path exists."""
    if not path.exists():
        return True
    while True:
        response = input(f"  '{path}' already exists. Overwrite? [y/n]: ").strip().lower()
        if response in ("y", "yes"):
            return True
        elif response in ("n", "no"):
            return False
        print("  Please enter 'y' or 'n'.")


def check_overwrite_dir(dir_path: Path) -> bool:
    """Prompt to confirm overwrite if directory has files."""
    if not dir_path.exists():
        return True
    files = list(dir_path.iterdir())
    if not files:
        return True
    while True:
        response = input(f"  '{dir_path}' already has {len(files)} files. Overwrite? [y/n]: ").strip().lower()
        if response in ("y", "yes"):
            return True
        elif response in ("n", "no"):
            return False
        print("  Please enter 'y' or 'n'.")
