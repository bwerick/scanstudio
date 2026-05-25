"""
Shared utilities for the ScanStudio pipeline.

Provides logging, overwrite prompts, and path helpers used by all scripts.
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

    If output_dir_override is provided, use that instead.
    Otherwise: recordings/foo.mp4 → output/foo/
    """
    if output_dir_override:
        return Path(output_dir_override)

    video_name = Path(video_path).stem  # e.g. "bookflip_20260220_140323"

    # Walk up from the video to find the project root (where 'scripts/' or 'output/' lives)
    # Fallback: use the current working directory
    project_root = Path.cwd()

    return project_root / "output" / video_name


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_overwrite(path: Path) -> bool:
    """
    If path exists, prompt the user to confirm overwrite.
    Returns True if we should proceed (file doesn't exist or user said yes).
    Returns False if user declined.
    """
    if not path.exists():
        return True

    while True:
        response = (
            input(f"  '{path}' already exists. Overwrite? [y/n]: ").strip().lower()
        )
        if response in ("y", "yes"):
            return True
        elif response in ("n", "no"):
            return False
        else:
            print("  Please enter 'y' or 'n'.")


def check_overwrite_dir(dir_path: Path) -> bool:
    """
    If directory exists and has files, prompt to confirm overwrite.
    Returns True if we should proceed.
    """
    if not dir_path.exists():
        return True

    files = list(dir_path.iterdir())
    if not files:
        return True

    while True:
        response = (
            input(f"  '{dir_path}' already has {len(files)} files. Overwrite? [y/n]: ")
            .strip()
            .lower()
        )
        if response in ("y", "yes"):
            return True
        elif response in ("n", "no"):
            return False
        else:
            print("  Please enter 'y' or 'n'.")
