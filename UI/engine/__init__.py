"""
ScanStudio Engine — Core pipeline logic as importable modules.

Used by both the CLI scripts and the web UI.
"""

from pathlib import Path


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
        for d in [self.images, self.pages, self.plots, self.data,
                  self.json, self.reports, self.pdf]:
            d.mkdir(parents=True, exist_ok=True)
        return self

    def ensure(self, *dirs: str):
        for name in dirs:
            getattr(self, name).mkdir(parents=True, exist_ok=True)
        return self


def derive_output_dir(video_path: str, override: str | None = None) -> Path:
    if override:
        return Path(override)
    return Path.cwd() / "output" / Path(video_path).stem
