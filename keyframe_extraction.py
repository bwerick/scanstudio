import argparse
import logging
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import flordb as flor

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def compute_histogram(image):
    hist = cv2.calcHist(
        [image], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256]
    )
    if hist is None:
        return None
    cv2.normalize(hist, hist)
    return hist


def histogram_similarity(hist_a, hist_b) -> float:
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def focus_measure(image) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def create_group(
    hist, frame_path: Path, sharpness: float, sharpness_floor: float
) -> dict:
    best_path = frame_path if sharpness >= sharpness_floor else None
    best_sharpness = sharpness if sharpness >= sharpness_floor else -1.0
    return {
        "ref_hist": hist,
        "best_path": best_path,
        "best_sharpness": best_sharpness,
        "fallback_path": frame_path,
    }


def collect_frame_dirs(root: Path) -> List[Path]:
    return sorted(
        d
        for d in root.iterdir()
        if d.is_dir()
        and any(
            f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS for f in d.iterdir()
        )
    )


def iter_frames(frame_dir: Path) -> List[Path]:
    return sorted(
        p
        for p in frame_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def process_directory(
    frame_dir: Path,
    output_subdir: str,
    min_similarity: float,
    sharpness_floor: float,
) -> int:
    frames = iter_frames(frame_dir)
    if not frames:
        logging.info("Skipping %s (no frames found).", frame_dir)
        return 0

    current_group: Optional[dict] = None
    selected: List[Path] = []

    for frame_path in flor.loop("frame", frames):
        image = cv2.imread(str(frame_path))
        if image is None:
            logging.warning("Failed to read %s, skipping.", frame_path)
            continue

        hist = compute_histogram(image)
        if hist is None:
            logging.warning("Failed to compute histogram for %s, skipping.", frame_path)
            continue

        sharpness = focus_measure(image)

        if current_group is None:
            current_group = create_group(hist, frame_path, sharpness, sharpness_floor)
            continue

        similarity = histogram_similarity(current_group["ref_hist"], hist)

        if similarity < min_similarity:
            chosen = current_group["best_path"] or current_group["fallback_path"]
            if chosen:
                selected.append(chosen)
            current_group = create_group(hist, frame_path, sharpness, sharpness_floor)
            continue

        current_group["fallback_path"] = frame_path
        if sharpness >= sharpness_floor and sharpness > current_group["best_sharpness"]:
            current_group["best_path"] = frame_path
            current_group["best_sharpness"] = sharpness
            current_group["ref_hist"] = hist

    if current_group:
        chosen = current_group["best_path"] or current_group["fallback_path"]
        if chosen:
            selected.append(chosen)

    if not selected:
        logging.info("No keyframes selected for %s.", frame_dir)
        return 0

    output_dir = frame_dir / output_subdir
    print(f"Saving keyframes to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.jpg"):
        old.unlink()

    for frame_path in selected:
        destination = output_dir / frame_path.name
        shutil.copy2(frame_path, destination)

    logging.info("Saved %d keyframes for %s.", len(selected), frame_dir)
    return len(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one sharp keyframe per page segment."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--frames-root", type=Path, help="Root folder containing frame subdirectories."
    )
    group.add_argument(
        "--frames-dir", type=Path, help="Single frame directory to process."
    )
    parser.add_argument(
        "--output-subdir",
        default="keyframes",
        help="Output folder name (default: keyframes).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if args.frames_dir:
        directories: Iterable[Path] = [args.frames_dir.resolve()]
    else:
        root = args.frames_root.resolve()
        if not root.exists():
            raise FileNotFoundError(f"Frames root {root} does not exist.")
        directories = collect_frame_dirs(root)

    min_similarity = flor.arg("min-similarity", 0.92)
    sharpness_floor = flor.arg("sharpness-floor", 0.1)

    total = 0
    for directory in flor.loop("document", directories):
        total += process_directory(
            directory,
            "keyframes",
            min_similarity,
            sharpness_floor,
        )

    logging.info("Total keyframes saved: %d", total)


if __name__ == "__main__":
    main()
