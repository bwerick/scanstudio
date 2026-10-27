#!/usr/bin/env python3
"""Phase 3.5: Flag likely duplicate keyframes before human review.

This image-only pass compares nearby selected keyframes. It writes a report to
``json/frame_quality.json`` and adds non-destructive ``quality_flags`` to the
corresponding entries in ``json/keyframes.json``. It never drops a frame.
"""

import argparse
import json
from pathlib import Path

from frame_quality import find_likely_duplicates, load_preview
from utils import ProjectPaths, log


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3.5: Flag duplicate keyframes")
    parser.add_argument("output_dir", help="Base output directory")
    parser.add_argument(
        "--max-neighbor-gap", type=int, default=1,
        help="Compare a frame with this many earlier frames (default: 1)",
    )
    parser.add_argument(
        "--hash-threshold", type=int, default=4,
        help="Maximum 64-bit dHash distance for a likely duplicate (default: 4)",
    )
    parser.add_argument(
        "--difference-threshold", type=float, default=0.02,
        help="Maximum normalized pixel difference for a likely duplicate (default: 0.02)",
    )
    args = parser.parse_args()

    if args.max_neighbor_gap < 1:
        parser.error("--max-neighbor-gap must be at least 1")

    paths = ProjectPaths(args.output_dir)
    keyframe_path = paths.json / "keyframes.json"
    if not keyframe_path.exists():
        parser.error(f"{keyframe_path} not found. Run Phase 3 first.")

    log("=" * 60)
    log("PHASE 3.5: Frame Quality Flags")
    log("=" * 60)
    keyframes = json.loads(keyframe_path.read_text())
    previews = [load_preview(paths.images / kf["filename"]) for kf in keyframes]
    unreadable = sum(image is None for image in previews)
    flags = find_likely_duplicates(
        previews,
        max_neighbor_gap=args.max_neighbor_gap,
        hash_threshold=args.hash_threshold,
        difference_threshold=args.difference_threshold,
    )

    for keyframe in keyframes:
        keyframe.pop("quality_flags", None)

    report_flags = []
    for flag in flags:
        current = keyframes[int(flag["current_index"])]
        match = keyframes[int(flag["match_index"])]
        detail = {
            "kind": "likely_duplicate",
            "matches_filename": match["filename"],
            "matches_frame_index": match["frame_index"],
            "hash_distance": flag["hash_distance"],
            "pixel_difference": round(float(flag["pixel_difference"]), 4),
        }
        current["quality_flags"] = [detail]
        report_flags.append({"filename": current["filename"], "frame_index": current["frame_index"], **detail})
        log(
            f"  FLAG {current['filename']} ~= {match['filename']} "
            f"(dHash={detail['hash_distance']}, diff={detail['pixel_difference']:.4f})"
        )

    report = {
        "phase": "P3.5",
        "purpose": "Flag likely duplicate keyframes for human review; never delete automatically.",
        "config": {
            "max_neighbor_gap": args.max_neighbor_gap,
            "hash_threshold": args.hash_threshold,
            "difference_threshold": args.difference_threshold,
        },
        "summary": {
            "keyframes_checked": len(keyframes),
            "likely_duplicates": len(report_flags),
            "unreadable_images": unreadable,
        },
        "flags": report_flags,
    }
    paths.json.mkdir(parents=True, exist_ok=True)
    keyframe_path.write_text(json.dumps(keyframes, indent=2) + "\n")
    report_path = paths.json / "frame_quality.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    log(f"  Checked {len(keyframes)} keyframes; {len(report_flags)} likely duplicate(s).")
    if unreadable:
        log(f"  WARNING: {unreadable} image(s) could not be read.")
    log(f"  Report: {report_path}")
    log("PHASE 3.5 COMPLETE")


if __name__ == "__main__":
    main()
