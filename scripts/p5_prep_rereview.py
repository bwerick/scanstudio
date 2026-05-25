#!/usr/bin/env python3
"""
Phase 5: Prep and Re-Review

Backs up the current review, promotes final_keyframes.json back to
keyframes/keyframes.json, then launches Phase 4 for re-review.

Usage:
  python scripts/p5_prep_rereview.py output/audiq5 recordings/audiq5.mp4
"""

import argparse
import json
import shutil
import sys
import subprocess
from pathlib import Path
from datetime import datetime

from utils import log, check_overwrite


def main():
    parser = argparse.ArgumentParser(description="Prep and re-review keyframes")
    parser.add_argument("output_dir", help="Base output directory (e.g. output/audiq5)")
    parser.add_argument("video", help="Path to original video file")
    args = parser.parse_args()

    base = Path(args.output_dir)
    review_dir = base / "review"
    keyframes_dir = base / "keyframes"

    final_path = review_dir / "final_keyframes.json"
    kf_path = keyframes_dir / "keyframes.json"

    if not final_path.exists():
        log(f"ERROR: {final_path} not found. Run Phase 4 review first.")
        sys.exit(1)

    log("=" * 60)
    log("PHASE 5: Prep for Re-Review")
    log("=" * 60)

    # Backup entire review directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    review_backup = base / f"review_pass_{timestamp}"
    shutil.copytree(review_dir, review_backup)
    log(f"Backed up review/ → {review_backup.name}/")

    # Load final keyframes
    final = json.loads(final_path.read_text())
    log(f"Loaded {len(final)} final keyframes from {final_path}")

    # Preserve crop bounds from previous review
    review_log_path = review_dir / "review_log.json"
    previous_crop = {}
    if review_log_path.exists():
        prev_log = json.loads(review_log_path.read_text())
        previous_crop = prev_log.get("crop_bounds", {})
        if previous_crop:
            log(f"Preserving crop bounds from previous review")

    # Sort by frame_index
    final.sort(key=lambda x: x["frame_index"])

    # Normalize fields and re-number
    normalized = []
    for i, kf in enumerate(final):
        entry = {
            "spread_index": i + 1,
            "frame_index": kf["frame_index"],
            "time_sec": kf.get("time_sec", round(kf["frame_index"] / 30.0, 2)),
            "motion_value": kf.get("motion_value", 0.0),
            "sharpness": kf.get("sharpness", 0.0),
            "filename": kf["filename"],
            "spread_start": kf.get("spread_start", kf["frame_index"]),
            "spread_end": kf.get("spread_end", kf["frame_index"]),
            "spread_duration": kf.get("spread_duration", 0.0),
        }
        # Preserve cover and crop flags
        if kf.get("is_cover"):
            entry["is_cover"] = True
        if kf.get("crop_bounds"):
            entry["crop_bounds"] = kf["crop_bounds"]
        normalized.append(entry)

    # Backup old keyframes.json
    if kf_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = keyframes_dir / f"keyframes_backup_{timestamp}.json"
        shutil.copy2(kf_path, backup)
        log(f"Backed up old keyframes.json to {backup.name}")

    # Write new keyframes.json
    kf_path.write_text(json.dumps(normalized, indent=2))
    log(f"Wrote {len(normalized)} keyframes to {kf_path}")

    log("")
    log("Launching Phase 4 for re-review...")
    log("=" * 60)

    # Find the p4 script relative to this script
    scripts_dir = Path(__file__).parent
    p4_script = scripts_dir / "p4_review_keyframes.py"

    if not p4_script.exists():
        log(f"ERROR: {p4_script} not found")
        sys.exit(1)

    # Launch p4 as a subprocess
    subprocess.run([sys.executable, str(p4_script), str(base), args.video])


if __name__ == "__main__":
    main()
