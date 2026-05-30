#!/usr/bin/env python3
"""Phase 8: Binarize Pages (Optional)
Usage: python scripts/p8_binarize.py output/mybook"""

import argparse, json, sys, time
from pathlib import Path
import cv2
from utils import log, ProjectPaths, ensure_dir, check_overwrite_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--block-size", type=int, default=51)
    parser.add_argument("--offset", type=int, default=10)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    args = parser.parse_args()
    if args.block_size % 2 == 0: args.block_size += 1

    log("=" * 60); log("PHASE 8: Binarize Pages"); log("=" * 60)
    paths = ProjectPaths(args.output_dir)
    bw_dir = ensure_dir(paths.base / "bw")
    pages = json.loads((paths.json / "pages.json").read_text())
    if not check_overwrite_dir(bw_dir): return

    log(f"Binarizing {len(pages)} pages (block={args.block_size}, offset={args.offset})...")
    t0 = time.time()
    for pg in pages:
        src = paths.pages / pg["filename"]
        if not src.exists(): continue
        img = cv2.imread(str(src))
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, args.block_size, args.offset)
        cv2.imwrite(str(bw_dir / pg["filename"]), binary, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

    meta = {"block_size": args.block_size, "offset": args.offset}
    (paths.json / "bw_metadata.json").write_text(json.dumps(meta, indent=2))
    log(f"  Done in {time.time()-t0:.1f}s")
    log("PHASE 8 COMPLETE")

if __name__ == "__main__": main()
