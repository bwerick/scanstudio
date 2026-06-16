#!/usr/bin/env python3
"""Phase 8: Binarize Pages (Optional)
Usage: python scripts/p8_binarize.py output/mybook
       python scripts/p8_binarize.py output/mybook --method adaptive   # old behaviour

Crispens text for the B&W PDF. The grain in a naive adaptive threshold comes
from two places, neither of them the threshold itself:

  * Thresholding at the page's native ~624px width quantizes each stroke edge
    onto a coarse grid, so curves come out jagged. Upscaling the *grayscale*
    first (cubic, which anti-aliases) gives smooth letter contours once binarized.
  * Saving bitonal pages as JPEG rings ("mosquito noise") around every sharp
    black/white edge and bloats the file. We write lossless PNG instead.

Sauvola local thresholding (vs Gaussian adaptiveThreshold) tolerates an
illumination/gutter-shadow gradient without smearing it into black blotches,
and a 3x3 median de-speckles both paths without eroding strokes. --method
adaptive restores the old algorithm for A/B comparison."""

import argparse, json, sys, time
from pathlib import Path
import cv2
import numpy as np
from utils import log, ProjectPaths, ensure_dir, check_overwrite_dir

def binarize(img, method, block_size, offset, upscale, k):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if upscale and upscale != 1.0:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale,
                          interpolation=cv2.INTER_CUBIC)
    # Track the upscale so the window spans the same physical area; force odd.
    win = max(3, int(round(block_size * (upscale or 1.0)))) | 1
    if method == "adaptive":
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, win, offset)
    else:
        from skimage.filters import threshold_sauvola  # optional dep; sauvola only
        t = threshold_sauvola(gray, window_size=win, k=k)
        binary = ((gray > t) * 255).astype(np.uint8)
    return cv2.medianBlur(binary, 3)   # de-speckle without eroding strokes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--method", choices=["sauvola", "adaptive"], default="sauvola")
    parser.add_argument("--block-size", type=int, default=51)
    parser.add_argument("--offset", type=int, default=10,
                        help="adaptive only: threshold bias")
    parser.add_argument("--sauvola-k", type=float, default=0.1,
                        help="sauvola only: lower = bolder/more ink kept")
    parser.add_argument("--upscale", type=float, default=2.0,
                        help="grayscale upscale before thresholding (anti-aliases edges)")
    args = parser.parse_args()
    if args.block_size % 2 == 0: args.block_size += 1

    log("=" * 60); log("PHASE 8: Binarize Pages"); log("=" * 60)
    paths = ProjectPaths(args.output_dir)
    bw_dir = ensure_dir(paths.base / "bw")
    pages = json.loads((paths.json / "pages.json").read_text())
    if not check_overwrite_dir(bw_dir): return
    # Clear stale outputs: filenames now end in .png, so a prior .jpg run would
    # otherwise leave orphans that P9 picks up before the fresh PNGs.
    for old in bw_dir.glob("*"):
        if old.is_file(): old.unlink()

    log(f"Binarizing {len(pages)} pages "
        f"(method={args.method}, block={args.block_size}, upscale={args.upscale})...")
    t0 = time.time()
    for pg in pages:
        src = paths.pages / pg["filename"]
        if not src.exists(): continue
        img = cv2.imread(str(src))
        if img is None: continue
        binary = binarize(img, args.method, args.block_size, args.offset,
                          args.upscale, args.sauvola_k)
        out = bw_dir / (Path(pg["filename"]).stem + ".png")
        cv2.imwrite(str(out), binary, [cv2.IMWRITE_PNG_COMPRESSION, 6])

    meta = {"method": args.method, "block_size": args.block_size, "offset": args.offset,
            "sauvola_k": args.sauvola_k, "upscale": args.upscale}
    (paths.json / "bw_metadata.json").write_text(json.dumps(meta, indent=2))
    log(f"  Done in {time.time()-t0:.1f}s")
    log("PHASE 8 COMPLETE")

if __name__ == "__main__": main()
