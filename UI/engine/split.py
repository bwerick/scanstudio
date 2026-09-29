"""
Page splitting.

Cuts a cropped spread into left and right pages along the gutter. The
gutter carries its own tilt, so each half is warped from its own quad —
a leaning spine yields two upright pages with no lost content.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np

from .geometry import (split_quads, quad_to_px, warp_quad, normalize_gutter,
                       DEFAULT_GUTTER)


FULL_QUAD = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def split_at_gutter(img: np.ndarray, gutter=0.5,
                     quad=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Split an image along the gutter.

    `gutter` may be a bare float (legacy position) or {"pos", "angle"}.
    `quad` is the region to cut, defaulting to the whole image — after
    Phase 5 the crop is already applied, so the whole frame *is* the box.
    """
    h, w = img.shape[:2]
    g = normalize_gutter(gutter) or DEFAULT_GUTTER
    box = quad if quad is not None else FULL_QUAD

    # Untilted: a plain slice is exact and much cheaper than a warp.
    if quad is None and abs(g["top"] - g["bot"]) < 1e-4:
        x = int(round(float(np.clip(g["top"], 0.02, 0.98)) * w))
        return img[:, :x], img[:, x:]

    lq, rq = split_quads(box, g)
    return (warp_quad(img, quad_to_px(lq, w, h)),
            warp_quad(img, quad_to_px(rq, w, h)))


def resolve_gutter(keyframes: list, idx: int):
    """
    The gutter in effect at `idx`: its own, else the most recent one set
    before it. Books drift slowly, so one correction should carry forward
    until another replaces it.
    """
    own = keyframes[idx].get("gutter", keyframes[idx].get("gutter_pct"))
    if own is not None:
        return normalize_gutter(own)
    for prev in reversed(keyframes[:idx]):
        g = prev.get("gutter", prev.get("gutter_pct"))
        if g is not None:
            return normalize_gutter(g)
    return None


def split_all_pages(keyframes: list, images_dir: Path,
                     output_dir: Path, mode: str = "double",
                     gutter=0.5, jpeg_quality: int = 92,
                     on_progress=None, already_cropped: bool = True) -> list[dict]:
    """
    Split every keyframe into pages.

    mode "double" cuts spreads at the gutter; "single" passes each frame
    through as one page. Covers are never split.

    already_cropped=True (the normal flow) treats each image as the crop box
    itself, since the crop pass has already warped it. False splits straight
    from the original frame inside the stored crop_quad — no separate crop
    pass needed, and nothing is modified on disk.
    """
    from .geometry import resolve_crop_quad
    output_dir.mkdir(parents=True, exist_ok=True)
    default_g = normalize_gutter(gutter) or DEFAULT_GUTTER

    page_list = []
    page_num = 0

    for i, kf in enumerate(keyframes):
        img_path = images_dir / kf["filename"]
        if not img_path.exists():
            continue

        frame_idx = kf.get("frame_index", i)
        is_cover = kf.get("is_cover", False)

        if mode == "single":
            page_num += 1
            fn = f"frame{frame_idx:06d}_page.jpg"
            shutil.copy2(img_path, output_dir / fn)
            page_list.append({"page_num": page_num, "type": "page",
                              "filename": fn, "source": kf["filename"]})

        elif is_cover:
            is_last = (kf is keyframes[-1])
            ctype = "backcover" if is_last else "cover"
            page_num += 1
            fn = f"frame{frame_idx:06d}_{ctype}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            cv2.imwrite(str(output_dir / fn), img,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            page_list.append({"page_num": page_num, "type": ctype,
                              "filename": fn, "source": kf["filename"]})

        else:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            g = resolve_gutter(keyframes, i) or default_g
            box = None if already_cropped else resolve_crop_quad(keyframes, i)[0]
            left, right = split_at_gutter(img, g, quad=box)

            for side, half in (("left", left), ("right", right)):
                page_num += 1
                fn = f"frame{frame_idx:06d}_{side}.jpg"
                cv2.imwrite(str(output_dir / fn), half,
                            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                page_list.append({
                    "page_num": page_num, "type": side, "filename": fn,
                    "source": kf["filename"],
                    "gutter_top": round(g["top"], 4),
                    "gutter_bot": round(g["bot"], 4),
                })

        if on_progress:
            on_progress(i + 1, len(keyframes))

    return page_list
