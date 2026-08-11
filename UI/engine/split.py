"""
Page splitting.

Splits a cropped spread image into left and right pages at
the gutter point. For single-page mode, passes through as-is.
"""

import cv2
import numpy as np
from pathlib import Path


def split_at_gutter(
    img: np.ndarray, gutter_pct: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split image at the gutter position.

    Args:
        img: cropped spread image (BGR)
        gutter_pct: horizontal position of gutter (0.0 to 1.0, default 0.5 = center)

    Returns:
        (left_page, right_page)
    """
    h, w = img.shape[:2]
    split_x = int(w * gutter_pct)
    return img[:, :split_x], img[:, split_x:]


def split_all_pages(
    keyframes: list[dict],
    images_dir: Path,
    output_dir: Path,
    mode: str = "double",
    gutter_pct: float = 0.5,
    jpeg_quality: int = 92,
    on_progress=None,
) -> list[dict]:
    """
    Split all keyframes into individual pages.

    Args:
        keyframes: list of keyframe metadata dicts
        images_dir: directory containing cropped keyframe images
        output_dir: directory to save page images
        mode: "double" (split) or "single" (pass-through)
        gutter_pct: where to split (0.0-1.0)
        jpeg_quality: output JPEG quality
        on_progress: callback(current, total)

    Returns:
        list of page metadata dicts
    """
    import shutil

    output_dir.mkdir(parents=True, exist_ok=True)

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
            page_list.append(
                {
                    "page_num": page_num,
                    "type": "page",
                    "filename": fn,
                    "source": kf["filename"],
                }
            )

        elif is_cover:
            is_last = kf is keyframes[-1]
            ctype = "backcover" if is_last else "cover"
            page_num += 1
            fn = f"frame{frame_idx:06d}_{ctype}.jpg"
            img = cv2.imread(str(img_path))
            cv2.imwrite(
                str(output_dir / fn), img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            page_list.append(
                {
                    "page_num": page_num,
                    "type": ctype,
                    "filename": fn,
                    "source": kf["filename"],
                }
            )

        else:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Use per-frame gutter if set, otherwise walk backward, otherwise global default
            g = kf.get("gutter_pct")
            if g is None:
                for prev in reversed(keyframes[:i]):
                    if prev.get("gutter_pct") is not None:
                        g = prev["gutter_pct"]
                        break
            if g is None:
                g = gutter_pct
            left, right = split_at_gutter(img, g)

            for side, half in [("left", left), ("right", right)]:
                page_num += 1
                fn = f"frame{frame_idx:06d}_{side}.jpg"
                cv2.imwrite(
                    str(output_dir / fn), half, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                )
                page_list.append(
                    {
                        "page_num": page_num,
                        "type": side,
                        "filename": fn,
                        "source": kf["filename"],
                    }
                )

        if on_progress:
            on_progress(i + 1, len(keyframes))

    return page_list
