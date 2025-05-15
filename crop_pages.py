import os
import numpy as np
from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import flor

input_dir = flor.arg("input_dir", os.path.join("test_frames", "WebOfBelief"))


def load_image_sequence(directory=input_dir):
    """Load all image frames in sorted order from a directory."""
    image_paths = os.listdir(directory)
    image_paths = sorted(
        [p for p in image_paths if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    return image_paths


def frames_generator(image_paths, start=0, stop=None, step=1):
    """Yield frames from image paths, supporting start, stop, and step (like slicing)."""
    if stop is None:
        stop = len(image_paths)
    for idx in range(start, stop, step):
        yield cv2.imread(os.path.join(input_dir, str(image_paths[idx])))


def track_pages_from_reference(
    frames,
    median_frame_rgb,
    left_anchor_px,
    right_anchor_px,
    box_size_ratio,
):
    """Track and crop pages in all frames using optical flow from the reference frame."""
    w = median_frame_rgb.shape[1]
    h = median_frame_rgb.shape[0]
    box_w, box_h = int(box_size_ratio[0] * w), int(box_size_ratio[1] * h)

    gray_ref = cv2.cvtColor(median_frame_rgb, cv2.COLOR_BGR2GRAY)
    left_pt = np.array([left_anchor_px], dtype=np.float32)
    right_pt = np.array([right_anchor_px], dtype=np.float32)

    left_crops, right_crops = [], []

    for f in flor.loop("frame", frames):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        tracked_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            gray_ref, gray, np.vstack([left_pt, right_pt]), None
        )

        left_tracked = tracked_pts[0] if status[0] else left_pt[0]
        right_tracked = tracked_pts[1] if status[1] else right_pt[0]

        lx, ly = int(left_tracked[0]), int(left_tracked[1])
        rx, ry = int(right_tracked[0]), int(right_tracked[1])

        left_crop = f[ly : ly + box_h, lx : lx + box_w]
        right_crop = f[ry : ry + box_h, rx : rx + box_w]

        left_crops.append(left_crop)
        right_crops.append(right_crop)

    return left_crops, right_crops


image_paths = load_image_sequence()
median_idx = len(image_paths) // 2

# We'll pause here for user input
median_frame = cv2.imread(os.path.join(input_dir, str(image_paths[median_idx])))
median_frame_rgb = cv2.cvtColor(median_frame, cv2.COLOR_BGR2RGB)

# Display the median frame for bounding box entry
plt.figure(figsize=(10, 6))
plt.imshow(median_frame_rgb)
plt.title(f"Select anchor points on median frame: {image_paths[median_idx]}")
plt.axis("off")
# plt.show()

# Prompt user to select two anchor points (left and right) on the displayed image
print(
    "Please click on the LEFT anchor point, then the RIGHT anchor point in the matplotlib window."
)
anchors = plt.ginput(2, timeout=0)
if len(anchors) != 2:
    raise ValueError("You must select exactly two anchor points.")
plt.close()
left_anchor = anchors[0]
right_anchor = anchors[1]
left_anchor = (int(left_anchor[0]), int(left_anchor[1]))
right_anchor = (int(right_anchor[0]), int(right_anchor[1]))


# Infer bbox size from the selected points
box_size_ratio = (
    abs(left_anchor[0] - right_anchor[0]) / median_frame.shape[1],  # width
    0.55,
)  # height


# right_anchor = (right_anchor[0], left_anchor[1])  # Align y-coordinates for right anchor

# Call the tracking function correctly
left_crops, right_crops = track_pages_from_reference(
    frames_generator(image_paths),  # all frames
    median_frame_rgb=median_frame_rgb,
    left_anchor_px=left_anchor,
    right_anchor_px=right_anchor,
    box_size_ratio=box_size_ratio,
)

# Save cropped pages to output folders
left_dir = os.path.join("left_pages", os.path.basename(input_dir))
right_dir = os.path.join("right_pages", os.path.basename(input_dir))
os.makedirs(left_dir, exist_ok=True)
os.makedirs(right_dir, exist_ok=True)

left_paths = []
right_paths = []

for i, (l_crop, r_crop) in enumerate(zip(left_crops, right_crops)):
    l_path = os.path.join(left_dir, f"{i:04d}.jpg")
    r_path = os.path.join(right_dir, f"{i:04d}.jpg")
    cv2.imwrite(str(l_path), l_crop)
    cv2.imwrite(str(r_path), r_crop)
