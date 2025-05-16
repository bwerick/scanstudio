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
    x_offset=-10,  # Default to 0 for no horizontal adjustment
    y_offset=-10,  # Default to 0 for no vertical adjustment
):
    """Track and crop a single bounding box in all frames using optical flow."""
    w = median_frame_rgb.shape[1]
    h = median_frame_rgb.shape[0]
    box_w, box_h = int(box_size_ratio[0] * w), int(box_size_ratio[1] * h)

    gray_ref = cv2.cvtColor(median_frame_rgb, cv2.COLOR_BGR2GRAY)
    top_left_pt = np.array([left_anchor_px], dtype=np.float32)
    bottom_right_pt = np.array([right_anchor_px], dtype=np.float32)

    crops = []

    for f in flor.loop("frame", frames):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        tracked_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            gray_ref, gray, np.vstack([top_left_pt, bottom_right_pt]), None
        )

        # Fallback to initial points if tracking fails
        top_left_tracked = tracked_pts[0] if status[0] else top_left_pt[0]
        bottom_right_tracked = tracked_pts[1] if status[1] else bottom_right_pt[0]

        # Stabilize bounding box size
        lx = int(top_left_tracked[0]) + x_offset
        ly = int(top_left_tracked[1]) + y_offset
        rx = lx + box_w
        ry = ly + box_h

        # Debugging: Print coordinates
        print(
            f"Top-Left: ({lx}, {ly}), Bottom-Right: ({rx}, {ry}), Offsets: x={x_offset}, y={y_offset}"
        )

        # Crop the frame
        crop = f[max(ly, 0) : max(ry, 0), max(lx, 0) : max(rx, 0)]
        crops.append(crop)

    return crops


image_paths = load_image_sequence()
selected_frame_index = 0

# We'll pause here for user input
median_frame = cv2.imread(
    os.path.join(input_dir, str(image_paths[selected_frame_index]))
)
median_frame_rgb = cv2.cvtColor(median_frame, cv2.COLOR_BGR2RGB)

# Display the median frame for bounding box entry
plt.figure(figsize=(10, 6))
plt.imshow(median_frame_rgb)
plt.title(
    f"Select the top-left and bottom-right corners of the bounding box: {image_paths[selected_frame_index]}"
)
plt.axis("off")
# plt.show()

# Prompt user to select two points (top-left and bottom-right) on the displayed image
print(
    "Please click on the TOP-LEFT corner, then the BOTTOM-RIGHT corner of the bounding box in the matplotlib window."
)
bbox_points = plt.ginput(2, timeout=0)
if len(bbox_points) != 2:
    raise ValueError("You must select exactly two points.")
plt.close()

# Extract the top-left and bottom-right points
top_left = bbox_points[0]
bottom_right = bbox_points[1]

# Convert to integer pixel coordinates
top_left = (int(top_left[0]), int(top_left[1]))
bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

# Calculate bounding box dimensions
box_width = bottom_right[0] - top_left[0]
box_height = bottom_right[1] - top_left[1]

# Infer bbox size ratio from the bounding box dimensions
box_size_ratio = (
    min(box_width / median_frame.shape[1] + 0.1, 1.0),  # width ratio
    min(box_height / median_frame.shape[0] + 0.1, 1.0),  # height ratio
)

# Call the tracking function with the bounding box points
crops = track_pages_from_reference(
    frames_generator(image_paths),  # all frames
    median_frame_rgb=median_frame_rgb,
    left_anchor_px=top_left,  # Use top-left as the anchor
    right_anchor_px=bottom_right,  # Use bottom-right as the anchor
    box_size_ratio=box_size_ratio,
    x_offset=-20,  # No horizontal offset needed for a single bounding box
    y_offset=10,  # No vertical offset needed for a single bounding box
)

# Save cropped frames to output folder
output_dir = os.path.join("cropped_pages", os.path.basename(input_dir))
os.makedirs(output_dir, exist_ok=True)

for i, crop in enumerate(crops):
    output_path = os.path.join(output_dir, f"{i:04d}.jpg")
    cv2.imwrite(str(output_path), crop)
