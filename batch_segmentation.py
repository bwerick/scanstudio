import os
import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
import platform
import flor

# --- Paths ---
image_dir = flor.arg("image_dir", default=os.path.join("test_frames", "WebOfBelief"))
output_dir = os.path.join(image_dir, "batch_segmented")
checkpoint_path = os.path.join(os.path.pardir, "sam_vit_h_4b8939.pth")

# --- Model setup ---
torch.set_default_dtype(torch.float32)

sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
if platform.system() == "Darwin":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
device = flor.arg("device", device)
sam = sam.to(device)

# --- Configure the automatic mask generator ---
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=8,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.90,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Filter out tiny masks
)

# --- Create output directory ---
os.makedirs(output_dir, exist_ok=True)

# --- Loop over images ---
for filename in flor.loop("frame", os.listdir(image_dir)):
    if not filename.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(image_dir, filename)
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    # Generate masks
    image = image.astype(np.float32)
    masks = mask_generator.generate(image)

    # Create blank overlay image
    segmented = np.zeros_like(image, dtype=np.uint8)

    # Optionally overlay all masks with unique colors
    for i, mask_dict in enumerate(masks):
        mask = mask_dict["segmentation"]
        color = np.random.randint(0, 255, size=3, dtype=np.uint8)
        segmented[mask] = color

    # Save output
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
