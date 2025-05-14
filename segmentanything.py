import os
import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import torch
import platform
import flor

# --- Paths ---
image_dir = flor.arg("image_dir", default=os.path.join("test_frames", "WebOfBelief"))
output_dir = os.path.join(image_dir, "segmented")
checkpoint_path = os.path.join(os.path.pardir, "sam_vit_h_4b8939.pth")

sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
predictor = SamPredictor(sam)

# Optional: move to GPU if available
if platform.system() == "Darwin":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(flor.arg("device", device))


# --- Define your fixed prompts ---
def get_prompt(image):
    y, x = image.shape[:2]
    x_center = x // 2
    y_center = y // 2
    input_point = np.array(
        [
            [x_center * 0.7, y_center * 0.5],
            [x_center * 0.5, y_center * 1.5],
            [x_center * 1.3, y_center * 0.5],
            [x_center * 1.5, y_center * 1.5],
        ]
    )
    input_label = np.array([1, 1, 0, 0])
    return input_point, input_label


# --- Processing loop ---
os.makedirs(output_dir, exist_ok=True)

for filename in flor.loop("frame", os.listdir(image_dir)):
    if filename.lower().endswith(".jpg"):
        img_path = os.path.join(image_dir, filename)
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # image = image.astype(np.float32)

        predictor.set_image(image)
        input_point, input_label = get_prompt(image)

        masks, scores, logits = predictor.predict(
            point_coords=input_point, point_labels=input_label, multimask_output=False
        )

        mask = masks[0]
        h, w = mask.shape
        mask_3ch = np.stack([mask] * 3, axis=-1)

        segmented = np.zeros_like(image)
        segmented[mask_3ch] = image[mask_3ch]

        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
