import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import platform
import flor


# Load the image
image_path = flor.arg(
    "image_path", os.path.join("test_frames", "WebOfBelief", "0000002940.jpg")
)
image2 = cv2.imread(image_path)

# Load a pre-trained YOLOv5 model from Ultralytics
if platform.system() == "Darwin":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
device = flor.arg("device", device)
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True, device=device)

# Resize the input image for faster inference
resized_image = cv2.resize(image2, (640, int(640 * image2.shape[0] / image2.shape[1])))
results = model(resized_image)

# Parse predictions
results_df = results.pandas().xyxy[0]

# Draw only large rectangular regions (e.g., assume book pages are the largest objects)
# For demo: filter by confidence and size
min_conf = 0.3
min_area = 0.05 * resized_image.shape[0] * resized_image.shape[1]

page_boxes = []
for _, row in results_df.iterrows():
    if row["confidence"] > min_conf:
        x1, y1, x2, y2 = (
            int(row["xmin"]),
            int(row["ymin"]),
            int(row["xmax"]),
            int(row["ymax"]),
        )
        area = (x2 - x1) * (y2 - y1)
        if area > min_area:
            page_boxes.append((x1, y1, x2, y2))

# Visualize detected bounding boxes
output_image = resized_image.copy()
for x1, y1, x2, y2 in page_boxes:
    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Convert BGR to RGB
output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

# Save the result to output.png
plt.figure(figsize=(10, 8))
plt.imshow(output_rgb)
plt.title("YOLOv5 Detected Objects (No Fine-tuning)")
plt.axis("off")
plt.savefig("output.png", bbox_inches="tight", pad_inches=0)
plt.close()
