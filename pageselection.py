import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import cv2
import os
import flor
import platform
import torch

# Load the images
folder_path = flor.arg("folder_path", os.path.join("test_frames", "WebOfBelief"))
image_list = []

"""Load images from the specified folder and store them in a list."""
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            image_list.append((img_path, img))
        else:
            print(f"Warning: could not read {filename}")

print(f"Loaded {len(image_list)} images.")

# Select the middle image for annotation
index = int(len(image_list) / 2)
img_path, img1 = image_list[index]
img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)


# === GLOBALS for shared state ===
coords = []
labels = []
dots = []
last_click = [None, None]


# === Click to add points ===
def onclick(event):
    if event.xdata is None or event.ydata is None:
        return
    x, y = int(event.xdata), int(event.ydata)
    last_click[0], last_click[1] = x, y

    if event.button == 1:
        coords.append((x, y))
        labels.append(1)
        (dot,) = plt.plot(x, y, "go")  # inclusive
        dots.append(dot)
        print(f"[INCLUSIVE] ({x}, {y})")

    elif event.button == 3:
        coords.append((x, y))
        labels.append(0)
        (dot,) = plt.plot(x, y, "ro")  # exclusive
        dots.append(dot)
        print(f"[EXCLUSIVE] ({x}, {y})")

    plt.draw()


# === Press 'd' to delete nearest point ===
def onkey(event):
    if event.key == "d" and coords:
        x, y = last_click
        if x is None or y is None:
            print("Click near a point first to select for deletion.")
            return
        distances = [np.hypot(x - px, y - py) for (px, py) in coords]
        index = np.argmin(distances)
        removed = coords.pop(index)
        removed_label = labels.pop(index)
        dot = dots.pop(index)
        dot.remove()
        print(f"Removed ({removed}) with label {removed_label}")
        plt.draw()


# === Run selection session ===
def run_annotation(title, imgIn):
    global coords, labels, dots, last_click
    coords = []
    labels = []
    dots = []
    last_click = [None, None]

    img = imgIn
    fig, ax = plt.subplots()
    ax.imshow(img)

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)

    print(f"\n📝 {title}")
    print(" - Left-click: INCLUSIVE (green)")
    print(" - Right-click: EXCLUSIVE (red)")
    print(" - Press 'd': Delete nearest clicked point")
    print(" - Close the window to finish selection\n")

    plt.title(title)
    plt.show()

    return np.array(coords), np.array(labels)


# === Run two annotation rounds ===
coords_left, labels_left = run_annotation("Step 1: Select LEFT PAGE Points", img)
coords_right, labels_right = run_annotation("Step 2: Select RIGHT PAGE Points", img)

# === Combine or keep separate ===
# print("\n=== LEFT PAGE ===")
# print("Coordinates:\n", coords_left)
# print("Labels:\n", labels_left)

# print("\n=== RIGHT PAGE ===")
# print("Coordinates:\n", coords_right)
# print("Labels:\n", labels_right)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


sam = sam_model_registry["vit_h"](
    checkpoint=flor.arg(
        "sam_checkpoint", os.path.join(os.path.pardir, "sam_vit_h_4b8939.pth")
    )
)
if platform.system() == "Darwin":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
device = flor.arg("device", device)
sam = sam.to(device)
predictor = SamPredictor(sam)


# === Output folder ===
output_folder = os.path.join(
    flor.arg("output_folder", "segmented_pages"), os.path.basename(folder_path)
)
os.makedirs(output_folder, exist_ok=True)


# === Segment the pages ===
def segment_page(image, point_coords, point_labels):
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )
    mask = masks[0].astype(np.uint8)

    if len(image.shape) == 3:
        mask_rgb = np.stack([mask] * 3, axis=-1)
    else:
        mask_rgb = np.expand_dims(mask, axis=-1)

    segmented = np.zeros_like(image)
    segmented[mask_rgb.astype(bool)] = image[mask_rgb.astype(bool)]
    return segmented


# === Main loop: process each image ===
for i, (img_path, image) in flor.loop("frame", enumerate(image_list)):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    frame_id = base_name

    if os.path.exists(
        os.path.join(output_folder, f"{frame_id}_left.png")
    ) and os.path.exists(os.path.join(output_folder, f"{frame_id}_right.png")):
        print(f"Skipping {frame_id} (already processed)")
        continue

    print(f"Processing frame: {frame_id}")

    # Segment left page
    left_segment = segment_page(image, coords_left, labels_left)
    cv2.imwrite(os.path.join(output_folder, f"{frame_id}_left.png"), left_segment)

    # Segment right page
    right_segment = segment_page(image, coords_right, labels_right)
    cv2.imwrite(os.path.join(output_folder, f"{frame_id}_right.png"), right_segment)
