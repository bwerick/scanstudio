import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

import flordb as flor


def load_images_from_folder(folder):
    """Load all images from a folder as a generator."""
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            yield filename, cv2.imread(os.path.join(folder, filename))


def select_crop_box(image):
    """Allow the user to select a cropping box on the image."""
    r = cv2.selectROI("Select Crop Box", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Crop Box")
    return r


# Arrow key codes vary by OS/backend and whether you use waitKey vs waitKeyEx.
# We support the common values across macOS, Linux (X11), and Windows.
_ARROW_KEYS = {
    # macOS (Cocoa)
    63232: (0, -1),  # Up
    63233: (0, 1),  # Down
    63234: (-1, 0),  # Left
    63235: (1, 0),  # Right
    # Linux (X11)
    65361: (-1, 0),  # Left
    65362: (0, -1),  # Up
    65363: (1, 0),  # Right
    65364: (0, 1),  # Down
    # Windows (typically from waitKeyEx)
    2424832: (-1, 0),  # Left
    2490368: (0, -1),  # Up
    2555904: (1, 0),  # Right
    2621440: (0, 1),  # Down
    # Some OpenCV builds return these when masking with & 0xFF
    81: (-1, 0),  # Left
    82: (0, -1),  # Up
    83: (1, 0),  # Right
    84: (0, 1),  # Down
}

_WINDOW = "Arrows: move | +/-: step size | 'a': reset box | 's': confirm | 'q': quit"


def _show_box(img, x, y, w, h, step):
    """Draw the crop box overlay and refresh the window immediately."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        overlay,
        f"step={step}px  pos=({x},{y})",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.imshow(_WINDOW, overlay)


def crop_images(image_generator, crop_box, step=10):
    """Crop all images using the initial crop box, with the option to nudge it."""
    cropped_images = []
    x, y, w, h = crop_box

    for filename, img in image_generator:
        print(f"Processing {filename}...")
        img_h, img_w = img.shape[:2]
        _show_box(img, x, y, w, h, step)

        while True:
            # waitKeyEx is required on many systems (notably macOS) to receive
            # special keys like arrows. Fall back to waitKey if not available.
            wait_fn = getattr(cv2, "waitKeyEx", cv2.waitKey)
            key = wait_fn(0)

            if key in _ARROW_KEYS:
                dx, dy = _ARROW_KEYS[key]
                x = max(0, min(img_w - w, x + dx * step))
                y = max(0, min(img_h - h, y + dy * step))
                _show_box(img, x, y, w, h, step)
            elif key == ord("+") or key == ord("="):
                step = min(step + 5, 100)
                _show_box(img, x, y, w, h, step)
            elif key == ord("-"):
                step = max(step - 5, 1)
                _show_box(img, x, y, w, h, step)
            elif key == ord("a"):  # Full reset via selectROI
                x, y, w, h = select_crop_box(img)
                _show_box(img, x, y, w, h, step)
            elif key == ord("s"):  # Confirm
                break
            elif key == ord("q"):  # Quit
                cv2.destroyAllWindows()
                return cropped_images

        cv2.destroyAllWindows()
        cropped = img[y : y + h, x : x + w]
        cropped_images.append((filename, cropped))

    return cropped_images


def save_cropped_images(cropped_images, output_folder, side="cropped"):
    """Save cropped images to the output folder."""
    os.makedirs(output_folder, exist_ok=True)
    for filename, cropped in cropped_images:
        # Modify filename to include side information
        name, ext = os.path.splitext(filename)
        if side.lower() == "left":
            name += "L"
        elif side.lower() == "right":
            name += "R"
        filename = f"{name}{ext}"
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, cropped)
        print(f"Saved {output_path}")


def main():

    book = flor.arg("book", "greenbook")
    side = flor.arg("side", "cropped")
    # side in ("cropped", "left", "right")
    input_folder = os.path.join("test_frames", book, "keyframes")
    output_folder = os.path.join("test_frames", book, side)
    os.makedirs(output_folder, exist_ok=True)

    image_generator = load_images_from_folder(input_folder)
    first_image = next(image_generator, None)
    del image_generator  # Clean up the generator

    if first_image is None:
        print("No images found in the folder.")
        return

    print("Select the cropping box on the first image.")
    crop_box = select_crop_box(first_image[1])

    cropped_images = crop_images(load_images_from_folder(input_folder), crop_box)
    save_cropped_images(cropped_images, output_folder, side)
    print("Batch cropping completed.")


if __name__ == "__main__":
    main()
