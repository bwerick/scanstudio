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


def crop_images(image_generator, crop_box):
    """Crop all images using the initial crop box, with the option to adjust."""
    cropped_images = []
    x, y, w, h = crop_box

    for i, (filename, img) in enumerate(image_generator):
        while True:  # Loop until the user confirms or adjusts the crop box
            print(f"Processing {filename}...")
            img_copy = img.copy()
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow(
                "Verify Crop Box (Press 'a' to adjust, 's' to confirm, 'q' to quit)",
                img_copy,
            )
            key = cv2.waitKey(0)

            if key == ord("a"):  # Adjust crop box
                x, y, w, h = select_crop_box(img)
            elif key == ord("s"):  # Confirm crop box
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
