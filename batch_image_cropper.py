import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


def load_images_from_folder(folder):
    """Load all images from a folder."""
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append((filename, img))
    return images


def select_crop_box(image):
    """Allow the user to select a cropping box on the image."""
    r = cv2.selectROI("Select Crop Box", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Crop Box")
    return r


def crop_images(images, crop_box):
    """Crop all images using the initial crop box, with the option to adjust."""
    cropped_images = []
    for i, (filename, img) in enumerate(images):
        print(f"Processing {filename}...")
        if i == 0:
            x, y, w, h = crop_box
        else:
            cv2.imshow("Current Image", img)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            key = cv2.waitKey(0)
            if key == ord("a"):  # Adjust crop box
                x, y, w, h = select_crop_box(img)
            elif key == ord("s"):  # Skip adjustment
                pass
            elif key == ord("q"):  # Quit
                break
            cv2.destroyAllWindows()
        cropped = img[y : y + h, x : x + w]
        cropped_images.append((filename, cropped))
    return cropped_images


def save_cropped_images(cropped_images, output_folder):
    """Save cropped images to the output folder."""
    os.makedirs(output_folder, exist_ok=True)
    for filename, cropped in cropped_images:
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, cropped)
        print(f"Saved {output_path}")


def main():
    input_folder = input("Enter the input folder path: ")
    output_folder = input("Enter the output folder path: ")

    images = load_images_from_folder(input_folder)
    if not images:
        print("No images found in the folder.")
        return

    print("Select the cropping box on the first image.")
    crop_box = select_crop_box(images[0][1])

    cropped_images = crop_images(images, crop_box)
    save_cropped_images(cropped_images, output_folder)
    print("Batch cropping completed.")


if __name__ == "__main__":
    main()
