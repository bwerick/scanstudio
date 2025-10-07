import os
import streamlit as st
import flordb as flor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_images(directory: str) -> list[str]:
    try:
        return sorted(
            filename
            for filename in os.listdir(directory)
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
        )
    except FileNotFoundError:
        st.error(f"Directory not found: {directory}")
        return []


def delete_images(directory: str, keep_filenames: list[str]) -> int:
    keep_set = set(keep_filenames)
    removed = 0

    for filename in os.listdir(directory):
        if filename not in keep_set:
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                removed += 1

    return removed


def main() -> None:
    output_dir = flor.arg("output_dir", "test_frames")

    st.title("Keyframe Selector App")

    for doc in flor.loop("document", os.listdir(output_dir)):
        image_dir = os.path.join(output_dir, doc, "keyframes")

        if not image_dir:
            st.info("Enter a directory containing keyframe images.")
            return

        images = load_images(image_dir)

        if not images:
            st.warning("No images found in the specified directory.")
            return

        st.write("Select the keyframes you want to keep:")
        keep_images: list[str] = []

        for filename in images:
            filepath = os.path.join(image_dir, filename)
            st.image(filepath, caption=filename, use_container_width=True)
            if st.checkbox(f"Keep {filename}", value=True, key=filename):
                keep_images.append(filename)

        if st.button("Delete unselected images"):
            deleted_count = delete_images(image_dir, keep_images)
            st.success(
                f"Kept {len(keep_images)} image(s). Deleted {deleted_count} image(s)."
            )


if __name__ == "__main__":
    main()
