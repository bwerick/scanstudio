import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
import os
import flor

test_frames = [
    each
    for each in os.listdir("test_frames")
    if os.path.isdir(os.path.join("test_frames", each))
    and os.listdir(os.path.join("test_frames", each))
]
if not test_frames:
    raise ValueError(
        "No test frames found in 'test_frames' directory. Did you run precursor steps i.e. w/ Make? Please add some directories with images."
    )

first_test_frame = test_frames[0]

# Constants
SOURCE_DIR = flor.arg(
    "input_dir", default=os.path.join("test_frames", first_test_frame)
)
CROPPED_DIR = os.path.join(SOURCE_DIR, "cropped")

# Initialize session state for frame index and crop position
if "frame_index" not in st.session_state:
    st.session_state.frame_index = 0
if "crop_position" not in st.session_state:
    st.session_state.crop_position = {"left": 0, "top": 0}

# UI
st.set_page_config(page_title="Batch Image Cropper", layout="centered")
st.title("🖼️ Server-Side Batch Cropper - Dataflow Step")

# Ensure directories exist
if not os.path.exists(SOURCE_DIR):
    st.error(f"Directory `{SOURCE_DIR}` does not exist.")
else:
    os.makedirs(CROPPED_DIR, exist_ok=True)

    image_files = sorted(
        [
            f
            for f in os.listdir(SOURCE_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and os.path.isfile(os.path.join(SOURCE_DIR, f))
        ]
    )

    if not image_files:
        st.warning("No image files found.")
    else:
        # Get the current frame
        current_frame = st.session_state.frame_index
        sample_path = os.path.join(SOURCE_DIR, image_files[current_frame])
        sample_image = Image.open(sample_path)

        st.markdown(f"### Frame {current_frame + 1} of {len(image_files)}")
        st.image(
            sample_image,
            caption=f"Current Frame: {image_files[current_frame]}",
            use_container_width=True,
        )

        # Adjust crop position
        left = st.number_input(
            "Left", value=st.session_state.crop_position["left"], step=10
        )
        top = st.number_input(
            "Top", value=st.session_state.crop_position["top"], step=10
        )
        width = 300  # Fixed width
        height = 300  # Fixed height

        # Update crop position in session state
        st.session_state.crop_position = {"left": left, "top": top}

        # Preview the crop
        preview_crop = sample_image.crop((left, top, left + width, top + height))
        st.image(preview_crop, caption="Preview Crop", use_container_width=True)

        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Previous", disabled=current_frame == 0):
                st.session_state.frame_index -= 1
        with col2:
            if st.button("✅ Accept Crop"):
                # Save the cropped image
                out_path = os.path.join(CROPPED_DIR, image_files[current_frame])
                preview_crop.save(out_path)
                st.success(f"Cropped image saved to `{out_path}`.")
        with col3:
            if st.button("➡️ Next", disabled=current_frame == len(image_files) - 1):
                st.session_state.frame_index += 1
