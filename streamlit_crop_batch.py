import os
from PIL import Image
import streamlit as st
from streamlit_cropper import st_cropper
import flordb as flor  # optional CLI override

# ──────────────────────────────────────────────────────────────────────
# 1. Discover a subfolder in test_frames that contains images
# ──────────────────────────────────────────────────────────────────────
test_frames = [
    d
    for d in os.listdir("test_frames")
    if os.path.isdir(os.path.join("test_frames", d))
    and any(
        f.lower().endswith((".png", ".jpg", ".jpeg"))
        for f in os.listdir(os.path.join("test_frames", d))
    )
]
if not test_frames:
    raise ValueError(
        "No test frames found under 'test_frames/'. "
        "Add a sub-directory with images first."
    )

first_test_frame = test_frames[0]

# Folder paths (can be overridden with `--input_dir=...` via Flor)
SOURCE_DIR = flor.arg(
    "input_dir", default=os.path.join("test_frames", first_test_frame)
)
CROPPED_DIR = os.path.join(SOURCE_DIR, "cropped")

# ──────────────────────────────────────────────────────────────────────
# 2. Streamlit app configuration
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Batch Image Cropper", layout="centered")
st.title("🖼️ Batch Image Cropper – Dynamic ROI")

# Ensure destination folder exists
os.makedirs(CROPPED_DIR, exist_ok=True)

# Build list of images
image_files = sorted(
    f for f in os.listdir(SOURCE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))
)
if not image_files:
    st.error(f"No images found in '{SOURCE_DIR}'.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────
# 3. Persistent session state
# ──────────────────────────────────────────────────────────────────────
if "frame_index" not in st.session_state:
    st.session_state.frame_index = 0

# Handy aliases
idx = st.session_state.frame_index
name = image_files[idx]

# ──────────────────────────────────────────────────────────────────────
# 4. Load & display cropper
# ──────────────────────────────────────────────────────────────────────
img_path = os.path.join(SOURCE_DIR, name)
image = Image.open(img_path)

st.markdown(f"### Frame {idx + 1} / {len(image_files)} — `{name}`")
st.markdown("Drag to select the region you wish to keep, then click **Accept Crop**.")

cropped_img = st_cropper(
    image,
    box_color="#00FF00",
    aspect_ratio=None,  # free aspect ratio
    return_type="image",  # returns PIL.Image
    realtime_update=True,
    key=name,  # isolate widget state per image
)

st.image(cropped_img, caption="Preview Crop", use_container_width=True)


# ──────────────────────────────────────────────────────────────────────
# 5. Navigation & save handlers
# ──────────────────────────────────────────────────────────────────────
def save_crop():
    out_path = os.path.join(CROPPED_DIR, name)
    cropped_img.save(out_path)
    st.success(f"Cropped image saved → `{out_path}`")


def prev_frame():
    if st.session_state.frame_index > 0:
        st.session_state.frame_index -= 1
        st.rerun()  # ← new API


def next_frame():
    if st.session_state.frame_index < len(image_files) - 1:
        st.session_state.frame_index += 1
        st.rerun()  # ← new API


# Button layout
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.button("⬅️ Previous", on_click=prev_frame, disabled=idx == 0)
with c2:
    st.button("✅ Accept Crop", on_click=save_crop)
with c3:
    st.button("➡️ Next", on_click=next_frame, disabled=idx == len(image_files) - 1)
