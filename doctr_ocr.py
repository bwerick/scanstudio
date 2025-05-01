import os
from PIL import Image
import platform

import io

import flor
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch

# df = flor.dataframe("out_path")
# df = flor.utils.latest(df[df["filename"] == "frameextraction.py"])
# DOC_DIR = str(df["out_path"].values[0])

frame = flor.arg("frame", default="test_frames/video_1/00001.jpg")
out_path = flor.arg("out_path", default="test_frames/video_1/00001.txt")
assert (
    os.path.splitext(frame)[0] == os.path.splitext(out_path)[0]
), "Frame and out_path must match except for the extension"


# # Determine the device based on the operating system
if platform.system() == "Darwin":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
device = flor.arg("device", device)

model = ocr_predictor(
    det_arch=flor.arg("det_arch", "linknet_resnet50"),
    reco_arch=flor.arg("reco_arch", "master"),
    pretrained=flor.arg("pretrained", True),
).to(device)


video_stem = os.path.basename(os.path.dirname(os.path.splitext(frame)[0]))
frame_jpg = os.path.basename(frame)

with flor.iteration("video", value=video_stem):
    with flor.iteration(
        "frame", idx=int(os.path.splitext(frame_jpg)[0]), value=frame_jpg
    ):
        # Convert the image to a DocumentFile
        doctr_doc = DocumentFile.from_images(frame)

        # Perform OCR
        result = model(doctr_doc)

        # Save the result to a text file
        with open(out_path, "w") as f:
            f.write(result.render())

        flor.log("status", "success")
