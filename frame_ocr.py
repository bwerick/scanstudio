import os
from PIL import Image
import platform

import io

import flor
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch

df = flor.dataframe("out_path")
df = flor.utils.latest(df[df["filename"] == "frameextraction.py"])
DOC_DIR = df.values[0]

# Determine the device based on the operating system
if platform.system() == "Darwin":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    model = ocr_predictor(
        det_arch="linknet_resnet50", reco_arch="master", pretrained=True
    ).to(device)

    for v in flor.loop("video", os.listdir(DOC_DIR)):
        frames = os.path.join(DOC_DIR, v)
        # skip hidden files
        if frames.startswith(".") or not os.path.isdir(frames):
            continue
        for frame in flor.loop("frame", os.listdir(frames)):
            img_path = os.path.join(frames, frame)

            doctr_doc = DocumentFile.from_images(img_path)
            result = model(doctr_doc)
            flor.log("text", result.render())

    print("Frame-by-frame OCR Done!")
