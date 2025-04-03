import fitz
import os
from PIL import Image
import platform

import io

import flor
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch

DOC_DIR = "test_frames"

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
        for frame in flor.loop("frame", os.listdir(frames)):
            img_path = os.path.join(frames, frame)

            doctr_doc = DocumentFile.from_images(img_path)
            result = model(doctr_doc)
            flor.log("text", result.render())

    print("Frame-by-frame OCR Done!")
