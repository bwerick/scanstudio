import os
from PIL import Image
import platform

import io

import flor
import numpy as np

import easyocr

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch

frame = flor.arg("frame", default="test_frames/video_1/00001.jpg")
out_path = flor.arg("out_path", default="test_frames/video_1/00001_doctr.txt")
ocr_mode = flor.arg("ocr_mode", default="doctr")
assert ocr_mode in ["easyocr", "doctr"], "ocr_mode must be either easyocr or doctr"

# # Determine the device based on the operating system
if platform.system() == "Darwin":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
device = flor.arg("device", device)

if ocr_mode == "easyocr":
    # Initialize the EasyOCR model
    model = easyocr.Reader(["en"], gpu=device == "cuda")

    # Read the image
    result = model.readtext(frame)

    # Extract text from the result
    with open(out_path, "w") as f:
        for bbox, text, prob in result:
            f.write(text + "\n")

elif ocr_mode == "doctr":

    model = ocr_predictor(
        det_arch=flor.arg("det_arch", "linknet_resnet50"),
        reco_arch=flor.arg("reco_arch", "master"),
        pretrained=flor.arg("pretrained", True),
    ).to(device)

    # Convert the image to a DocumentFile
    doctr_doc = DocumentFile.from_images(frame)

    # Perform OCR
    result = model(doctr_doc)

    # Save the result to a text file
    with open(out_path, "w") as f:
        f.write(result.render())

flor.log("status", "success")
