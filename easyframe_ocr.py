import easyocr
import os
from PIL import Image
import platform

import io

import flor
import numpy as np
import torch


df = flor.dataframe("out_path")
df = flor.utils.latest(df[df["filename"] == "frameextraction.py"])
DOC_DIR = str(df["out_path"].values[0])

# Determine the device based on the operating system
if platform.system() == "Darwin":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    model = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

    for v in flor.loop("video", os.listdir(DOC_DIR)):
        frames = os.path.join(DOC_DIR, v)
        # skip hidden files
        if frames.startswith(".") or not os.path.isdir(frames):
            continue
        for frame in flor.loop("frame", os.listdir(frames)):
            img_path = os.path.join(frames, frame)

            easy_doc = model.readtext(img_path)
            
            result = ""
            for (bbox, text, prob) in result:
                result = text
            flor.log("text", result)

    print("Frame-by-frame OCR Done!")








#reader = easyocr.Reader(['ru']) # this needs to run only once to load the model into memory
#result = model.readtext('/Users/erickduarte/git/segmentation/test_frames/Spin Dictators.mov/0000001395.jpg')

""" for (bbox, text, prob) in result:
    print(f'Text: {text}, Probability: {prob}') """