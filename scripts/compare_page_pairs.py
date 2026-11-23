#!/usr/bin/env python3
"""Compare labeled page-image pairs with a baseline and an experimental model.

This is a research tool, not an automatic deletion stage. The ONNX model must
be supplied separately. Its 1,000 ImageNet classification logits are used as
features; this is not a document-trained embedding or a calibrated detector.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from frame_quality import load_preview, compare


def model_input(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    image = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return np.transpose(image, (2, 0, 1))[None, :, :, :]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pairs_csv', help='CSV with first, second, label columns')
    parser.add_argument('--model', required=True, help='Path to ResNet-18 v1-7 ONNX')
    parser.add_argument('--output', required=True, help='JSON output path')
    args = parser.parse_args()
    csv_path = Path(args.pairs_csv).resolve()
    session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    cache = {}

    def features(path):
        if path not in cache:
            vector = session.run(None, {input_name: model_input(path)})[0].reshape(-1).astype(np.float64)
            norm = np.linalg.norm(vector)
            if norm == 0:
                raise ValueError(f'Zero feature vector: {path}')
            cache[path] = vector / norm
        return cache[path]

    records = []
    with csv_path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            first = (csv_path.parent / row['first']).resolve()
            second = (csv_path.parent / row['second']).resolve()
            if row['label'] not in ('same', 'different'):
                raise ValueError(f"Unexpected label: {row['label']}")
            a, b = load_preview(first), load_preview(second)
            if a is None or b is None:
                raise ValueError('Cannot load a baseline preview')
            scores = compare(a, b)
            records.append({'first': row['first'], 'second': row['second'], 'label': row['label'],
                            'hash_distance': scores['hash_distance'],
                            'pixel_difference': scores['pixel_difference'],
                            'baseline_flag': scores['hash_distance'] <= 4 and scores['pixel_difference'] <= 0.02,
                            'resnet_logit_cosine': float(np.dot(features(first), features(second)))})

    result = {'scope': 'Exploratory pair comparison; not end-to-end frame-level accuracy.',
              'model': 'ONNX Model Zoo ResNet-18 v1-7, ImageNet classification logits',
              'baseline': '96x96 grayscale + blur, dHash <= 4 and pixel difference <= 0.02',
              'records': records}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
