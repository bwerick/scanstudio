# Exploratory page-pair comparison

`scripts/compare_page_pairs.py` compares labeled image pairs using the existing Phase 3.5 preview/thresholds and cosine similarity between the classification logits of a separately supplied ResNet-18 v1-7 ONNX model. This is an experimental comparison, not a calibrated duplicate detector. It never deletes pages. The model was trained for ImageNet classification, not document deduplication.

Prepare a CSV with `first,second,label` columns. Image paths are relative to the CSV; labels must be `same` or `different`. Supply the [ONNX Model Zoo ResNet-18 v1-7 model](https://github.com/onnx/models/tree/main/validated/vision/classification/resnet) yourself; model weights and scan images are not included in the repository. Install the repository requirements, including OpenCV, NumPy, and ONNX Runtime, then run:

```sh
.venv/bin/python scripts/compare_page_pairs.py pairs.csv --model /path/to/resnet18-v1-7.onnx --output pair_results.json
```

The JSON reports the two baseline scores and decision, and the ResNet cosine score for each pair. The baseline defaults compare 96-by-96 grayscale, blurred previews using hash distance <= 4 and normalized pixel difference <= 0.02. The learned model resizes color images to 224 by 224 pixels and uses ImageNet normalization. Cosine scores are continuous; no decision threshold is selected by this script.

An exploratory November 23 comparison used four selected page pairs from one existing PDF: two repeated pages and two distinct pages. The baseline flagged neither repeated pair; the model assigned a different-page pair a higher score than either repeated pair. This small, selected sample is not a statistically valid before-and-after evaluation, is not a test of the original frame-level pipeline, and cannot establish which approach performs better. Preserve original full frames, label them independently, and test multiple recordings before claiming precision, recall, or an improvement.
