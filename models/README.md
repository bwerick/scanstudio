# Models

## corner_net-rm333-rig.onnx

Corner + gutter regression for Phase-4 crop tracking (see
`scripts/corner_net.py`). Maps a raw double-spread frame to the four corners
of the operator-convention *page-block* crop box plus the spine position.
MobileNetV3-Small head, 448×256 input, ~10 MB, CPU inference via onnxruntime
(already a pipeline dependency) — no torch needed at runtime.

**Rig-specific.** Trained on the `rm333-rig` setup — its camera geometry,
table, lighting, and the operator's own page-block convention — from
operator-validated Phase-4 geometry across this repo's scanned books.
Provenance (rig, date, frame/book counts) is embedded in the file's ONNX
metadata:

```bash
python -c "import onnx; print(*onnx.load('models/corner_net-rm333-rig.onnx').metadata_props, sep='')"
```

On any other rig, expect degraded accuracy: delete this file (the pipeline
falls back to pure mask-based tracking, byte-for-byte the pre-model
behavior) and retrain for your own rig once you have a few reviewed books
with recordings still on disk:

```bash
.venv/bin/python scripts/train_corner_net.py extract            # build dataset
.venv/bin/python scripts/train_corner_net.py lobo               # honest eval
.venv/bin/python scripts/train_corner_net.py final --rig my-rig # export
```

Training needs torch/torchvision (`make install-legacy` territory); the
loader picks up the first `models/corner_net*.onnx`, so keep exactly one.
