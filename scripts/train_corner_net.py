"""Train the Phase-4 corner + gutter regression model (models/corner_net.onnx).

Not a pipeline phase and not imported by one: this is the offline tool that
turns accumulated operator geometry into the model ``corner_net.py`` serves.
It needs torch/torchvision (several GB — ``make install-legacy`` territory);
the pipeline itself only ever needs onnxruntime.

The training set is every keyframe that carries an operator ``crop_quad``
across every book that still has its recording: frames are re-extracted from
``recordings/<book>.mp4`` because Phase 5 crops ``images/`` in place. Frames
whose geometry the operator explicitly *drew* (review_log split events) are
the cleanest labels and get 3x sampling weight over merely-validated ones.

Subcommands:
  extract   re-extract labeled raw frames into the dataset directory
  lobo      leave-one-book-out folds — the honest generalization number
  final     train on all books, export models/corner_net.onnx

Typical retrain after scanning more books:
  .venv/bin/python scripts/train_corner_net.py extract
  .venv/bin/python scripts/train_corner_net.py lobo    # sanity-check errors
  .venv/bin/python scripts/train_corner_net.py final
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "output" / "_corner_dataset"
MODEL_OUT = REPO / "models" / "corner_net.onnx"

WORK_W = 1280                 # extracted frame width
CACHE_W, CACHE_H = 640, 360   # in-memory training cache
IN_W, IN_H = 448, 256         # model input (must match corner_net.py)
EPOCHS = 36
BATCH = 32
LR = 3e-4
EDIT_WEIGHT = 3.0
SEED = 0
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ── extract ──────────────────────────────────────────────────────────────


def _edited_frames(book):
    p = REPO / "output" / book / "json" / "review_log.json"
    if not p.exists():
        return set()
    log = json.loads(p.read_text())
    return {int(e["frame"])
            for s in log.get("sessions", [])
            for e in s.get("events", [])
            if e.get("type") == "split" and e.get("frame") is not None}


def extract():
    (DATASET / "frames").mkdir(parents=True, exist_ok=True)
    labels_path = DATASET / "labels.jsonl"
    done = set()
    if labels_path.exists():
        for line in labels_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["book"], r["frame_index"]))
    books = sorted(
        d.name for d in (REPO / "output").iterdir()
        if (d / "json" / "keyframes.json").exists()
        and (REPO / "recordings" / f"{d.name}.mp4").exists()
    )
    with open(labels_path, "a") as lf:
        for book in books:
            kfs = json.loads(
                (REPO / "output" / book / "json" / "keyframes.json").read_text())
            edits = _edited_frames(book)
            todo = sorted(
                (k for k in kfs
                 if (k.get("crop_quad") or k.get("gutter") is not None)
                 and (book, k["frame_index"]) not in done),
                key=lambda k: k["frame_index"])
            print(f"[{book}] {len(todo)} frames to extract", flush=True)
            if not todo:
                continue
            cap = cv2.VideoCapture(str(REPO / "recordings" / f"{book}.mp4"))
            n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for k in todo:
                fi = k["frame_index"]
                if fi >= n_total:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, img = cap.read()
                if not ok or img is None:
                    print(f"  [warn] {book} frame {fi} unreadable", flush=True)
                    continue
                h, w = img.shape[:2]
                small = cv2.resize(
                    img, (WORK_W, int(round(h * WORK_W / w))),
                    interpolation=cv2.INTER_AREA)
                name = f"{book}_{fi:06d}.jpg"
                cv2.imwrite(str(DATASET / "frames" / name), small,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                lf.write(json.dumps({
                    "book": book, "frame_index": fi, "file": name,
                    "src_size": [w, h], "quad": k.get("crop_quad"),
                    "gutter": k.get("gutter"),
                    "validated": bool(k.get("validated")),
                    "edited": fi in edits,
                }) + "\n")
                lf.flush()
            cap.release()
    print("extract done", flush=True)


# ── training (torch imported lazily so `extract` runs without it) ────────


def load_records():
    recs = [json.loads(line)
            for line in (DATASET / "labels.jsonl").read_text().splitlines()]
    return [r for r in recs if r.get("quad")]


def make_dataset_cls():
    import torch
    from torch.utils.data import Dataset

    class Frames(Dataset):
        def __init__(self, recs, train):
            self.recs, self.train, self.cache = recs, train, {}

        def _img(self, rec):
            im = self.cache.get(rec["file"])
            if im is None:
                im = cv2.imread(str(DATASET / "frames" / rec["file"]))
                im = cv2.resize(im, (CACHE_W, CACHE_H),
                                interpolation=cv2.INTER_AREA)
                self.cache[rec["file"]] = im
            return im

        def __len__(self):
            return len(self.recs)

        def __getitem__(self, i):
            rec = self.recs[i]
            img = self._img(rec).copy()
            pts = np.array(rec["quad"], np.float32)
            g = rec.get("gutter")
            g = -1.0 if g is None else float(g)
            if self.train:
                rng = np.random
                ang, sc = rng.uniform(-2.5, 2.5), rng.uniform(0.92, 1.08)
                M = cv2.getRotationMatrix2D((CACHE_W / 2, CACHE_H / 2), ang, sc)
                M[0, 2] += rng.uniform(-0.04, 0.04) * CACHE_W
                M[1, 2] += rng.uniform(-0.04, 0.04) * CACHE_H
                img = cv2.warpAffine(img, M, (CACHE_W, CACHE_H),
                                     borderMode=cv2.BORDER_REPLICATE)
                px = pts * [CACHE_W, CACHE_H]
                px = (M[:, :2] @ px.T).T + M[:, 2]
                pts = (px / [CACHE_W, CACHE_H]).astype(np.float32)
                if rng.rand() < 0.5:               # flip: tl<->tr, bl<->br
                    img = img[:, ::-1]
                    pts[:, 0] = 1.0 - pts[:, 0]
                    pts = pts[[1, 0, 3, 2]]
                    if g >= 0:
                        g = 1.0 - g
                img = img.astype(np.float32)
                img *= rng.uniform(0.75, 1.25)
                img = (img - 128) * rng.uniform(0.85, 1.15) + 128
                if rng.rand() < 0.3:
                    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
                img = np.clip(img, 0, 255)
            img = cv2.resize(img.astype(np.uint8), (IN_W, IN_H),
                             interpolation=cv2.INTER_AREA)
            x = (img[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
            x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
            y = torch.from_numpy(
                np.concatenate([pts.reshape(-1), [g]]).astype(np.float32))
            return x, y

    return Frames


def make_model():
    import torch.nn as nn
    import torchvision

    m = torchvision.models.mobilenet_v3_small(
        weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 9)
    return m


def train_one(train_recs, dev, log_prefix=""):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    Frames = make_dataset_cls()
    weights = [EDIT_WEIGHT if r.get("edited") else 1.0 for r in train_recs]
    dl = DataLoader(
        Frames(train_recs, train=True), batch_size=BATCH,
        sampler=WeightedRandomSampler(weights, num_samples=len(train_recs),
                                      replacement=True))
    model = make_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for ep in range(EPOCHS):
        model.train()
        tot, nb, t0 = 0.0, 0, time.time()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = model(x)
            corners = nn.functional.smooth_l1_loss(
                out[:, :8] * 10, y[:, :8] * 10, reduction="none").mean(dim=1)
            has_g = (y[:, 8] >= 0).float()
            gut = nn.functional.smooth_l1_loss(
                out[:, 8] * 10, y[:, 8].clamp(min=0) * 10,
                reduction="none") * has_g
            loss = (corners + 0.5 * gut).mean()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        if ep % 6 == 5 or ep == EPOCHS - 1:
            print(f"{log_prefix}ep {ep+1:2d}/{EPOCHS} loss {tot/nb:.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    return model


def predict_all(model, recs, dev):
    import torch
    from torch.utils.data import DataLoader

    Frames = make_dataset_cls()
    model.eval()
    outs = []
    with torch.no_grad():
        for x, _ in DataLoader(Frames(recs, train=False), batch_size=BATCH):
            outs.append(model(x.to(dev)).cpu().numpy())
    return np.concatenate(outs)


def eval_book(model, recs, dev):
    """Held-out errors: absolute, anchor-relative, gutter (% frame width)."""
    preds = predict_all(model, recs, dev)
    gts = np.array([r["quad"] for r in recs], np.float32)
    pq = preds[:, :8].reshape(-1, 4, 2)
    ar = np.array([1.0, recs[0]["src_size"][1] / recs[0]["src_size"][0]])
    err_abs = 100 * np.linalg.norm((pq - gts) * ar, axis=2).max(axis=1)
    bias = pq[0] - gts[0]          # one anchor frame calibrates the session
    err_rel = 100 * np.linalg.norm(
        (pq - bias - gts) * ar, axis=2).max(axis=1)[1:]
    g_lab = np.array([r["gutter"] if r.get("gutter") is not None else np.nan
                      for r in recs], np.float32)
    gv = ~np.isnan(g_lab)
    err_g = 100 * np.abs(preds[gv, 8] - g_lab[gv])
    return {"abs": err_abs, "rel": err_rel, "gut": err_g}


def stats(a):
    if len(a) == 0:
        return "n/a"
    return (f"med {np.median(a):5.2f}  p90 {np.percentile(a, 90):5.2f}  "
            f"max {a.max():6.2f}")


def device():
    import torch

    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def lobo():
    dev = device()
    recs = load_records()
    books = sorted({r["book"] for r in recs})
    print(f"{len(recs)} records across {len(books)} books; device {dev}",
          flush=True)
    agg = {"abs": [], "rel": [], "gut": []}
    for book in books:
        tr = [r for r in recs if r["book"] != book]
        te = sorted((r for r in recs if r["book"] == book),
                    key=lambda r: r["frame_index"])
        model = train_one(tr, dev, log_prefix=f"[{book}] ")
        e = eval_book(model, te, dev)
        for k in agg:
            agg[k].append(e[k])
        print(f"[{book}] n={len(te)}  ABS  {stats(e['abs'])}", flush=True)
        print(f"[{book}]        ANCH {stats(e['rel'])}", flush=True)
        print(f"[{book}]        GUT  {stats(e['gut'])}", flush=True)
    print("\n===== LOBO aggregate (held-out frames pooled) =====")
    for k, name in (("abs", "ABSOLUTE  "), ("rel", "ANCHOR-REL"),
                    ("gut", "GUTTER    ")):
        print(f"{name}: {stats(np.concatenate(agg[k]))}")


def final():
    import torch

    dev = device()
    recs = load_records()
    model = train_one(recs, dev, log_prefix="[final] ")
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    torch.onnx.export(model, torch.zeros(1, 3, IN_H, IN_W), str(MODEL_OUT),
                      input_names=["image"], output_names=["out"],
                      dynamo=False)
    print(f"saved {MODEL_OUT}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["extract", "lobo", "final"])
    cmd = ap.parse_args().cmd
    {"extract": extract, "lobo": lobo, "final": final}[cmd]()


if __name__ == "__main__":
    main()
