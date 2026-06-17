#!/usr/bin/env python3
"""Find which camera index is your 4K camera.

USB camera indices shift as devices connect/disconnect (Continuity Camera, the
built-in FaceTime cam, external webcams), so the 4K cam isn't always index 0.
This probes a range of indices, requests 4K from each, and reports the
resolution actually delivered — pass the index that reports 3840x2160 to
`make live CAMERA=<n>`.

Usage:
  python scripts/probe_camera.py          # scan indices 0-4
  python scripts/probe_camera.py 2         # probe a single index in detail
"""
import sys
import cv2

# No arg → scan a range and report each index's best mode (one line each).
if len(sys.argv) <= 1:
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    for i in range(5):
        cap = cv2.VideoCapture(i, backend)
        if not cap.isOpened():
            print(f"index {i}: (not present)")
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        ok, f = cap.read()
        got = f"{f.shape[1]}x{f.shape[0]}" if ok and f is not None else "read failed"
        flag = "  <-- 4K, use this" if ok and f is not None and f.shape[1] >= 3840 else ""
        print(f"index {i}: 4K-request -> {got}{flag}")
        cap.release()
    sys.exit(0)

idx = int(sys.argv[1])

# Try the macOS-native backend first; fall back to default.
backends = [("AVFOUNDATION", cv2.CAP_AVFOUNDATION), ("DEFAULT", cv2.CAP_ANY)]
modes = [(3840, 2160), (4096, 2160), (2560, 1440), (1920, 1080), (1280, 720)]

for bname, backend in backends:
    cap = cv2.VideoCapture(idx, backend)
    if not cap.isOpened():
        print(f"[{bname}] cannot open camera {idx}")
        continue
    # Some UVC cams only expose 4K under MJPG; try with and without.
    for fourcc_name in ("default", "MJPG"):
        if fourcc_name == "MJPG":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        print(f"\n[{bname} / fourcc={fourcc_name}]")
        for rw, rh in modes:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(rw))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(rh))
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ok, frame = cap.read()
            got = f"{frame.shape[1]}x{frame.shape[0]}" if ok and frame is not None else "READ FAILED"
            flag = "  <-- honored" if (aw, ah) == (rw, rh) else ""
            print(f"  req {rw}x{rh:<5} -> reports {aw}x{ah:<5} read {got}{flag}")
    cap.release()
