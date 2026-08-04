"""
Tests for the browser-capture session logic (p0_web_capture.WebSession).

The session is deliberately socket-free: it takes messages via handle_text /
handle_binary and emits dicts through a send callable. These tests drive it
exactly the way the WebSocket handler does — synthetic analysis frames in,
JSON out — and check the artifacts written at finish match the shapes the
native front end (p0_live_capture) produces, since downstream phases read
them interchangeably.

Run standalone (`python tests/test_web_session.py`) or under pytest.
"""

import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import p0_web_capture as web  # noqa: E402
from utils import ProjectPaths  # noqa: E402

H, W = 90, 160
FPS = 30.0
US_PER_FRAME = 1e6 / FPS


def still_frame(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 200, size=(H, W), dtype=np.int16).astype(np.uint8)


def frame_msg(ts_us, gray):
    return (struct.pack("<BdHH", web.MSG_FRAME, ts_us,
                        gray.shape[1], gray.shape[0]) + gray.tobytes())


def jpeg_msg(frame_index, payload=b"\xff\xd8 fake jpeg"):
    return struct.pack("<BI", web.MSG_JPEG, frame_index) + payload


def chunk_msg(seq, payload):
    return struct.pack("<BI", web.MSG_CHUNK, seq) + payload


class Session:
    """A WebSession in a temp project, with its outbound messages collected."""

    def __init__(self, **overrides):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        argv = [str(root / "out"), str(root / "rec.mp4"),
                "--settle-time", "0.2", "--smoothing-window", "5"]
        for k, v in overrides.items():
            argv += [f"--{k.replace('_', '-')}", str(v)]
        self.cfg = web.parse_args(argv)
        self.paths = ProjectPaths(self.cfg.output_dir)
        self.paths.ensure("images", "json", "data", "plots")
        self.sent = []
        self.s = web.WebSession(self.cfg, self.paths, self.sent.append)

    def feed(self, grays, start_index=0):
        for i, g in enumerate(grays):
            self.s.handle_binary(frame_msg((start_index + i) * US_PER_FRAME, g))

    def of_type(self, t):
        return [m for m in self.sent if m["type"] == t]


# ── Timeline mapping ─────────────────────────────────────────


def test_frames_at_constant_rate_get_contiguous_indices():
    env = Session()
    env.feed([still_frame(1)] * 10)
    ticks = env.of_type("tick")
    assert [t["frame_index"] for t in ticks] == list(range(10))
    assert env.s.frames_seen == 10 and env.s.gap_filled == 0


def test_gap_is_filled_silently_and_counted():
    env = Session()
    g = still_frame(1)
    for i in (0, 1, 11):                       # frames 2..10 never arrive
        env.s.handle_binary(frame_msg(i * US_PER_FRAME, g))
    assert env.s.gap_filled == 9
    assert env.s.frames_seen == 3
    assert len(env.s.det.motion_signal()) == 12   # timeline stays contiguous
    # Filled frames emit no ticks — nothing new for the HUD.
    assert [t["frame_index"] for t in env.of_type("tick")] == [0, 1, 11]


def test_duplicate_timestamps_are_dropped():
    env = Session()
    g = still_frame(1)
    env.s.handle_binary(frame_msg(0.0, g))
    env.s.handle_binary(frame_msg(0.0, g))
    assert env.s.dup_dropped == 1
    assert len(env.s.det.motion_signal()) == 1


# ── Capture round trip ───────────────────────────────────────


def test_settle_requests_pixels_then_commits_the_jpeg():
    env = Session()
    env.feed([still_frame(1)] * 20)
    reqs = env.of_type("capture")
    assert len(reqs) == 1 and reqs[0]["reason"] == "initial"
    fi = reqs[0]["frame_index"]
    # The requested timestamp must be resolvable back to the frame we sent.
    assert reqs[0]["ts_us"] == fi * US_PER_FRAME

    env.s.handle_binary(jpeg_msg(fi))
    assert (env.paths.images / f"frame{fi:06d}.jpg").read_bytes().startswith(b"\xff\xd8")
    assert env.of_type("captured")[0]["count"] == 1
    kf = env.s.keyframes[0]
    assert kf["frame_index"] == fi and kf["source"] == "live"


def test_keyframe_entries_match_the_native_front_end_shape():
    env = Session()
    env.feed([still_frame(1)] * 20)
    env.s.handle_binary(jpeg_msg(env.of_type("capture")[0]["frame_index"]))
    assert list(env.s.keyframes[0].keys()) == [
        "frame_index", "time_sec", "motion_value", "sharpness", "filename",
        "spread_start", "spread_end", "spread_duration", "source"]


def test_unrequested_jpeg_is_ignored():
    env = Session()
    env.s.handle_binary(jpeg_msg(7))
    assert env.s.keyframes == [] and not list(env.paths.images.iterdir())


def test_undo_removes_the_file_and_reports_the_count():
    env = Session()
    env.feed([still_frame(1)] * 20)
    fi = env.of_type("capture")[0]["frame_index"]
    env.s.handle_binary(jpeg_msg(fi))
    env.s.handle_text('{"type": "undo"}')
    assert not (env.paths.images / f"frame{fi:06d}.jpg").exists()
    assert env.of_type("undone")[0]["count"] == 0 and env.s.keyframes == []


def test_capture_missing_drops_the_pending_capture():
    env = Session()
    env.feed([still_frame(1)] * 20)
    fi = env.of_type("capture")[0]["frame_index"]
    env.s.handle_text(json.dumps({"type": "capture_missing", "frame_index": fi}))
    env.s.handle_binary(jpeg_msg(fi))     # too late — must not commit
    assert env.s.keyframes == []
    # The operator must hear about a lost capture, not just the server log.
    assert any(str(fi) in m["text"] for m in env.of_type("notice"))


def test_force_before_any_frame_is_a_no_op():
    env = Session()
    env.s.handle_text('{"type": "force"}')
    assert env.of_type("capture") == []


def test_pause_reaches_the_detector():
    env = Session()
    env.s.handle_text('{"type": "pause", "value": true}')
    assert env.s.det.paused
    env.feed([still_frame(1)] * 40)
    assert env.of_type("capture") == []


# ── Recording chunks + finish ────────────────────────────────


def test_chunks_append_in_order_and_finish_writes_all_artifacts():
    env = Session()
    env.s.handle_text(json.dumps({"type": "hello", "width": 640, "height": 360,
                                  "frame_rate": 30, "mime": "video/webm"}))
    env.feed([still_frame(1)] * 20)
    env.s.handle_binary(jpeg_msg(env.of_type("capture")[0]["frame_index"]))
    env.s.handle_binary(chunk_msg(0, b"AB"))
    env.s.handle_binary(chunk_msg(1, b"CD"))
    env.s.handle_text('{"type": "finish"}')

    assert env.s.finished
    # Garbage bytes can't normalize; the raw file must survive for salvage.
    assert env.s.raw_path.read_bytes() == b"ABCD"
    meta = json.loads((env.paths.json / "metadata.json").read_text())
    assert meta["web"]["normalize"].startswith("failed")
    assert meta["capture_source"] == "live-web"
    assert meta["total_frames"] == 20 and meta["original_width"] == 640

    for rel in ("data/motion_signal.npy", "data/smoothed_signal.npy",
                "data/peaks.npy", "json/spreads.json", "json/keyframes.json"):
        assert (Path(env.cfg.output_dir) / rel).exists(), rel
    assert len(np.load(env.paths.data / "motion_signal.npy")) == 20
    kfs = json.loads((env.paths.json / "keyframes.json").read_text())
    assert len(kfs) == 1
    assert env.of_type("finished")[0]["keyframes"] == 1


def test_finish_without_recording_reports_it_and_still_writes_artifacts():
    env = Session()
    env.feed([still_frame(1)] * 8)
    env.s.finish()
    meta = json.loads((env.paths.json / "metadata.json").read_text())
    assert meta["web"]["normalize"] == "no recording received"
    assert meta["total_frames"] == 8


# ── CFR normalization (OpenCV path) ──────────────────────────


def test_normalize_opencv_path_produces_a_readable_cfr_video():
    import cv2
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        out = Path(tmp) / "out.mp4"
        w = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
        for i in range(12):
            w.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
        w.release()
        status = web.normalize_recording(raw, out, 30.0, use_ffmpeg=False)
        assert status.startswith("ok"), status
        cap = cv2.VideoCapture(str(out))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert 10 <= n <= 14, f"expected ~12 frames, got {n}"


def test_normalize_missing_file_fails_cleanly():
    status = web.normalize_recording("/nonexistent/raw.webm", "/nonexistent/out.mp4",
                                     30.0, use_ffmpeg=False)
    assert status.startswith("failed")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
