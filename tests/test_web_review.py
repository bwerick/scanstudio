"""
End-to-end tests of the browser review servers (p4/p7 web) — no browser.

Boots the real HTTP/WebSocket servers on ephemeral ports and plays the
page's role over an actual socket, miniws in client mode: state on connect,
actions, editor confirm, save, and the file endpoints (images with Range —
the contract the p4 scrubber's <video> depends on).

Run standalone (`python tests/test_web_review.py`) or under pytest.
"""

import json
import shutil
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import p4_web_review  # noqa: E402
import p7_web_review  # noqa: E402
from miniws import TEXT, WebSocket, accept_key  # noqa: E402

KEY = "dGhlIHNhbXBsZSBub25jZQ=="


def ws_connect(port):
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    sock.settimeout(20)
    sock.sendall((f"GET /ws HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                  "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                  f"Sec-WebSocket-Key: {KEY}\r\n"
                  "Sec-WebSocket-Version: 13\r\n\r\n").encode())
    head = b""
    while b"\r\n\r\n" not in head:
        head += sock.recv(1)
    assert b" 101 " in head.split(b"\r\n", 1)[0], head
    assert accept_key(KEY).encode() in head
    return WebSocket(sock, client=True), sock


def next_msg(ws):
    op, payload = ws.recv()
    assert op == TEXT, f"unexpected opcode {op}"
    return json.loads(payload.decode())


def read_until(ws, wanted, limit=800):
    for _ in range(limit):
        m = next_msg(ws)
        if m["type"] == wanted:
            return m
    raise AssertionError(f"no '{wanted}' message within {limit} messages")


def send(ws, msg):
    ws.send_text(json.dumps(msg))


# ── P4 fixtures: a small project with a real page-on-table look ──

def make_p4_project(root):
    """Three frames of a bright 'page' on a dark table, plus the metadata
    the review needs. Big enough that page_mask's 25 px morphology keeps
    the page blob; consensus needs >= 3 same-sized frames."""
    out = root / "out"
    for d in ("images", "json", "data"):
        (out / d).mkdir(parents=True)
    kfs = []
    for i, fi in enumerate((10, 40, 70)):
        img = np.full((400, 640, 3), 30, np.uint8)
        img[60:340, 120:520] = (235, 240, 245)   # the "spread"
        name = f"frame{fi:06d}.jpg"
        cv2.imwrite(str(out / "images" / name), img)
        kfs.append({"frame_index": fi, "time_sec": fi / 30.0,
                    "motion_value": 1.0, "sharpness": 100.0,
                    "filename": name, "source": "test"})
    (out / "json" / "keyframes.json").write_text(json.dumps(kfs))
    (out / "json" / "metadata.json").write_text(json.dumps({"fps": 30.0}))
    np.save(str(out / "data" / "smoothed_signal.npy"), np.ones(100))
    return out


def start_p4(tmp):
    out = make_p4_project(Path(tmp))
    args = p4_web_review.parse_args([str(out), "--port", "0"])
    server = p4_web_review.build_server(args)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1], out


def test_p4_state_actions_confirm_and_save():
    with tempfile.TemporaryDirectory() as tmp:
        server, port, out = start_p4(tmp)
        ws, sock = ws_connect(port)
        state = read_until(ws, "state")
        assert state["mode"] == "double"
        assert len(state["keyframes"]) == 3
        sig = read_until(ws, "signal")
        assert sig["video"] is False and len(sig["smoothed"]) == 100
        # The consensus vote runs off the connect path; a second state is
        # pushed when it lands, and only then does geometry resolve.
        read_until(ws, "state")

        # Frame geometry resolves (consensus voted from the 3 frames).
        send(ws, {"type": "show", "idx": 0})
        fr = read_until(ws, "frame")
        assert fr["idx"] == 0
        assert fr["geom"] is not None and len(fr["geom"]["box"]) == 4

        # Keep validates and pins the geometry onto the keyframe.
        send(ws, {"type": "action", "idx": 0, "action": "keep"})
        state = read_until(ws, "state")
        assert state["actions"]["0"] == "keep" and 0 in state["validated"]

        # Editor round trip: seed, then confirm a manual box + gutter.
        send(ws, {"type": "seed", "idx": 1})
        seed = read_until(ws, "seed")
        assert seed["W"] == 640 and len(seed["box"]) == 4
        box = [[0.2, 0.15], [0.8, 0.15], [0.8, 0.85], [0.2, 0.85]]
        send(ws, {"type": "confirm", "idx": 1, "gutter": 0.5, "box": box,
                  "box_dirty": True, "rotation_deg": 0.0})
        state = read_until(ws, "state")
        assert state["actions"]["1"] == "keep"

        # Delete the third frame, then save; the file goes, the log appends.
        send(ws, {"type": "action", "idx": 2, "action": "dup"})
        read_until(ws, "state")
        send(ws, {"type": "save"})
        saved = read_until(ws, "saved")
        # untracked: keyframes the drift sweep never reached, so Phase 5 will
        # crop them with the anchor box unadjusted. Reported so a mid-sweep
        # save can't silently ship stale boxes.
        assert saved == {"type": "saved", "deleted": 1, "kept": 2,
                         "untracked": saved["untracked"]}
        assert isinstance(saved["untracked"], int)

        kfs = json.loads((out / "json" / "keyframes.json").read_text())
        assert len(kfs) == 2
        assert kfs[0]["validated"] and kfs[0]["crop_quad"] is not None
        assert kfs[1]["crop_quad"] == box and kfs[1]["gutter"] == 0.5
        assert not (out / "images" / "frame000070.jpg").exists()
        rl = json.loads((out / "json" / "review_log.json").read_text())
        assert rl["sessions"][-1]["deletions"][0]["reason"] == "dup"
        assert rl["sessions"][-1]["frontend"] == "web"

        # Finish saves again and releases the server — what lets a chained
        # `make finish-web` proceed to P5.
        send(ws, {"type": "finish"})
        read_until(ws, "bye")
        assert server.done.wait(timeout=10)

        ws.close()
        sock.close()
        server.shutdown()


def test_p4_serves_page_and_ranged_images():
    with tempfile.TemporaryDirectory() as tmp:
        server, port, out = start_p4(tmp)
        base = f"http://127.0.0.1:{port}"

        with urllib.request.urlopen(base + "/") as r:
            assert r.status == 200
            assert b"Review Keyframes" in r.read()

        full = (out / "images" / "frame000010.jpg").read_bytes()
        with urllib.request.urlopen(base + "/img/frame000010.jpg") as r:
            assert r.status == 200
            assert r.headers["Accept-Ranges"] == "bytes"
            assert r.read() == full

        # The Range contract Chrome's <video> needs from /video, exercised
        # on an image (same send_file path).
        req = urllib.request.Request(base + "/img/frame000010.jpg",
                                     headers={"Range": "bytes=10-29"})
        with urllib.request.urlopen(req) as r:
            assert r.status == 206
            assert r.headers["Content-Range"] == f"bytes 10-29/{len(full)}"
            assert r.read() == full[10:30]

        req = urllib.request.Request(base + "/img/frame000010.jpg",
                                     headers={"Range": f"bytes={len(full)}-"})
        try:
            urllib.request.urlopen(req)
            raise AssertionError("unsatisfiable range must 416")
        except urllib.error.HTTPError as e:
            assert e.code == 416

        server.shutdown()


# ── P4 scrub source: browser-playable recording, or a proxy ──

def make_recording(path, fourcc, n=24, size=(64, 48)):
    """A tiny video in ``fourcc``; None if this build can't write it."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 30.0, size)
    if not w.isOpened():
        w.release()
        return None
    for i in range(n):
        frame = np.full((size[1], size[0], 3), (i * 9) % 255, np.uint8)
        w.write(frame)
    w.release()
    return path


def p4_session(out, video, send=None):
    args = p4_web_review.parse_args([str(out), str(video), "--mode", "single"])
    from utils import ProjectPaths
    return p4_web_review.ReviewSession(args, ProjectPaths(str(out)),
                                       send or (lambda m: None))


def test_p4_plays_a_browser_codec_directly():
    with tempfile.TemporaryDirectory() as tmp:
        out = make_p4_project(Path(tmp))
        vid = make_recording(Path(tmp) / "h264.mp4", "avc1")
        if vid is None:
            return                       # build without an H.264 encoder
        s = p4_session(out, vid)
        s._init_scrub_source()
        assert s.proxy_state == "native", s.proxy_state
        assert s.scrub_video() == vid    # served straight, no transcode
        assert s.video_frames == 24
        assert not s.proxy_path.exists()


def test_p4_transcodes_a_proxy_for_a_codec_no_browser_plays():
    if shutil.which("ffmpeg") is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = make_p4_project(Path(tmp))
        vid = make_recording(Path(tmp) / "mp4v.mp4", "mp4v")
        msgs = []
        s = p4_session(out, vid, msgs.append)
        s._init_scrub_source()
        assert s.proxy_state == "building"
        assert s.scrub_video() == vid    # the original until the proxy lands

        # Grabbing before the proxy exists must not insert: a scrubber whose
        # <video> never loaded reports frame 0 for every position.
        before = len(s.keyframes)
        s.handle({"type": "insert", "frame_index": 0})
        assert len(s.keyframes) == before and not s.pending_inserts
        assert msgs[-1]["type"] == "notice"

        s._proxy_thread.join(timeout=120)
        assert s.proxy_state == "ready", s.proxy_msg
        assert s.scrub_video() == s.proxy_path
        assert not s.proxy_path.with_suffix(".part.mp4").exists()
        # Frame-for-frame with the recording: the scrubber addresses the
        # proxy by index and grabs that index from the original.
        assert s._probe(s.proxy_path)[1] == s.video_frames
        assert [m["state"] for m in msgs if m["type"] == "proxy"][-1] == "ready"

        # A proxy that still lines up is reused by the next session.
        s2 = p4_session(out, vid)
        s2._init_scrub_source()
        assert s2.proxy_state == "ready" and s2._proxy_thread is None

        # One that doesn't (here: a longer recording) is rebuilt.
        vid2 = make_recording(Path(tmp) / "mp4v2.mp4", "mp4v", n=48)
        s3 = p4_session(out, vid2)
        s3._init_scrub_source()
        assert s3.proxy_state == "building"
        s3._proxy_thread.join(timeout=120)
        assert s3._probe(s3.proxy_path)[1] == 48


def test_p4_says_so_when_it_cannot_make_a_proxy():
    with tempfile.TemporaryDirectory() as tmp:
        out = make_p4_project(Path(tmp))
        vid = make_recording(Path(tmp) / "mp4v.mp4", "mp4v")
        real_which = shutil.which
        p4_web_review.shutil.which = lambda n, *a, **k: (
            None if n == "ffmpeg" else real_which(n, *a, **k))
        try:
            s = p4_session(out, vid)
            s._init_scrub_source()
        finally:
            p4_web_review.shutil.which = real_which
        assert s.proxy_state == "unavailable"
        assert "make ffmpeg" in s.proxy_msg
        before = len(s.keyframes)
        s.handle({"type": "insert", "frame_index": 0})
        assert len(s.keyframes) == before and not s.pending_inserts


# ── P7 ───────────────────────────────────────────────────────

def make_p7_project(root):
    out = root / "out"
    for d in ("pages", "json"):
        (out / d).mkdir(parents=True)
    pages = []
    for pn in (1, 2, 3):
        img = np.full((200, 150, 3), 240, np.uint8)
        cv2.putText(img, str(pn), (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (0, 0, 0), 8)
        fn = f"page_{pn:03d}.jpg"
        cv2.imwrite(str(out / "pages" / fn), img)
        pages.append({"page_num": pn, "filename": fn, "type": "left"})
    (out / "json" / "pages.json").write_text(json.dumps(pages))
    return out


def start_p7(tmp):
    out = make_p7_project(Path(tmp))
    args = p7_web_review.parse_args([str(out), "--port", "0"])
    server = p7_web_review.build_server(args)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1], out


def test_p7_update_and_save_apply_everything():
    with tempfile.TemporaryDirectory() as tmp:
        server, port, out = start_p7(tmp)
        ws, sock = ws_connect(port)
        state = read_until(ws, "state")
        assert [p["page_num"] for p in state["pages"]] == [1, 2, 3]

        # Mirror a working state: drop page 2, tag + name page 3, tilt page 1.
        send(ws, {"type": "update",
                  "notes": {"3": "Appendix"},
                  "drops": [2],
                  "geometry": {"1": {"rot": 1.5, "dx": 0.01, "dy": 0.0}},
                  "doc_starts": [3]})
        # The mirror lands in page_review.json before any save. The update
        # has no reply, so poll briefly for the write.
        deadline = time.monotonic() + 5
        while (time.monotonic() < deadline
               and not (out / "json" / "page_review.json").exists()):
            time.sleep(0.02)
        pr = json.loads((out / "json" / "page_review.json").read_text())
        assert pr["drops"] == [2] and pr["doc_starts"] == [3]

        send(ws, {"type": "save"})
        saved = read_until(ws, "saved")
        assert saved["dropped"] == 1 and saved["adjusted"] == 1
        assert saved["documents"] == 2

        pages = json.loads((out / "json" / "pages.json").read_text())
        assert [p["page_num"] for p in pages] == [1, 3]
        assert not (out / "pages" / "page_002.jpg").exists()
        p3 = next(p for p in pages if p["page_num"] == 3)
        assert p3["is_doc_start"] and p3["doc_title"] == "Appendix"
        p1 = next(p for p in pages if p["page_num"] == 1)
        assert p1["geometry"]["rot"] == 1.5
        # The pristine copy exists so the nudge is re-renderable.
        assert (out / "pages_orig" / "page_001.jpg").exists()

        # Clearing the nudge restores the original and drops the copy.
        send(ws, {"type": "update", "notes": {"3": "Appendix"}, "drops": [],
                  "geometry": {}, "doc_starts": [3]})
        send(ws, {"type": "save"})
        read_until(ws, "saved")
        assert not (out / "pages_orig").exists()

        send(ws, {"type": "finish"})
        read_until(ws, "bye")
        assert server.done.wait(timeout=10)

        ws.close()
        sock.close()
        server.shutdown()


def test_p7_serves_pages():
    with tempfile.TemporaryDirectory() as tmp:
        server, port, out = start_p7(tmp)
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(base + "/") as r:
            assert b"Page Review" in r.read()
        full = (out / "pages" / "page_001.jpg").read_bytes()
        with urllib.request.urlopen(base + "/page/page_001.jpg") as r:
            assert r.read() == full
        server.shutdown()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok    {name}")
    print("all web review tests passed")
