"""
End-to-end test of the browser capture server — no browser required.

Boots the real HTTP/WebSocket server (p0_web_capture) on an ephemeral port
and plays the browser's role over an actual socket, using miniws in client
mode: HTTP GET for the page, the RFC 6455 upgrade by hand, then the full
session protocol — config, hello, analysis frames, the capture round trip,
a recorder chunk, finish. This is the one place the handshake-inside-
http.server integration is exercised; the session logic itself is covered
unit-style in test_web_session.

Run standalone (`python tests/test_web_capture_e2e.py`) or under pytest.
"""

import json
import socket
import struct
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import p0_web_capture as web  # noqa: E402
from miniws import TEXT, WebSocket, accept_key  # noqa: E402

US_PER_FRAME = 1e6 / 30.0
KEY = "dGhlIHNhbXBsZSBub25jZQ=="


def start_server(tmp):
    root = Path(tmp)
    args = web.parse_args([str(root / "out"), str(root / "rec.mp4"),
                           "--port", "0", "--settle-time", "0.2",
                           "--smoothing-window", "5"])
    server = web.build_server(args)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1], args


def ws_connect(port):
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    sock.settimeout(10)
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


def read_until(ws, wanted, limit=500):
    for _ in range(limit):
        m = next_msg(ws)
        if m["type"] == wanted:
            return m
    raise AssertionError(f"no '{wanted}' message within {limit} messages")


def frame_msg(ts_us, gray):
    return (struct.pack("<BdHH", web.MSG_FRAME, ts_us,
                        gray.shape[1], gray.shape[0]) + gray.tobytes())


def test_full_session_over_a_real_socket():
    with tempfile.TemporaryDirectory() as tmp:
        server, port, args = start_server(tmp)
        try:
            # The page is served, self-contained, at the root.
            html = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read()
            assert b"ScanStudio" in html and b"MediaStreamTrackProcessor" in html
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/nope", timeout=5)
                raise AssertionError("expected a 404")
            except urllib.error.HTTPError as e:
                assert e.code == 404

            ws, sock = ws_connect(port)
            cfg = next_msg(ws)
            assert cfg["type"] == "config" and cfg["settle_frames"] == 6

            # A second tab must be refused while a session is live.
            ws2, sock2 = ws_connect(port)
            assert next_msg(ws2)["type"] == "error"
            sock2.close()

            ws.send_text(json.dumps({"type": "hello", "width": 640, "height": 360,
                                     "frame_rate": 30, "mime": "video/webm"}))
            gray = np.random.default_rng(1).integers(
                0, 200, size=(90, 160), dtype=np.int16).astype(np.uint8)
            for i in range(12):     # identical frames -> settles at frame 5
                ws.send_binary(frame_msg(i * US_PER_FRAME, gray))

            req = read_until(ws, "capture")
            assert req["reason"] == "initial"
            ws.send_binary(struct.pack("<BI", web.MSG_JPEG, req["frame_index"])
                           + b"\xff\xd8 e2e jpeg")
            assert read_until(ws, "captured")["count"] == 1

            ws.send_binary(struct.pack("<BI", web.MSG_CHUNK, 0) + b"RAWBYTES")
            ws.send_text('{"type": "finish"}')
            fin = read_until(ws, "finished")
            assert fin["keyframes"] == 1 and fin["total_frames"] == 12

            out = Path(tmp) / "out"
            assert (out / "images" / f"frame{req['frame_index']:06d}.jpg").exists()
            assert (out / "json" / "keyframes.json").exists()
            meta = json.loads((out / "json" / "metadata.json").read_text())
            assert meta["capture_source"] == "live-web"
            assert server.done.wait(5), "server should signal completion"
            sock.close()
        finally:
            server.shutdown()


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
