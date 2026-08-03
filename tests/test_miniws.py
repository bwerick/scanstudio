"""
Tests for the minimal WebSocket framing (miniws).

Everything runs over a socketpair — one end in client mode (masked frames,
what a browser sends), one end the server. Hand-built byte vectors cover the
wire format where behavior tests alone could hide a symmetric bug (a codec
that masks wrongly on both ends still roundtrips).

Run standalone (`python tests/test_miniws.py`) or under pytest.
"""

import os
import socket
import struct
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from miniws import BINARY, CLOSE, PING, PONG, TEXT, WebSocket, accept_key  # noqa: E402


def pair():
    a, b = socket.socketpair()
    return WebSocket(a, client=True), WebSocket(b), a, b


def masked_frame(opcode, payload, fin=True, mask=b"\x01\x02\x03\x04"):
    """Hand-build one client frame, independent of the codec under test."""
    b0 = (0x80 if fin else 0) | opcode
    n = len(payload)
    if n < 126:
        header = bytes([b0, 0x80 | n])
    elif n < 1 << 16:
        header = bytes([b0, 0x80 | 126]) + struct.pack(">H", n)
    else:
        header = bytes([b0, 0x80 | 127]) + struct.pack(">Q", n)
    body = bytes(c ^ mask[i % 4] for i, c in enumerate(payload))
    return header + mask + body


# ── Handshake ────────────────────────────────────────────────


def test_accept_key_matches_rfc6455_example():
    assert accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


# ── Framing ──────────────────────────────────────────────────


def test_hand_built_masked_frame_is_decoded():
    _, server, a, b = pair()
    a.sendall(masked_frame(TEXT, b"abc"))
    assert server.recv() == (TEXT, b"abc")
    a.close(), b.close()


def test_text_roundtrip_client_to_server_is_masked_on_the_wire():
    client, server, a, b = pair()
    client.send_text("hello")
    # The wire bytes must have the mask bit set and NOT contain the plaintext.
    raw = b.recv(1024, socket.MSG_PEEK)
    assert raw[1] & 0x80, "client frames must be masked"
    assert b"hello" not in raw
    assert server.recv() == (TEXT, b"hello")
    a.close(), b.close()


def test_binary_server_to_client_is_unmasked():
    client, server, a, b = pair()
    payload = bytes(range(256))
    server.send_binary(payload)
    raw = a.recv(4096, socket.MSG_PEEK)
    assert not raw[1] & 0x80, "server frames must not be masked"
    assert client.recv() == (BINARY, payload)
    a.close(), b.close()


def test_length_encodings_7_16_and_64_bit():
    client, server, a, b = pair()
    for size in (125, 300, 70000):     # 7-bit, 16-bit, 64-bit length paths
        payload = os.urandom(size)
        t = threading.Thread(target=client.send_binary, args=(payload,))
        t.start()                      # sender thread: 70 KB > socketpair buffer
        assert server.recv() == (BINARY, payload), f"size {size}"
        t.join()
    a.close(), b.close()


def test_fragmented_message_is_reassembled():
    _, server, a, b = pair()
    a.sendall(masked_frame(TEXT, b"he", fin=False))
    a.sendall(masked_frame(0x0, b"ll", fin=False))   # continuation
    a.sendall(masked_frame(0x0, b"o", fin=True))
    assert server.recv() == (TEXT, b"hello")
    a.close(), b.close()


# ── Control frames ───────────────────────────────────────────


def test_ping_is_answered_with_pong_and_skipped():
    client, server, a, b = pair()
    client._send_frame(PING, b"x")
    client.send_text("after")
    assert server.recv() == (TEXT, b"after")   # ping handled invisibly
    fin, op, payload = client._read_frame()
    assert (fin, op, payload) == (True, PONG, b"x")
    a.close(), b.close()


def test_close_is_echoed_and_surfaced():
    client, server, a, b = pair()
    client.close(code=1000)
    assert server.recv() == (CLOSE, b"")
    fin, op, payload = client._read_frame()
    assert op == CLOSE and struct.unpack(">H", payload)[0] == 1000
    a.close(), b.close()


def test_torn_socket_raises_connection_error():
    _, server, a, b = pair()
    a.sendall(masked_frame(TEXT, b"abc")[:3])   # cut off mid-frame
    a.close()
    try:
        server.recv()
        raise AssertionError("expected ConnectionError")
    except ConnectionError:
        pass
    b.close()


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
