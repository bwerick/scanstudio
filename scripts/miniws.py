"""
Minimal WebSocket framing (RFC 6455) over an already-connected socket.

The browser front end (p0_web_capture) talks to Chrome over one WebSocket:
JSON control messages one way, analysis frames / JPEGs / recorder chunks the
other. That needs exactly the server half of RFC 6455 — handshake key,
frame parsing with masking, fragment reassembly, ping/pong, close — and
nothing else: no wss, no extensions, no subprotocols, no async. Hand-rolling
those ~150 lines keeps requirements.txt untouched, which matters most on the
ChromeOS/Crostini setups this exists for, where every extra dependency is one
more thing to install inside the container.

Client mode (outgoing frames masked, as the RFC requires of clients) exists so
the tests can drive a real server socket without a browser.
"""

import base64
import hashlib
import os
import struct

# Handshake GUID fixed by RFC 6455 §1.3.
_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# Opcodes (RFC 6455 §5.2). CONT is internal to reassembly and never returned.
TEXT, BINARY, CLOSE, PING, PONG = 0x1, 0x2, 0x8, 0x9, 0xA
_CONT = 0x0


def accept_key(client_key: str) -> str:
    """Sec-WebSocket-Accept value for a client's Sec-WebSocket-Key."""
    digest = hashlib.sha1((client_key + _GUID).encode()).digest()
    return base64.b64encode(digest).decode()


class WebSocket:
    """Framed messages over a connected socket.

    ``recv()`` returns the next complete *message* as ``(opcode, bytes)`` —
    ping, pong and continuation frames are handled internally, and a peer
    close is answered and surfaced as ``(CLOSE, b"")``. Sends write whole
    unfragmented messages. All calls block; use one thread per connection.
    """

    def __init__(self, sock, client=False, max_message=64 * 1024 * 1024):
        self.sock = sock
        self.client = client        # clients mask outgoing frames (RFC §5.3)
        self.max_message = max_message

    # ── Receiving ────────────────────────────────────────────

    def recv(self):
        """Next complete message: ``(TEXT|BINARY, payload)`` or ``(CLOSE, b"")``."""
        opcode, parts, total = None, [], 0
        while True:
            fin, op, payload = self._read_frame()
            if op == PING:
                self._send_frame(PONG, payload)
                continue
            if op == PONG:
                continue
            if op == CLOSE:
                try:                       # echo the close (status code only)
                    self._send_frame(CLOSE, payload[:2])
                except OSError:
                    pass
                return CLOSE, b""
            if op in (TEXT, BINARY):
                if opcode is not None:
                    raise ConnectionError("new message inside a fragmented one")
                opcode = op
            elif op == _CONT:
                if opcode is None:
                    raise ConnectionError("continuation frame with no start")
            else:
                raise ConnectionError(f"unsupported opcode {op:#x}")
            parts.append(payload)
            total += len(payload)
            if total > self.max_message:
                raise ConnectionError(f"message exceeds {self.max_message} bytes")
            if fin:
                return opcode, b"".join(parts)

    def _read_frame(self):
        b0, b1 = self._read_exact(2)
        fin, opcode = bool(b0 & 0x80), b0 & 0x0F
        masked, length = bool(b1 & 0x80), b1 & 0x7F
        if length == 126:
            (length,) = struct.unpack(">H", self._read_exact(2))
        elif length == 127:
            (length,) = struct.unpack(">Q", self._read_exact(8))
        if length > self.max_message:
            raise ConnectionError(f"frame exceeds {self.max_message} bytes")
        mask = self._read_exact(4) if masked else None
        payload = self._read_exact(length) if length else b""
        if mask:
            payload = _mask(payload, mask)
        return fin, opcode, payload

    def _read_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(min(n - len(buf), 1 << 20))
            if not chunk:
                raise ConnectionError("socket closed mid-frame")
            buf += chunk
        return bytes(buf)

    # ── Sending ──────────────────────────────────────────────

    def send_text(self, s: str):
        self._send_frame(TEXT, s.encode())

    def send_binary(self, b: bytes):
        self._send_frame(BINARY, b)

    def close(self, code=1000):
        try:
            self._send_frame(CLOSE, struct.pack(">H", code))
        except OSError:
            pass

    def _send_frame(self, opcode, payload):
        header = bytearray([0x80 | opcode])       # FIN always set: no fragmenting
        mask_bit = 0x80 if self.client else 0
        n = len(payload)
        if n < 126:
            header.append(mask_bit | n)
        elif n < 1 << 16:
            header.append(mask_bit | 126)
            header += struct.pack(">H", n)
        else:
            header.append(mask_bit | 127)
            header += struct.pack(">Q", n)
        if self.client:
            mask = os.urandom(4)
            header += mask
            payload = _mask(payload, mask)
        self.sock.sendall(bytes(header) + payload)


def _mask(payload: bytes, mask: bytes) -> bytes:
    """XOR ``payload`` with the repeating 4-byte mask (its own inverse).

    Done as one big-int XOR: CPython evaluates that in C, which keeps even the
    ~230 KB analysis frames at 30 fps far from mattering.
    """
    n = len(payload)
    if n == 0:
        return payload
    full = mask * (n // 4) + mask[:n % 4]
    return (int.from_bytes(payload, "little")
            ^ int.from_bytes(full, "little")).to_bytes(n, "little")
