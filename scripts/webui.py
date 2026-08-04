"""
Shared HTTP/WebSocket plumbing for the browser review UIs (p4/p7 web).

p0_web_capture proved the pattern — a stdlib ThreadingHTTPServer, a hand-run
RFC 6455 upgrade on /ws (miniws), one self-contained HTML page — and the two
review servers repeat it, so the repeated half lives here: the server class,
the page/websocket routing, and static file replies. The one genuinely new
piece is Range support (send_file): the p4 scrubber is a <video> element, and
Chrome will not seek a video it cannot byte-range into.

Sessions are single-client, like capture: review mutates one JSON state, and
two browsers editing it would silently overwrite each other. A second
connection is refused with an error message rather than queued.
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from miniws import CLOSE, TEXT, WebSocket, accept_key
from utils import log

CHUNK = 1 << 20

CTYPES = {
    ".html": "text/html; charset=utf-8",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".json": "application/json",
}


def send_file(handler, path, head_only=False):
    """Reply with a file, honoring a single-range Range header (RFC 7233).

    Streams from disk in 1 MB chunks — the scrubber's recording can be
    gigabytes — and always advertises Accept-Ranges so Chrome knows the
    <video> is seekable. An unsatisfiable range gets the 416 the spec asks
    for; a malformed one is ignored (full 200 reply), which is also what the
    spec asks for.
    """
    try:
        size = path.stat().st_size
    except OSError:
        handler.send_error(404)
        return
    ctype = CTYPES.get(path.suffix.lower(), "application/octet-stream")
    start, end = 0, size - 1
    rng = handler.headers.get("Range")
    partial = False
    if rng and rng.startswith("bytes=") and "," not in rng:
        lo, _, hi = rng[6:].partition("-")
        try:
            if lo:
                start = int(lo)
                end = int(hi) if hi else size - 1
            elif hi:                      # suffix form: last N bytes
                start = max(0, size - int(hi))
            else:
                raise ValueError
            partial = True
        except ValueError:
            start, end, partial = 0, size - 1, False
        if partial and (start >= size or start > end):
            handler.send_response(416)
            handler.send_header("Content-Range", f"bytes */{size}")
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
        end = min(end, size - 1)
    handler.send_response(206 if partial else 200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(end - start + 1))
    if partial:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    if head_only:
        return
    try:
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(CHUNK, left))
                if not chunk:
                    break
                handler.wfile.write(chunk)
                left -= len(chunk)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass                       # the browser aborted a seek; routine


class WebUIServer(ThreadingHTTPServer):
    """HTTP server hosting one review page, one websocket session at a time.

    ``make_session(send) -> session`` builds the app object for a connection;
    the session must expose ``hello()`` (called once, to push initial state)
    and ``handle(msg: dict)``. ``route(handler, path)`` — an optional
    callable — serves the app's file endpoints (images, video) and returns
    True when it handled the path.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, html_path, make_session, route=None):
        super().__init__(addr, WebUIHandler)
        self.html_path = html_path
        self.make_session = make_session
        self.route = route
        self.session_lock = threading.Lock()
        self.done = threading.Event()


class WebUIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):      # keep utils.log the only voice
        pass

    def do_HEAD(self):
        self._get(head_only=True)

    def do_GET(self):
        self._get(head_only=False)

    def _get(self, head_only):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            body = self.server.html_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if not head_only:
                self.wfile.write(body)
        elif path == "/ws":
            if head_only:
                self.send_error(400)
            else:
                self._websocket()
        elif self.server.route is not None and self.server.route(
            self, path, head_only
        ):
            pass
        else:
            self.send_error(404)

    def _websocket(self):
        key = self.headers.get("Sec-WebSocket-Key")
        if key is None or "websocket" not in self.headers.get("Upgrade", "").lower():
            self.send_error(400, "WebSocket upgrade expected")
            return
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept_key(key))
        self.end_headers()
        self.wfile.flush()
        self.close_connection = True

        ws = WebSocket(self.connection)
        if not self.server.session_lock.acquire(blocking=False):
            ws.send_text(json.dumps({
                "type": "error",
                "message": "another review session is already connected",
            }))
            ws.close()
            return
        try:
            self._run_session(ws)
        finally:
            self.server.session_lock.release()

    def _run_session(self, ws):
        # Background threads (the p4 watchdog) push over the same socket the
        # handler replies on, so every send goes through one lock.
        send_lock = threading.Lock()

        def send(msg):
            try:
                with send_lock:
                    ws.send_text(json.dumps(msg))
            except (OSError, ConnectionError):
                pass

        session = self.server.make_session(send)
        log("  Browser connected")
        try:
            session.hello()
            # A session that sets ``finished`` (the browser's Finish button)
            # ends the review server itself, so a chained make target
            # (finish-web) proceeds to the next phase — the web analogue of
            # closing the Tk window.
            while not getattr(session, "finished", False):
                op, payload = ws.recv()
                if op == CLOSE:
                    break
                if op == TEXT:
                    session.handle(json.loads(payload.decode()))
        except (ConnectionError, OSError) as e:
            log(f"  Connection lost: {e}")
        except Exception as e:      # a protocol bug must not kill the server
            log(f"  ERROR in session: {e!r}")
        finally:
            close = getattr(session, "close", None)
            if close is not None:
                close()
        ws.close()
        log("  Browser disconnected")
        if getattr(session, "finished", False):
            self.server.done.set()


def serve_forever_in_thread(server):
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return t


def chromeos_note(port):
    log(f"  Open   http://localhost:{port}   in Chrome.")
    log("")
    log("  ChromeOS: Chrome runs outside the Linux container — add port "
        f"{port} under Settings > Linux > Port forwarding first.")
