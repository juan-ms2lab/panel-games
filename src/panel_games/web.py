"""HTTP server (standard library): game catalog, phone controller endpoints, the controller page and /screen.

Open to the whole LAN: any phone may start, stop, join and play. Writes need the X-Requested-With header
(blocks simple cross-site form posts), and each client is rate limited.
"""

import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .games.pad import PAD

STATIC_DIR = Path(__file__).parent / "static"
MAX_JSON = 1 << 16
MAX_MESSAGE = 40000
LOOPBACK_IPS = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
HEADER = "panel-games"
GAME_PATHS = ("/api/game/join", "/api/game/input", "/api/game/leave", "/api/game/message", "/api/game/state")


class RateLimiter:
    """Token buckets keyed by (client, action)."""

    def __init__(self):
        self._buckets: dict[tuple, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: tuple, per_second: float, burst: float = 1.0) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (burst, now))
            tokens = min(burst, tokens + (now - last) * per_second)
            ok = tokens >= 1
            self._buckets[key] = (tokens - 1 if ok else tokens, now)
            if len(self._buckets) > 5000:
                cutoff = now - 600
                self._buckets = {k: v for k, v in self._buckets.items() if v[1] > cutoff}
            return ok


class Handler(BaseHTTPRequestHandler):
    server_version = "panel-games"

    def log_message(self, *args):
        pass

    def _send(self, code, body: bytes, ctype="application/json", headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def _error(self, code, msg):
        self._json({"error": msg}, code)

    def _json_body(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("bad Content-Length")
        if not 0 <= n <= MAX_JSON:
            raise ValueError("body too large")
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError("invalid JSON")
        if not isinstance(body, dict):
            raise ValueError("body must be a JSON object")
        return body

    def _handle(self, method):
        url = urlparse(self.path)
        try:
            if method == "POST" and self.headers.get("X-Requested-With") != HEADER:
                raise PermissionError("missing X-Requested-With header")
            self._dispatch(method, url.path, parse_qs(url.query))
        except PermissionError as e:
            self._error(403, str(e))
        except FileNotFoundError:
            self._error(404, "not found")
        except ValueError as e:
            self._error(400, str(e))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            print(f"web error on {method} {url.path}: {type(e).__name__}: {e}")
            self._error(500, "internal error")

    do_GET = lambda self: self._handle("GET")
    do_POST = lambda self: self._handle("POST")

    def _dispatch(self, method, path, query):
        host, ip = self.server.host, self.client_address[0]
        limiter = self.server.limiter
        if method == "GET":
            if path in ("/", "/index.html"):
                return self._send(200, (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/screen":
                return self._send(200, (STATIC_DIR / "screen.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/games":
                return self._json(host.catalog())
            if path == "/api/status":
                return self._json({"game": host.status(), "error": host.error,
                                   "target": f"{host.target[0]}x{host.target[1]}",
                                   "urls": lan_urls(self.server.server_address[1])})
            if path == "/api/screen.png":
                if self.server.screen is None:
                    raise FileNotFoundError
                count, png = self.server.screen.png()
                if query.get("since", [""])[0] == str(count):
                    return self._send(204, b"", "image/png")
                return self._send(200, png, "image/png", {"X-Frame": str(count)})
            raise FileNotFoundError
        if method != "POST":
            raise FileNotFoundError
        if path in ("/api/game/start", "/api/game/stop"):
            if not limiter.allow((ip, "start"), 0.5, 3):
                return self._error(429, "slow down")
            if path == "/api/game/start":
                host.start(str(self._json_body().get("id", "")))
            else:
                host.stop()
            return self._json({"ok": True, "game": host.status()})
        if path not in GAME_PATHS:
            raise FileNotFoundError
        if not limiter.allow((ip, "game"), 60, 90):
            return self._error(429, "slow down")
        body = self._json_body()
        token = str(body.get("token", ""))
        if path == "/api/game/join":
            # One slot per phone; requests from this computer may take several (handy for testing in tabs).
            slot, token = PAD.join(None if ip in LOOPBACK_IPS else ip)
            self._json({"slot": slot, "token": token, "game": host.status()})
        elif path == "/api/game/input":
            PAD.set_keys(token, body.get("keys"))
            self._json({"ok": True})
        elif path == "/api/game/message":
            data = body.get("data")
            if len(json.dumps(data)) > MAX_MESSAGE:
                raise ValueError("message too large")
            PAD.put_message(token, data)
            self._json({"ok": True})
        elif path == "/api/game/state":
            self._json(host.private_state(token))
        else:
            PAD.leave(token)
            self._json({"ok": True})


def lan_urls(port: int) -> list[str]:
    """Addresses other devices on the network can use (no traffic is sent to find them)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            return [f"http://{s.getsockname()[0]}:{port}/"]
    except OSError:
        return []


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, host, addr: str, port: int, screen=None):
        super().__init__((addr, port), Handler)
        self.host, self.screen = host, screen
        self.limiter = RateLimiter()

    def handle_error(self, request, client_address):
        # A phone that drops off the Wi-Fi mid-request is routine, not worth a traceback.
        if isinstance(sys.exception(), (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def serve_in_thread(host, addr="127.0.0.1", port=8090, screen=None) -> Server:
    server = Server(host, addr, port, screen)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
