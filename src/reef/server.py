"""A dependency-free HTTP server for a Reef artifact.

    POST /v1/systemone   {"state": ..., "questions": {...}}  ->  SystemOne-shaped response
    GET  /healthz        {"ok": true, "model": ..., "device": ...}
    GET  /v1/models      OpenAI-style model listing (for clients that discover the model name)

The request and response shapes follow the typed-decisions / SystemOne convention, so a Reef
model can stand in for a hosted decision endpoint on a LAN or a Raspberry Pi. One Edge TPU
serves one request at a time; ``Reef.run_window`` holds a lock around the interpreter.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from reef.packing import CapacityError

MAX_BODY = 16 << 20


def make_handler(reef):
    class Handler(BaseHTTPRequestHandler):
        server_version = "reef"

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _route(self) -> str:
            return self.path.split("?", 1)[0].rstrip("/") or "/"

        def do_GET(self):  # noqa: N802
            route = self._route()
            if route == "/healthz":
                self._send(200, {"ok": True, "model": reef.name, "device": reef.device})
            elif route == "/v1/models":
                # OpenAI-style listing so clients can discover the model name.
                self._send(200, {"object": "list", "data": [
                    {"id": reef.name, "object": "model", "owned_by": "reef"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self._route() != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                self._send(413 if n > MAX_BODY else 400, {"error": "bad Content-Length"})
                return
            try:
                body = json.loads(self.rfile.read(n))
                self._send(200, reef.systemone(body))
            except CapacityError as e:
                self._send(413, {"error": str(e)})
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001 - always answer, never kill the handler silently
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            pass

    return Handler


def serve(artifact: str, host: str = "127.0.0.1", port: int = 8080, device: str = "auto",
          max_state_windows: int = 8, strict: bool = False, threads: int | None = None) -> None:
    from reef.runtime import Reef

    reef = Reef(artifact, device=device, max_state_windows=max_state_windows, strict=strict, num_threads=threads)
    httpd = ThreadingHTTPServer((host, port), make_handler(reef))
    print(f"reef: serving {reef.name} on http://{host}:{port}  (device: {reef.device})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
