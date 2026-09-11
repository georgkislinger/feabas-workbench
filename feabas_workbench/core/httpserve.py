"""A tiny CORS-enabled static file server so Neuroglancer can read a local precomputed volume."""

from __future__ import annotations

import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _Handler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt, *args) -> None:  # silence
        pass


class VolumeServer:
    def __init__(self, directory: Path, port: int = 0):
        handler = functools.partial(_Handler, directory=str(directory))
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
        self.port = self.httpd.server_address[1]
        self.directory = Path(directory)
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self) -> "VolumeServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def neuroglancer_url(self, rel: str = "", layer_name: str = "aligned") -> str:
        import json
        import urllib.parse
        src = f"precomputed://{self.url}/{rel}".rstrip("/")
        state = {"layers": [{"type": "image", "source": src, "name": layer_name}], "layout": "4panel"}
        return "https://neuroglancer-demo.appspot.com/#!" + urllib.parse.quote(json.dumps(state))
