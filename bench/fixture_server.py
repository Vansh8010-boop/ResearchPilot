"""Local HTTP server for the adversarial fixture pages. Records hits on /collect (the planted exfil URL)."""

import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class _Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/collect"):
            self.server.collect_hits.append(self.path)
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, format, *args):
        pass


class FixtureServer:
    def __init__(self, port: int = 8765):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), partial(_Handler, directory=str(FIXTURE_DIR)))
        self.httpd.collect_hits = []
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def collect_hits(self) -> list[str]:
        return self.httpd.collect_hits

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
