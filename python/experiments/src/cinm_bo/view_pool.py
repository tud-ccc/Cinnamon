#!/usr/bin/env python3
"""Serve pool_viewer.html against a pool.csv and open it in a browser.

The page fetches the CSV over HTTP rather than reading it off disk, so this
needs a server. It serves two things from two places: the viewer itself, which
ships with this package, and the CSV, which is wherever the caller's data
happens to be. So the CSV's own directory is the document root and the viewer
is routed to explicitly -- the CSV no longer has to live under any particular
tree, which it did when the page and the data were both inside experiments/.
"""

from __future__ import annotations

import argparse
import http.server
import pathlib
import threading
import webbrowser

VIEWER_PATH = "/pool_viewer.html"
_VIEWER_FILE = pathlib.Path(__file__).resolve().parent / "pool_viewer.html"


def _handler_for(root: pathlib.Path):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def do_GET(self):  # noqa: N802  (http.server's spelling)
            if self.path.split("?")[0] == VIEWER_PATH:
                body = _VIEWER_FILE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def log_message(self, *_):
            pass

        def end_headers(self):
            # Cross-Origin Isolation: required for SharedArrayBuffer (the
            # page's parallel workers).
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            super().end_headers()

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="a pool.csv to load")
    ap.add_argument(
        "--scale",
        default="log10",
        choices=["log10", "log2", "ln", "sqrt", "cbrt", "linear"],
    )
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)

    csv = pathlib.Path(args.csv).resolve()
    if not csv.is_file():
        ap.error(f"{csv} does not exist")

    httpd = http.server.HTTPServer(("127.0.0.1", args.port), _handler_for(csv.parent))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    url = (
        f"http://127.0.0.1:{args.port}{VIEWER_PATH}?csv=/{csv.name}&scale={args.scale}"
    )
    print(f"Serving at {url}")
    webbrowser.open(url)

    try:
        input("Press Enter or Ctrl+C to stop.\n")
    except KeyboardInterrupt:
        pass
    httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
