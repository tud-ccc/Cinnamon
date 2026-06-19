#!/usr/bin/env python3
"""Start a local HTTP server and open pool_viewer.html with the given CSV."""
import argparse
import http.server
import pathlib
import threading
import webbrowser

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--scale", default="log10",
                choices=["log10", "log2", "ln", "sqrt", "cbrt", "linear"])
args = ap.parse_args()

here    = pathlib.Path(__file__).parent.parent.resolve()
csv_abs = pathlib.Path(args.csv).resolve()

try:
    csv_rel = csv_abs.relative_to(here)
except ValueError:
    ap.error(f"{csv_abs} is not inside {here}")

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(here), **kw)
    def log_message(self, *_):
        pass
    def end_headers(self):
        # Cross-Origin Isolation: required for SharedArrayBuffer (parallel workers).
        self.send_header("Cross-Origin-Opener-Policy",   "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

port  = 8765
httpd = http.server.HTTPServer(("127.0.0.1", port), Handler)
threading.Thread(target=httpd.serve_forever, daemon=True).start()

url = f"http://127.0.0.1:{port}/viewer/pool_viewer.html?csv=/{csv_rel}&scale={args.scale}"
print(f"Serving at {url}")
webbrowser.open(url)

try:
    input("Press Enter or Ctrl+C to stop.\n")
except KeyboardInterrupt:
    pass
httpd.shutdown()
