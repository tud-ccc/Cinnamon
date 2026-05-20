#!/usr/bin/env python3
"""Start a local HTTP server and open pool_viewer.html with the given CSV."""
import http.server
import pathlib
import sys
import threading
import webbrowser

here    = pathlib.Path(__file__).parent.resolve()
csv_abs = pathlib.Path(sys.argv[1]).resolve()

try:
    csv_rel = csv_abs.relative_to(here)
except ValueError:
    print(f"Error: {csv_abs} is not inside {here}", file=sys.stderr)
    sys.exit(1)

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(here), **kw)
    def log_message(self, *_):
        pass

port  = 8765
httpd = http.server.HTTPServer(("127.0.0.1", port), Handler)
threading.Thread(target=httpd.serve_forever, daemon=True).start()

url = f"http://127.0.0.1:{port}/pool_viewer.html?csv={csv_rel}"
print(f"Serving at {url}")
webbrowser.open(url)

try:
    input("Press Enter or Ctrl+C to stop.\n")
except KeyboardInterrupt:
    pass
httpd.shutdown()
