"""Zero-dependency HTTP bridge: SSE metrics + MJPEG video + REST control (CORS-open)."""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

try:
    import src.config as C
    import src.ui_controls as ui
except ImportError:
    import config as C
    import ui_controls as ui

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_INDEX = ROOT_DIR / "web" / "index.html"

ALLOWED_PARAMS = [
    "CONF", "MIN_HEAD_HEIGHT", "TRACK_MAX_DISPLACEMENT_SCALE", "KDE_GAMMA",
    "RISK_P_CRIT", "RISK_DENS_CRIT", "RISK_COUNT_FLOOR", "RISK_TURB_CRIT",
    "RISK_SIGMOID_K", "RISK_SIGMOID_MID", "SAVE_LIVE_VIDEO",
]


class Bridge:
    """Shared state between the pipeline and the web server."""

    def __init__(self):
        self.lock = threading.Lock()
        self.metrics = {}
        self.metrics_id = 0
        self.jpeg = None
        self.jpeg_t = 0.0
        self.pending_source = None

    def push_metrics(self, m):
        with self.lock:
            self.metrics = dict(m)
            self.metrics_id += 1

    def push_frame(self, canvas, min_interval=0.25):
        now = time.time()
        if now - self.jpeg_t < min_interval:
            return
        ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            with self.lock:
                self.jpeg = buf.tobytes()
                self.jpeg_t = now

    def snapshot(self):
        with self.lock:
            params = {k: getattr(C, k, None) for k in ALLOWED_PARAMS}
            metrics = dict(self.metrics)
        return {"metrics": metrics, "params": params, "ts": time.time()}


BRIDGE = Bridge()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # suppress per-request log spam

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/api/state":
            self._send(200, json.dumps(BRIDGE.snapshot()).encode())
        elif p == "/stream":
            self._sse()
        elif p == "/video":
            self._mjpeg()
        elif p in ("/", "/index.html") and WEB_INDEX.exists():
            self._send(200, WEB_INDEX.read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            body = {}
        p = self.path.split("?")[0]
        if p == "/api/params":
            applied = {}
            for k, v in body.items():
                if k in ALLOWED_PARAMS and isinstance(v, (int, float, bool)):
                    ui.set_param(k, v)
                    applied[k] = v
            self._send(200, json.dumps({"applied": applied}).encode())
        elif p == "/api/source":
            src = body.get("source")
            if src is None:
                self._send(400, b'{"error":"source required"}')
                return
            try:
                BRIDGE.pending_source = int(src)
            except (ValueError, TypeError):
                BRIDGE.pending_source = str(src)
            self._send(200, json.dumps({"queued": BRIDGE.pending_source}).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._cors()
        self.end_headers()
        last_id = None
        try:
            while True:
                with BRIDGE.lock:
                    m, mid = BRIDGE.metrics, BRIDGE.metrics_id
                if mid != last_id and m:
                    self.wfile.write(f"data: {json.dumps(m)}\n\n".encode())
                    self.wfile.flush()
                    last_id = mid
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self._cors()
        self.end_headers()
        last_t = 0.0
        try:
            while True:
                with BRIDGE.lock:
                    jpeg, jt = BRIDGE.jpeg, BRIDGE.jpeg_t
                if jpeg is not None and jt != last_t:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                    self.wfile.flush()
                    last_t = jt
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def start(port=None):
    """Start the web bridge on a background thread. Returns the server instance."""
    port = port or getattr(C, "WEB_PORT", 8000)
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[WebBridge] API + dashboard at http://localhost:{port}")
    return srv
