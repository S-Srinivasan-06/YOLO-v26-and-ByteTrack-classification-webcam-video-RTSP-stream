"""CrowdShield Unified Web Backend & REST API Server.

Zero-dependency, multi-threaded HTTP + SSE + MJPEG server managing sessions,
multipart video uploads, dynamic schema queries, and run history.
"""
import email.parser
import email.policy
import json
import mimetypes
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import cv2

try:
    import src.config_schema as schema
    from src.session_manager import GLOBAL_SESSION, ROOT_DIR
except ImportError:
    import config_schema as schema
    from session_manager import GLOBAL_SESSION, ROOT_DIR

WEB_DIR = ROOT_DIR / "web"
UPLOAD_DIR = ROOT_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = ROOT_DIR / "data" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


class UnifiedHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress routine log spam for streaming endpoints
        if any(p in self.path for p in ("/stream", "/video")):
            return
        super().log_message(format, *args)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send_json(self, code: int, data: Dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path, filename: Optional[str] = None):
        if not file_path.exists():
            self._send_json(404, {"error": "File not found"})
            return
        mime_type, _ = mimetypes.guess_type(str(file_path))
        mime_type = mime_type or "application/octet-stream"
        file_size = file_path.stat().st_size

        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(file_size))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self._cors()
        self.end_headers()

        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                self.wfile.write(chunk)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "" or path == "/index.html":
            index_path = WEB_DIR / "index.html"
            self._send_file(index_path)
            return

        # ── API Endpoints ──
        if path == "/api/config/schema":
            self._send_json(200, {
                "schema": schema.PARAM_SCHEMA,
                "presets": schema.PRESETS
            })
            return

        if path == "/api/state":
            self._send_json(200, {
                "status": GLOBAL_SESSION.status,
                "run_id": GLOBAL_SESSION.run_id,
                "elapsed_sec": round(GLOBAL_SESSION.elapsed_time, 1),
                "remaining_sec": max(
                    0.0,
                    GLOBAL_SESSION.config["MAX_RUNTIME_SEC"] - GLOBAL_SESSION.elapsed_time
                ),
                "frames": GLOBAL_SESSION.frames_processed,
                "inferences": GLOBAL_SESSION.inferences_count,
                "metrics": GLOBAL_SESSION.latest_metrics,
                "config": GLOBAL_SESSION.config,
                "overlays": GLOBAL_SESSION.overlay_flags,
                "events": GLOBAL_SESSION.event_log[-10:],
            })
            return

        if path == "/api/runs":
            runs = []
            if RUNS_DIR.exists():
                for d in sorted(RUNS_DIR.iterdir(), reverse=True):
                    if d.is_dir():
                        s_file = d / "summary.json"
                        if s_file.exists():
                            try:
                                with open(s_file, "r", encoding="utf-8") as f:
                                    runs.append(json.load(f))
                            except Exception:
                                pass
            self._send_json(200, runs)
            return

        if path.startswith("/api/runs/") and "/download/" in path:
            parts = path.split("/")
            # e.g., /api/runs/run_20260820_213000/download/csv
            if len(parts) >= 6:
                run_id = parts[3]
                artifact = parts[5]
                target_dir = RUNS_DIR / run_id
                file_map = {
                    "csv": (target_dir / "log.csv", f"{run_id}_log.csv"),
                    "video": (target_dir / "output_dynamics.mp4", f"{run_id}_dynamics.mp4"),
                    "plot": (target_dir / "count_moving_average.png", f"{run_id}_plot.png"),
                    "config": (target_dir / "config.json", f"{run_id}_config.json"),
                    "summary": (target_dir / "summary.json", f"{run_id}_summary.json"),
                }
                if artifact in file_map:
                    fpath, fname = file_map[artifact]
                    self._send_file(fpath, fname)
                    return

        # ── Streaming Feeds ──
        if path == "/stream":
            self._handle_sse()
            return

        if path == "/video":
            self._handle_mjpeg()
            return

        # Static assets
        static_file = WEB_DIR / path.lstrip("/")
        if static_file.exists() and static_file.is_file():
            self._send_file(static_file)
            return

        self._send_json(404, {"error": "Not Found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        ctype = self.headers.get("Content-Type", "")
        content_len = int(self.headers.get("Content-Length", 0))

        # ── Multipart Video Upload ──
        if path == "/api/upload" and "multipart/form-data" in ctype:
            body = self.rfile.read(content_len)
            msg_bytes = f"Content-Type: {ctype}\r\nMIME-Version: 1.0\r\n\r\n".encode("latin-1") + body
            msg = email.parser.BytesParser(policy=email.policy.default).parsebytes(msg_bytes)

            uploaded_file = None
            filename = "video.mp4"
            for part in msg.iter_parts():
                fn = part.get_filename()
                if fn:
                    filename = fn
                    uploaded_file = part.get_payload(decode=True)
                    break

            if not uploaded_file:
                self._send_json(400, {"error": "No valid video file uploaded"})
                return

            dest_path = UPLOAD_DIR / filename
            with open(dest_path, "wb") as f:
                f.write(uploaded_file)

            # Probe video metadata using OpenCV
            cap = cv2.VideoCapture(str(dest_path))
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = round(frames / fps, 2) if fps > 0 else 0.0
            cap.release()

            self._send_json(200, {
                "status": "uploaded",
                "filename": filename,
                "filepath": str(dest_path),
                "size_mb": round(os.path.getsize(dest_path) / (1024 * 1024), 2),
                "resolution": f"{fw}x{fh}",
                "fps": round(fps, 1),
                "duration_sec": duration,
            })
            return

        # ── JSON Body Parsing ──
        body_json = {}
        if content_len > 0 and "application/json" in ctype:
            try:
                body_json = json.loads(self.rfile.read(content_len).decode("utf-8"))
            except Exception:
                body_json = {}

        if path == "/api/config":
            params = body_json.get("params", {})
            updated = GLOBAL_SESSION.update_config(params)
            self._send_json(200, {"status": "ok", "config": updated})
            return

        if path == "/api/overlays":
            overlays = body_json.get("overlays", body_json)
            GLOBAL_SESSION.overlay_flags.update(overlays)
            self._send_json(200, {"status": "ok", "overlays": GLOBAL_SESSION.overlay_flags})
            return

        if path == "/api/session/start":
            source = body_json.get("source", 0)
            source_type = body_json.get("source_type", "file")
            params = body_json.get("params")
            roi = body_json.get("roi_polygon")
            if roi:
                GLOBAL_SESSION.set_roi(roi)

            res = GLOBAL_SESSION.start_session(
                source=source,
                source_type=source_type,
                custom_params=params
            )
            if "error" in res:
                self._send_json(400, res)
            else:
                self._send_json(200, res)
            return

        if path == "/api/session/pause":
            GLOBAL_SESSION.pause_session()
            self._send_json(200, {"status": "paused"})
            return

        if path == "/api/session/resume":
            GLOBAL_SESSION.resume_session()
            self._send_json(200, {"status": "resumed"})
            return

        if path == "/api/session/stop":
            GLOBAL_SESSION.stop_session()
            self._send_json(200, {"status": "stopped"})
            return

        self._send_json(404, {"error": "Endpoint not found"})

    def _handle_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._cors()
        self.end_headers()

        try:
            while True:
                payload = {
                    "ts": time.time(),
                    "status": GLOBAL_SESSION.status,
                    "elapsed": round(GLOBAL_SESSION.elapsed_time, 1),
                    "remaining": max(0.0, GLOBAL_SESSION.config["MAX_RUNTIME_SEC"] - GLOBAL_SESSION.elapsed_time),
                    "metrics": GLOBAL_SESSION.latest_metrics,
                }
                msg = f"data: {json.dumps(payload)}\n\n".encode("utf-8")
                self.wfile.write(msg)
                self.wfile.flush()
                time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _handle_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self._cors()
        self.end_headers()

        last_jpeg = None
        try:
            while True:
                jpeg = GLOBAL_SESSION.latest_jpeg
                if jpeg is not None and jpeg != last_jpeg:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                    self.wfile.flush()
                    last_jpeg = jpeg
                time.sleep(0.033)  # ~30 FPS
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def start_server(port: int = 8000, host: str = "0.0.0.0"):
    """Starts the multi-threaded HTTP server on the given port."""
    server = ThreadingHTTPServer((host, port), UnifiedHandler)
    print(f"\n=======================================================")
    print(f"  CrowdShield Unified Web Platform")
    print(f"  Live Dashboard: http://localhost:{port}")
    print(f"=======================================================\n")
    return server
