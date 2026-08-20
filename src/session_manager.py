"""CrowdShield Unified Session & Run Lifecycle Manager."""
import csv
import json
import os
import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import src.config as C
    import src.config_schema as schema
    from src.features import (
        CrowdDynamicsEngine,
        FastCentroidTracker,
        PerspectiveGridEstimator,
        RollingMetrics,
        StampedeRiskClassifier,
        extract_features,
    )
    from src.plot_results import generate_plot
except ImportError:
    import config as C
    import config_schema as schema
    from features import (
        CrowdDynamicsEngine,
        FastCentroidTracker,
        PerspectiveGridEstimator,
        RollingMetrics,
        StampedeRiskClassifier,
        extract_features,
    )
    from plot_results import generate_plot

ROOT_DIR = Path(__file__).resolve().parent.parent


class SessionManager:
    """Manages full lifecycle of an active analysis run without OpenCV UI dependencies."""

    def __init__(self, model_path: str = C.HEAD_MODEL):
        self.model_path = model_path
        self.model: Optional[YOLO] = None
        self.lock = threading.RLock()

        # Session State
        self.status = "IDLE"  # IDLE, RUNNING, PAUSED, COMPLETED, ERROR
        self.run_id: Optional[str] = None
        self.run_dir: Optional[Path] = None
        self.source_desc: str = ""
        self.start_time: float = 0.0
        self.elapsed_time: float = 0.0
        self.frames_processed: int = 0
        self.inferences_count: int = 0

        # Dynamic Configurations & Overlays
        self.config = {k: v["default"] for k, v in schema.PARAM_SCHEMA.items()}
        self.roi_polygon: Optional[np.ndarray] = None  # Normalized [[x, y], ...]
        self.overlay_flags = {
            "boxes": True,
            "track_ids": True,
            "velocity": True,
            "hud": True,
            "density_map": False,
        }

        # Buffers & Streams
        self.latest_jpeg: Optional[bytes] = None
        self.latest_metrics: Dict = {
            "count": 0,
            "alert_name": "NORMAL",
            "risk_score_sma": 0.0,
        }
        self.event_log: List[Dict] = []
        self._prev_stage = "NORMAL"

        # Worker threads & queues
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    def load_model(self):
        if self.model is None:
            print(f"[Model] Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path, task="detect")

    def update_config(self, params: Dict) -> Dict:
        with self.lock:
            sanitized = schema.validate_and_sanitize(params)
            self.config.update(sanitized)
            # Sync back to global config namespace
            for k, v in sanitized.items():
                setattr(C, k, v)
            return self.config.copy()

    def set_roi(self, polygon_norm: Optional[List[List[float]]]):
        with self.lock:
            if polygon_norm and len(polygon_norm) >= 3:
                self.roi_polygon = np.array(polygon_norm, dtype=np.float32)
            else:
                self.roi_polygon = None

    def start_session(
        self, source: str, source_type: str = "file", custom_params: Optional[Dict] = None
    ) -> Dict:
        with self.lock:
            if self.status in ("RUNNING", "PAUSED"):
                return {"error": "A session is already active. Stop it first."}

            self.load_model()
            if custom_params:
                self.update_config(custom_params)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"run_{timestamp}"
            self.run_dir = ROOT_DIR / "data" / "runs" / self.run_id
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.source_desc = f"{source_type}:{source}"

            self._stop_event.clear()
            self._pause_event.clear()
            self.frames_processed = 0
            self.inferences_count = 0
            self.event_log.clear()
            self._prev_stage = "NORMAL"
            self.status = "RUNNING"

            self._worker_thread = threading.Thread(
                target=self._run_loop, args=(source, source_type), daemon=True
            )
            self._worker_thread.start()

            return {
                "status": "started",
                "run_id": self.run_id,
                "max_runtime": self.config["MAX_RUNTIME_SEC"],
            }

    def pause_session(self):
        with self.lock:
            if self.status == "RUNNING":
                self._pause_event.set()
                self.status = "PAUSED"

    def resume_session(self):
        with self.lock:
            if self.status == "PAUSED":
                self._pause_event.clear()
                self.status = "RUNNING"

    def stop_session(self):
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
        with self.lock:
            self.status = "COMPLETED"

    def _resolve_source_cap(self, source, source_type):
        if source_type == "webcam":
            try:
                idx = int(source)
            except (ValueError, TypeError):
                idx = 0
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
            return cap, 30.0, True

        if source_type == "youtube":
            try:
                import yt_dlp
                opts = {"format": "best[height<=720]/best", "quiet": True}
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(str(source), download=False)
                    stream_url = info.get("url")
                cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                return cap, 30.0, True
            except Exception as e:
                print(f"[YouTube] Error resolving stream: {e}")
                return None, 30.0, True

        if source_type == "rtsp":
            cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            return cap, 30.0, True

        # File
        cap = cv2.VideoCapture(str(source))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        return cap, fps, False

    def _run_loop(self, source, source_type):
        cap, fps, is_live = self._resolve_source_cap(source, source_type)
        if not cap or not cap.isOpened():
            self.status = "ERROR"
            print(f"[Session] Failed to open source: {source} ({source_type})")
            return

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

        # Artifact paths
        csv_path = self.run_dir / "log.csv"
        video_path = self.run_dir / "output_dynamics.mp4"
        summary_path = self.run_dir / "summary.json"
        plot_path = self.run_dir / "count_moving_average.png"
        config_path = self.run_dir / "config.json"

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_header_written = False

        writer = None
        if self.config.get("SAVE_LIVE_VIDEO", True):
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh)
            )

        # Processing Engines
        rolling = RollingMetrics()
        grid_est = PerspectiveGridEstimator()
        tracker = FastCentroidTracker(
            max_displacement_scale=self.config["TRACK_MAX_DISPLACEMENT_SCALE"]
        )
        dynamics = CrowdDynamicsEngine((fh, fw))
        risk_clf = StampedeRiskClassifier()

        self.start_time = time.time()
        next_infer_t = 0.0
        log_interval = getattr(C, "LOG_INTERVAL", 0.25)

        last_boxes = np.empty((0, 4), np.float32)
        last_track_ids = np.empty(0, np.int64)
        last_velocities = np.empty((0, 2), np.float32)

        try:
            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    time.sleep(0.05)
                    continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    if not is_live:
                        break
                    time.sleep(0.01)
                    continue

                self.frames_processed += 1
                cur_wall = time.time()
                self.elapsed_time = cur_wall - self.start_time

                # Check runtime limit
                if self.elapsed_time >= self.config["MAX_RUNTIME_SEC"]:
                    print(f"[Session] Reached max runtime of {self.config['MAX_RUNTIME_SEC']}s. Stopping.")
                    break

                video_t = (
                    self.elapsed_time
                    if is_live
                    else (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
                )

                # Inference Tick
                if video_t >= next_infer_t:
                    self.inferences_count += 1
                    res = self.model(
                        frame,
                        conf=self.config["CONF"],
                        iou=self.config["IOU"],
                        imgsz=self.config["IMGSZ"],
                        max_det=self.config["MAX_DET"],
                        device="cpu",
                        verbose=False,
                    )[0]

                    boxes = (
                        res.boxes.xyxy.cpu().numpy().astype(np.float32)
                        if res.boxes is not None
                        else np.empty((0, 4), np.float32)
                    )
                    confs = (
                        res.boxes.conf.cpu().numpy().astype(np.float32)
                        if res.boxes is not None
                        else np.empty(0, np.float32)
                    )

                    # ── ROI Polygon Gating ──
                    if self.roi_polygon is not None and len(boxes) > 0:
                        abs_poly = (self.roi_polygon * np.array([fw, fh])).astype(np.int32)
                        c_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
                        c_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
                        inside_mask = np.array(
                            [
                                cv2.pointPolygonTest(
                                    abs_poly, (float(x), float(y)), False
                                ) >= 0
                                for x, y in zip(c_x, c_y)
                            ]
                        )
                        boxes = boxes[inside_mask]
                        confs = confs[inside_mask]

                    (
                        metrics,
                        dens,
                        U,
                        V,
                        pres,
                        track_ids,
                        velocities,
                        valid_boxes,
                    ) = extract_features(
                        boxes,
                        confs,
                        fh,
                        fw,
                        rolling,
                        grid_est,
                        tracker,
                        dynamics,
                        video_t,
                    )

                    metrics["time_sec"] = round(video_t, 3)
                    metrics.update(risk_clf.evaluate(metrics))

                    # Event stage transitions
                    curr_stage = metrics.get("alert_name", "NORMAL")
                    if curr_stage != self._prev_stage:
                        self.event_log.append(
                            {
                                "time": round(video_t, 1),
                                "stage": curr_stage,
                                "score": metrics.get("risk_score_sma", 0.0),
                            }
                        )
                        self._prev_stage = curr_stage

                    # CSV flush
                    if not csv_header_written:
                        header = ["time_sec", "sys_time"] + [
                            k
                            for k in metrics
                            if k not in ("time_sec", "sys_time")
                        ]
                        csv_writer.writerow(header)
                        csv_header_written = True

                    csv_writer.writerow([metrics.get(k, "") for k in header])
                    csv_file.flush()

                    last_boxes, last_track_ids, last_velocities = (
                        valid_boxes,
                        track_ids,
                        velocities,
                    )
                    self.latest_metrics = metrics
                    next_infer_t = video_t + log_interval

                # Render Canvas for Web Stream & Video Recording
                canvas = frame.copy()

                # Draw ROI Polygon if configured
                if self.roi_polygon is not None:
                    abs_poly = (self.roi_polygon * np.array([fw, fh])).astype(np.int32)
                    cv2.polylines(canvas, [abs_poly], True, (255, 200, 0), 2, cv2.LINE_AA)

                # Draw Overlay Entities
                if self.overlay_flags.get("boxes", True):
                    for i, b in enumerate(last_boxes):
                        tid = int(last_track_ids[i]) if i < len(last_track_ids) else 0
                        clr = tracker._get_color(tid)
                        cv2.rectangle(
                            canvas,
                            (int(b[0]), int(b[1])),
                            (int(b[2]), int(b[3])),
                            clr,
                            2,
                        )
                        if self.overlay_flags.get("track_ids", True) and i < len(last_track_ids):
                            cv2.putText(
                                canvas,
                                f"#{tid}",
                                (int(b[0]), int(b[1]) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                clr,
                                1,
                            )
                        if self.overlay_flags.get("velocity", True) and i < len(last_velocities):
                            vx, vy = last_velocities[i]
                            if (vx * vx + vy * vy) > 36.0:
                                cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
                                cv2.arrowedLine(
                                    canvas,
                                    (cx, cy),
                                    (int(cx + vx * 0.18), int(cy + vy * 0.18)),
                                    clr,
                                    2,
                                    tipLength=0.35,
                                )

                if writer:
                    writer.write(canvas)

                # Encode JPEG for browser MJPEG feed
                ok_jpg, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ok_jpg:
                    self.latest_jpeg = buf.tobytes()

        finally:
            cap.release()
            if writer:
                writer.release()
            csv_file.close()

            # Auto-generate summary and publication graph
            try:
                generate_plot(str(csv_path), str(plot_path))
            except Exception as e:
                print(f"[Plot] Auto-plot error: {e}")

            summary = {
                "run_id": self.run_id,
                "source": self.source_desc,
                "elapsed_sec": round(self.elapsed_time, 2),
                "frames": self.frames_processed,
                "inferences": self.inferences_count,
                "peak_crowd": self.latest_metrics.get("count_sma", 0),
                "peak_density": self.latest_metrics.get("peak_density_sma", 0),
                "max_risk": self.latest_metrics.get("risk_score_sma", 0),
                "final_stage": self.latest_metrics.get("alert_name", "NORMAL"),
                "events": self.event_log,
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            self.status = "COMPLETED"


# Global singleton instance
GLOBAL_SESSION = SessionManager()
