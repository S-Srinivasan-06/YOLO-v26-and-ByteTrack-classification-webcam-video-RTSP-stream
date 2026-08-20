"""CrowdShield L1 pipeline — interactive UI + web bridge + real-time live path.

Architecture:
  - Live sources  → InferenceWorker (background thread) + main-loop display at full FPS
  - File sources  → Synchronous frame-accurate processing with guaranteed output length
  - Both paths    → draw_hud + draw_param_panel + web bridge push every frame
  - Source switch → hotkey 'o' or POST /api/source triggers graceful session restart
"""
import csv
import os
import sys
import time
import math
import queue
import threading
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
from ultralytics import YOLO

import src.config as C
import src.ui_controls as ui
import src.web_server as web_server
from src.features import (
    RollingMetrics, PerspectiveGridEstimator, FastCentroidTracker,
    CrowdDynamicsEngine, StampedeRiskClassifier, extract_features,
)

# Configure FFmpeg for resilient live streaming
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "reconnect;1|reconnect_streamed;1|reconnect_delay_max;2|analyzeduration;5000000|probesize;5000000"
)

# Stage colors (BGR)
STAGE_COLORS = {
    "NORMAL":   (0, 200, 0),
    "WATCH":    (0, 220, 220),
    "WARNING":  (0, 150, 255),
    "CRITICAL": (0, 0, 255),
}


def fmt_k(val):
    """Format large numbers compactly: 33000 -> 33.0K"""
    if val >= 1000:
        return f"{val / 1000:.1f}K"
    return f"{val:.0f}"


def draw_hud(canvas, metrics, live=False):
    """Draw a compact live-metrics panel displaying moving averages.
    Position is read from ui.UI so the panel can be dragged."""
    stage = metrics.get("alert_name", "NORMAL")
    color = STAGE_COLORS.get(stage, (255, 255, 255))

    count_raw  = metrics.get("count", 0)
    count_sma  = metrics.get("count_sma", count_raw)
    risk_score = metrics.get("risk_score_sma", metrics.get("risk_score_5s", 0.0))
    peak_dens  = metrics.get("peak_density_sma", metrics.get("peak_density", 0.0))
    max_p      = metrics.get("max_pressure_sma", metrics.get("max_pressure", 0.0))
    turb       = metrics.get("turbulence_index_sma", metrics.get("turbulence_index", 0.0))
    coh        = metrics.get("flow_coherence_sma", metrics.get("flow_coherence", 0.0))
    danger     = metrics.get("danger_area_pct_sma", metrics.get("danger_area_pct", 0.0))

    # Panel geometry (draggable via ui.UI)
    x, y       = ui.UI["hud_pos"]
    w, line_h  = 280, 22
    header_h   = 34
    n_rows     = 7
    h          = header_h + n_rows * line_h + 12

    # Semi-transparent dark background
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)

    # Color-coded stage banner
    cv2.rectangle(canvas, (x, y), (x + w, y + header_h), color, -1)
    banner = f"{stage}  \xb7  {int(risk_score * 100)}% (SMA)"
    cv2.putText(canvas, banner, (x + 10, y + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

    # Metric rows (Moving Averages)
    tx = x + 12
    ty = y + header_h + 18
    rows_data = [
        ("COUNT (SMA)",     f"{count_sma:.1f} ({count_raw})"),
        ("PEAK DENS (SMA)", f"{peak_dens:.1f}"),
        ("PRESSURE (SMA)",  fmt_k(max_p)),
        ("TURBULENCE (SMA)",f"{turb:.1f}"),
        ("COHERENCE (SMA)", f"{coh:.2f}"),
        ("DANGER AREA (SMA)",f"{danger:.1f}%"),
        ("RISK SCORE (SMA)", f"{risk_score:.2f}"),
    ]
    for label, val in rows_data:
        cv2.putText(canvas, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(canvas, val, (x + w - 12 - tw, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        ty += line_h

    # LIVE / REC indicator
    if live:
        rec = getattr(C, "SAVE_LIVE_VIDEO", False)
        tag = "LIVE \u25cf REC" if rec else "LIVE \u25cb not recording"
        tag_color = (0, 0, 255) if rec else (200, 200, 200)
        cv2.putText(canvas, tag, (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, tag_color, 1, cv2.LINE_AA)

    # Record HUD rect for drag detection
    ui.UI["hud_rect"] = (x, y, w, h)
    return canvas


# ═══════════════ VIDEO WRITER ═══════════════

def _open_video_writer(path, fps, frame_size):
    """Try the preferred codec first, fall back to mp4v."""
    preferred = getattr(C, "VIDEO_CODEC_PREFERRED", "avc1")
    fallback = getattr(C, "VIDEO_CODEC_FALLBACK", "mp4v")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*preferred), fps, frame_size)
    if not writer.isOpened():
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fallback), fps, frame_size)
    return writer


class AsyncVideoWriter:
    """Writes frames to disk in a background thread.

    For FILE sources: write() blocks when the queue is full → no dropped frames.
    For LIVE streams: pass drop_on_full=True → drop-oldest so main loop never stalls.
    """

    def __init__(self, path, fps, frame_size, drop_on_full=False):
        self.drop_on_full = drop_on_full
        self.queue  = queue.Queue(maxsize=128)
        self.writer = _open_video_writer(path, fps, frame_size)
        self.running = True
        self.thread  = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            try:
                frame = self.queue.get(timeout=1.0)
                if frame is None:
                    break
                self.writer.write(frame)
                self.queue.task_done()
            except queue.Empty:
                if not self.running:
                    break
        self.writer.release()

    def write(self, frame):
        if self.drop_on_full:
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put_nowait(frame)
        else:
            self.queue.put(frame)

    def stop(self):
        self.running = False
        self.queue.put(None)
        self.thread.join(timeout=5.0)


# ═══════════════ THREADED CAPTURE (Stream + Webcam) ═══════════════

class ThreadedVideoCapture:
    """Background-thread video capture with reconnect and DirectShow webcam support."""

    def __init__(self, raw_source, timeout=6.0):
        self.raw_source = raw_source
        self.timeout = timeout
        self.is_device = isinstance(raw_source, int) or (isinstance(raw_source, str) and raw_source.isdigit())
        self.resolved_source = int(raw_source) if self.is_device else resolve_source(raw_source)
        self.cap = self._open_cap()
        if not self.is_device and isinstance(self.resolved_source, str) and \
           (self.resolved_source.startswith(("rtsp://", "rtsps://")) or "http" in self.resolved_source):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.grabbed, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.last_t = time.time()
        threading.Thread(target=self._loop, daemon=True).start()

    def _open_cap(self):
        if self.is_device:
            dev_idx = self.resolved_source
            cap = cv2.VideoCapture(dev_idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(dev_idx)
            return cap
        else:
            return cv2.VideoCapture(self.resolved_source, cv2.CAP_FFMPEG)

    def _loop(self):
        while self.running:
            try:
                if not self.cap.isOpened():
                    time.sleep(1.0)
                    if not self.running:
                        break
                    self._reconnect()
                    continue
                if self.cap.grab():
                    ret, fr = self.cap.retrieve()
                    if ret and fr is not None:
                        with self.lock:
                            self.frame = fr
                            self.grabbed = True
                            self.last_t = time.time()
                else:
                    if time.time() - self.last_t > self.timeout:
                        self._reconnect()
                    time.sleep(0.005)
            except Exception:
                time.sleep(0.5)
                self._reconnect()

    def _reconnect(self):
        if not self.running:
            return
        with self.lock:
            self.grabbed = False
        try:
            self.cap.release()
        except Exception:
            pass
        time.sleep(1.0)
        if not self.running:
            return
        try:
            if not self.is_device:
                self.resolved_source = resolve_source(self.raw_source)
            self.cap = self._open_cap()
            if not self.is_device and isinstance(self.resolved_source, str) and \
               (self.resolved_source.startswith(("rtsp://", "rtsps://")) or "http" in self.resolved_source):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.last_t = time.time()
        except Exception:
            pass

    def read(self):
        with self.lock:
            return self.grabbed, (
                self.frame.copy() if self.grabbed and self.frame is not None else None
            )

    def get(self, prop):
        return self.cap.get(prop)

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass


# ═══════════════ HELPERS ═══════════════

def resolve_source(src):
    """Resolve YouTube URL to direct streaming link using yt_dlp if applicable."""
    if isinstance(src, str) and ("youtube.com" in src or "youtu.be" in src):
        try:
            import yt_dlp
            ydl_opts = {
                "format": "best[height<=720]/best",
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(src, download=False)
                resolved = info.get("url")
                print(f"[YouTube Stream] Resolved: {info.get('title', 'Live')[:50]}")
                return resolved
        except Exception as e:
            print(f"[YouTube Stream] Warning: Failed to resolve ({e}). Using raw URL.")
    return src


def is_stream(src):
    return (
        isinstance(src, int)
        or str(src).startswith(("rtsp", "http", "rtmp", "https"))
        or "youtube" in str(src)
        or "youtu.be" in str(src)
    )


def _draw_overlay(canvas, boxes, track_ids, velocities, tracker):
    """Draw detection boxes and velocity arrows on the canvas."""
    n = min(len(boxes), len(track_ids), len(velocities))
    for i in range(n):
        x1, y1, x2, y2 = map(int, boxes[i])
        color = tracker._get_color(int(track_ids[i]))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        vx, vy = velocities[i]
        if math.hypot(vx, vy) > 8.0:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.arrowedLine(canvas, (cx, cy),
                            (int(cx + vx * 0.15), int(cy + vy * 0.15)),
                            color, 2, tipLength=0.4)
    return canvas


# ═══════════════ ASYNC INFERENCE WORKER ═══════════════

class InferenceWorker:
    """Runs YOLO detection + full feature pipeline on a background thread.
    Only the single most recent frame is ever pending — stale frames are dropped."""

    def __init__(self, model, rolling, grid_est, tracker, risk_clf):
        self.model    = model
        self.rolling  = rolling
        self.grid_est = grid_est
        self.tracker  = tracker
        self.risk_clf = risk_clf
        self.dynamics = None

        self._in_q      = queue.Queue(maxsize=1)
        self.csv_q      = queue.Queue()
        self._lock       = threading.Lock()
        self._boxes      = np.empty((0, 4), np.float32)
        self._track_ids  = np.empty(0, np.int64)
        self._velocities = np.empty((0, 2), np.float32)
        self._metrics    = {"count": 0, "alert_name": "NORMAL", "risk_score_5s": 0.0}
        self.last_infer_dt = 0.0

        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, frame, cur_t):
        """Non-blocking. Replaces any not-yet-processed pending frame."""
        if self._in_q.full():
            try:
                self._in_q.get_nowait()
            except queue.Empty:
                pass
        try:
            self._in_q.put_nowait((frame, cur_t))
        except queue.Full:
            pass

    def latest(self):
        with self._lock:
            return self._boxes, self._track_ids, self._velocities, self._metrics

    def _run(self):
        while self.running:
            try:
                frame, cur_t = self._in_q.get(timeout=0.5)
            except queue.Empty:
                continue

            t_start = time.time()
            fh, fw = frame.shape[:2]
            if self.dynamics is None:
                self.dynamics = CrowdDynamicsEngine(frame.shape)

            res = self.model(frame, conf=C.CONF, iou=C.IOU, imgsz=C.IMGSZ,
                              max_det=C.MAX_DET, device="cpu", verbose=False)[0]
            boxes = (res.boxes.xyxy.cpu().numpy().astype(np.float32)
                     if res.boxes is not None else np.empty((0, 4), np.float32))
            confs = (res.boxes.conf.cpu().numpy().astype(np.float32)
                     if res.boxes is not None else np.empty(0, np.float32))

            metrics, dens, U, V, pres, track_ids, velocities, valid_boxes = extract_features(
                boxes, confs, fh, fw, self.rolling, self.grid_est, self.tracker,
                self.dynamics, cur_t,
            )

            metrics["time_sec"] = round(cur_t, 3)
            metrics.update(self.risk_clf.evaluate(metrics))

            with self._lock:
                self._boxes      = valid_boxes
                self._track_ids  = track_ids
                self._velocities = velocities
                self._metrics    = metrics
            self.last_infer_dt = time.time() - t_start

            # Push to web bridge and CSV queue
            web_server.BRIDGE.push_metrics(metrics)
            self.csv_q.put(metrics)

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)


# ═══════════════ RUN SESSION ═══════════════

def run_session(model, source):
    """Execute one video/stream session. Returns 'quit' | 'switch' | 'ended'."""
    live = is_stream(source)
    cap = ThreadedVideoCapture(source) if live else cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[WARN] Cannot open {source}")
        return "ended"

    file_fps = cap.get(cv2.CAP_PROP_FPS) if not live else 0.0
    if file_fps <= 0:
        file_fps = 30.0

    rolling  = RollingMetrics()
    grid_est = PerspectiveGridEstimator()
    tracker  = FastCentroidTracker()
    risk_clf = StampedeRiskClassifier()

    save_video = bool(getattr(C, "SAVE_LIVE_VIDEO", False)) if live else True
    async_writer = None
    dynamics = None

    try:
        csv_f = open(C.LOG_CSV, "w", newline="", encoding="utf-8")
    except PermissionError:
        alt = C.LOG_CSV.replace(".csv", f"_{int(time.time())}.csv")
        print(f"[WARN] {C.LOG_CSV} is locked. Writing to {alt}")
        csv_f = open(alt, "w", newline="", encoding="utf-8")

    csv_w  = csv.writer(csv_f)
    header = None
    buf    = []
    last_flush = 0.0
    t0 = time.time()
    fidx = 0

    def flush_csv_row(metrics, cur_t):
        nonlocal header, last_flush
        if header is None:
            header = ["time_sec", "sys_time"] + [
                k for k in metrics if k not in ("time_sec", "sys_time")
            ]
            csv_w.writerow(header)
        buf.append([metrics.get(k, "") for k in header])
        if cur_t - last_flush >= C.CSV_FLUSH_SEC:
            csv_w.writerows(buf)
            buf.clear()
            csv_f.flush()
            last_flush = cur_t

    src_name = os.path.basename(str(source))
    print(f"Pipeline | source={src_name} | log={C.LOG_INTERVAL}s | "
          f"live={live} | save_video={save_video}")

    status = "ended"
    inferences = 0

    try:
        if live:
            # ── LIVE PATH: async inference, real-time display ──────────────
            worker = InferenceWorker(model, rolling, grid_est, tracker, risk_clf)
            next_submit = 0.0
            max_runtime = getattr(C, "MAX_RUNTIME_SEC", None)
            frame_times = []
            video_size = None

            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                fidx += 1
                now = time.time()
                cur_t = now - t0
                fh, fw = frame.shape[:2]
                ui.UI["frame_size"] = (fw, fh)

                frame_times.append(now)
                if len(frame_times) > 60:
                    frame_times.pop(0)

                if max_runtime and cur_t >= max_runtime:
                    print(f"Reached runtime limit of {max_runtime}s. Stopping.")
                    break

                # Submit frame to inference worker at LOG_INTERVAL cadence
                if cur_t >= next_submit:
                    worker.submit(frame, cur_t)
                    next_submit = cur_t + C.LOG_INTERVAL

                # Drain CSV rows from worker
                while True:
                    try:
                        m = worker.csv_q.get_nowait()
                    except queue.Empty:
                        break
                    inferences += 1
                    flush_csv_row(m, m.get("time_sec", cur_t))

                # Draw every frame with freshest available results
                boxes, track_ids, velocities, metrics = worker.latest()
                canvas = _draw_overlay(frame.copy(), boxes, track_ids, velocities, tracker)
                canvas = draw_hud(canvas, metrics, live=True)
                canvas = ui.draw_param_panel(canvas)
                web_server.BRIDGE.push_frame(canvas)

                if not C.HEADLESS:
                    cv2.imshow(ui.WINDOW, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        status = "quit"
                        break
                    if key == ord("o"):
                        web_server.BRIDGE.pending_source = ui.select_source(default=source)
                        status = "switch"
                        break

                # Check for web-triggered source switch
                if web_server.BRIDGE.pending_source is not None:
                    status = "switch"
                    break

                if save_video:
                    if video_size is None:
                        video_size = (fw, fh)
                    if async_writer is None and len(frame_times) >= 10:
                        span = frame_times[-1] - frame_times[0]
                        meas_fps = (len(frame_times) - 1) / span if span > 0 else 0
                        rec_fps = meas_fps if meas_fps > 1.0 else C.LIVE_VIDEO_FALLBACK_FPS
                        async_writer = AsyncVideoWriter(
                            C.OUTPUT_VIDEO, rec_fps, video_size, drop_on_full=True
                        )
                    if async_writer is not None:
                        async_writer.write(canvas)

            # Drain remaining metrics from worker
            worker.stop()
            while True:
                try:
                    m = worker.csv_q.get_nowait()
                except queue.Empty:
                    break
                inferences += 1
                flush_csv_row(m, m.get("time_sec", 0.0))

        else:
            # ── FILE PATH: synchronous, frame-accurate processing ─────────
            last_boxes      = np.empty((0, 4),  np.float32)
            last_track_ids  = np.empty(0,       np.int64)
            last_velocities = np.empty((0, 2),  np.float32)
            last_metrics    = {"count": 0, "alert_name": "NORMAL", "risk_score_5s": 0.0}
            next_log = 0.0
            max_runtime = getattr(C, "MAX_RUNTIME_SEC", None)

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                fidx += 1
                fh, fw = frame.shape[:2]
                ui.UI["frame_size"] = (fw, fh)

                if dynamics is None:
                    dynamics = CrowdDynamicsEngine(frame.shape)
                    async_writer = AsyncVideoWriter(
                        C.OUTPUT_VIDEO, file_fps, (fw, fh), drop_on_full=False
                    )

                cur_t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if max_runtime and cur_t >= max_runtime:
                    break

                if cur_t >= next_log:
                    inferences += 1
                    res = model(frame, conf=C.CONF, iou=C.IOU, imgsz=C.IMGSZ,
                                max_det=C.MAX_DET, device="cpu", verbose=False)[0]
                    raw_boxes = (res.boxes.xyxy.cpu().numpy().astype(np.float32)
                                 if res.boxes is not None else np.empty((0, 4), np.float32))
                    raw_confs = (res.boxes.conf.cpu().numpy().astype(np.float32)
                                 if res.boxes is not None else np.empty(0, np.float32))

                    metrics, dens, U, V, pres, track_ids, velocities, valid_boxes = extract_features(
                        raw_boxes, raw_confs, fh, fw, rolling, grid_est, tracker, dynamics, cur_t
                    )

                    metrics["time_sec"] = round(cur_t, 3)
                    metrics.update(risk_clf.evaluate(metrics))

                    last_boxes      = valid_boxes
                    last_track_ids  = track_ids
                    last_velocities = velocities
                    last_metrics    = metrics

                    flush_csv_row(metrics, cur_t)
                    web_server.BRIDGE.push_metrics(metrics)

                    while next_log <= cur_t:
                        next_log += C.LOG_INTERVAL

                canvas = _draw_overlay(frame.copy(), last_boxes, last_track_ids,
                                       last_velocities, tracker)
                canvas = draw_hud(canvas, last_metrics, live=False)
                canvas = ui.draw_param_panel(canvas)
                web_server.BRIDGE.push_frame(canvas)

                if not C.HEADLESS:
                    cv2.imshow(ui.WINDOW, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        status = "quit"
                        break
                    if key == ord("o"):
                        web_server.BRIDGE.pending_source = ui.select_source(default=source)
                        status = "switch"
                        break

                # Check for web-triggered source switch
                if web_server.BRIDGE.pending_source is not None:
                    status = "switch"
                    break

                if async_writer is not None:
                    async_writer.write(canvas)

    finally:
        elapsed_total = time.time() - t0
        if buf:
            csv_w.writerows(buf)
            csv_f.flush()
        cap.release()
        if async_writer:
            async_writer.stop()
        csv_f.close()
        cv2.destroyAllWindows()

        video_note = C.OUTPUT_VIDEO if save_video and async_writer else "(not recorded)"

        # Generate moving average plot
        plot_path = str(Path(C.LOG_CSV).parent / "count_moving_average.png")
        try:
            from src.plot_results import generate_plot
            generate_plot(C.LOG_CSV, plot_path)
        except Exception:
            plot_path = "(plot generation failed)"

        print("\n" + "=" * 55)
        print(f"Source File      : {src_name}")
        print(f"Saved Log CSV    : {C.LOG_CSV}")
        print(f"Saved Video      : {video_note}")
        print(f"Saved Plot Graph : {plot_path}")
        print(f"Time Elapsed     : {elapsed_total:.2f} s")
        print(f"Frames Evaluated : {fidx} frames ({inferences} inference ticks)")
        print("=" * 55 + "\n")

    return status


# ═══════════════ MAIN ═══════════════

def main():
    os.makedirs(os.path.dirname(os.path.abspath(C.LOG_CSV)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(C.OUTPUT_VIDEO)), exist_ok=True)

    print(f"Loading model: {C.HEAD_MODEL}")
    model = YOLO(C.HEAD_MODEL)

    # Start web bridge (SSE + MJPEG + REST)
    web_server.start()

    # Setup interactive OpenCV window
    if not C.HEADLESS:
        cv2.namedWindow(ui.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ui.WINDOW, 1280, 720)
        ui.attach()
        cv2.setMouseCallback(ui.WINDOW, ui._mouse)

    # Interactive source picker
    source = ui.select_source(default=C.SOURCE)

    while True:
        status = run_session(model, source)
        nxt = web_server.BRIDGE.pending_source
        web_server.BRIDGE.pending_source = None
        if nxt is not None and status != "quit":
            source = nxt
            print(f"\n[Source Switch] New source: {source}")
            continue
        break


if __name__ == "__main__":
    main()
