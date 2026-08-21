# CrowdShield – Codebase Reference

## Unified Headless Architecture Overview

CrowdShield v0.3 is a **headless, unified, web-first research platform** for real-time crowd dynamics analysis and early stampede warning.

The browser dashboard (`http://localhost:8000`) serves as the **single source of truth** for all operations:
- Drag-and-drop video uploads with automatic OpenCV metadata probing (resolution, FPS, duration, size).
- Multi-source ingestion: Local video files, USB/DirectShow Webcams, low-latency RTSP/HTTP streams, and YouTube Live feeds (via `yt-dlp`).
- Interactive ROI (Region-of-Interest) polygon drawing directly on top of the live video canvas to restrict detection to specific physical zones.
- Full-spectrum parameter configuration with synchronized dual controls (sliders + numeric inputs) enabling precise tuning across continuous ranges (`CONF` 0.001 – 1.000, 15-minute default runtime timeout).
- Dynamic preset profiles (`High-Recall CCTV`, `Dense Crowd / Chokepoint`, `Webcam / Close-Range`, `High-Precision (Strict)`).
- Real-time Server-Sent Events (SSE) telemetry (10 Hz) & continuous low-latency MJPEG video streaming.
- Isolated per-run storage (`data/runs/{run_id}/`) recording structured time-series CSVs (44 features), annotated dynamics video MP4s, exact parameter JSON snapshots, summary metrics JSON, and publication-quality 3-panel matplotlib dynamics charts.

```text
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   UNIFIED WEB CLIENT                                   │
 │  ┌─────────────────────┐  ┌────────────────────────┐  ┌─────────────────────────────┐  │
 │  │ Source & Video Drop │  │ Live Canvas + ROI Draw │  │ Parameter Tuning & Presets  │  │
 │  │ (MP4 / RTSP / Cam)  │  │ (MJPEG + Canvas ROI)   │  │ (CONF 0.001-1.0, 15m timer) │  │
 │  └──────────┬──────────┘  └───────────▲────────────┘  └──────────────┬──────────────┘  │
 └─────────────┼─────────────────────────┼──────────────────────────────┼─────────────────┘
               │ POST /api/upload        │ GET /video (MJPEG)           │ POST /api/config
               │ POST /api/session/start │ GET /stream (SSE Metrics)    │ GET /api/runs
               ▼                         ▼                              ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                            UNIFIED HTTP REST & STREAM SERVER                           │
 │  • Multipart Upload Handler & Metadata Probe (OpenCV)                                  │
 │  • Real-time SSE Broadcaster & Event Trigger Engine                                    │
 │  • Parameter Schema Validator (Bounds, Types, Presets)                                 │
 └─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                           │ Dispatches state
                                           ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                             SESSION & LIFECYCLE MANAGER                                │
 │  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
 │  │ States: IDLE ──► INITIALIZING ──► RUNNING ◄──► PAUSED ──► STOPPING ──► COMPLETED │  │
 │  └──────────────────────────────────────────────────────────────────────────────────┘  │
 │  • Threaded Video Ingestion (Webcam, RTSP, YouTube-DLP, Local File)                    │
 │  • Async Inference Worker: YOLOv8 Head Detector (OpenVINO)                             │
 │  • ROI Filter: In-polygon point testing before Tracker ingestion                        │
 │  • Physics Engine: Fast Kinematic Tracker, Adaptive KDE, Helbing Pressure Field       │
 │  • Dual-Gated Stampede Classifier + Anomaly Detection                                 │
 │  • Per-Run Directory Manager (log.csv, video, plot, summary.json)                      │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical & Physical Models

### 1. Perspective Scale Field $S(y)$
- Decouples perspective geometry from noisy detections.
- Evaluates median head height across horizontal grid rows with exponential moving average (EMA):
  $$S_t(y) = (1 - \alpha) S_{t-1}(y) + \alpha \cdot \operatorname{median}(h_i)$$
- Ground-normalized density factor:
  $$\text{density\_score} = \sum_{i} \left(\frac{S(y_i)}{S(y_{\text{bottom}})}\right)^2$$

### 2. Fast Kinematic Velocity-Projected Centroid Tracker (with Multi-Frame Coasting)
- Uses constant-velocity projection to predict expected positions:
  $$\hat{\mathbf{c}}_{t} = \mathbf{c}_{t-\Delta t} + \mathbf{v}_{t-\Delta t} \cdot \Delta t$$
- Matches current detections against projected positions $\hat{\mathbf{c}}$ rather than stale static positions.
- **Multi-Frame Track Coasting**: Tracks survive momentary occlusions or low-confidence detector drops (up to 6 ticks / 1.5s) without losing their unique track ID.
- **Dynamic Search Cones**: Expands search radius for fast movers proportional to speed:
  $$d_{\text{max}} = \max\left(\text{scale} \cdot S(y), 0.7 \cdot \text{scale} \cdot S(y) + 0.5 \cdot \|\mathbf{v}_{prev}\| \cdot \Delta t\right)$$
- Applies exponential momentum smoothing on velocity vectors ($\alpha = 0.8$):
  $$\mathbf{v}_{t} = 0.8 \cdot \frac{\mathbf{c}_t - \mathbf{c}_{t-\Delta t}}{\Delta t} + 0.2 \cdot \mathbf{v}_{t-\Delta t}$$

### 3. Scale-Adaptive Gaussian KDE
- Continuous spatial crowd density field:
  $$\rho(\mathbf{x}) = \sum_i \frac{1}{2\pi \sigma_i^2} \exp\left(-\frac{\|\mathbf{x} - \mathbf{c}_i\|^2}{2\sigma_i^2}\right), \quad \sigma_i = \max\left(\frac{\gamma \cdot S(y_i)}{\text{scale\_down}}, 1.5\right)$$

### 4. Helbing Hydrodynamic Crowd Pressure
- Compressive mechanical force precursor to stampede:
  $$P(\mathbf{x}) = \rho(\mathbf{x}) \cdot \operatorname{Var}(\mathbf{v})(\mathbf{x})$$
  $$\operatorname{Var}(\mathbf{v}) = \frac{\sum w_i \|\mathbf{v}_i\|^2}{\sum w_i} - \left[\left(\frac{\sum w_i u_i}{\sum w_i}\right)^2 + \left(\frac{\sum w_i v_i}{\sum w_i}\right)^2\right]$$

### 5. Local Momentum & Flow Coherence
- Vector momentum field $\mathbf{M}(\mathbf{x}) = (M_x, M_y) = (\rho U, \rho V)$.
- Flow Coherence index $\Phi \in [0, 1]$:
  $$\Phi = \frac{\sqrt{\left(\sum M_x\right)^2 + \left(\sum M_y\right)^2}}{\sum \|\mathbf{M}\| + \epsilon}$$

### 6. Dual-Gated Moving-Average Stampede Risk Classifier
- Evaluated directly on smoothed moving-average features (`count_sma`, `peak_density_sma`, `max_pressure_sma`, `turbulence_index_sma`, `flow_coherence_sma`):
  $$s_{\text{score}} = \min\left(\frac{\text{peak\_density\_sma}}{\text{RISK\_DENS\_CRIT}}, 1.0\right)$$
  $$d_{\text{score}} = \min\left(\frac{\text{max\_pressure\_sma}}{\text{RISK\_P\_CRIT}}, 1.0\right) \cdot (1 - \text{flow\_coherence\_sma}) \cdot \min\left(1 + \frac{\text{turbulence\_index\_sma}}{\text{RISK\_TURB\_CRIT}}, 2.0\right)$$
  $$\text{raw\_risk} = 0.35 \cdot s_{\text{score}} + 0.65 \cdot d_{\text{score}}$$
  $$\text{risk\_score\_sma} = \frac{1}{1 + \exp\left(-k \cdot (\text{raw\_risk} - \text{mid})\right)}$$
- Alert Stage Mapping:
  - `0 (NORMAL)`: $\text{risk\_score} < 0.30$
  - `1 (WATCH)`: $0.30 \le \text{risk\_score} < 0.55$
  - `2 (WARNING)`: $0.55 \le \text{risk\_score} < 0.80$
  - `3 (CRITICAL)`: $\text{risk\_score} \ge 0.80$

---

## File Tree

```text
crowd_prototype/v0.3/
├── main.py                   (Unified headless launcher -> http://localhost:8000)
├── setup_model.py            (Automated model downloader & OpenVINO exporter)
├── requirements.txt          (Dependencies)
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py             (Central parameters, paths, defaults)
│   ├── config_schema.py      (Dynamic parameter schema, bounds, validation, presets)
│   ├── session_manager.py    (Unified SessionManager, state machine, per-run artifacts, ROI)
│   ├── server.py             (REST API + SSE metrics + MJPEG video stream + multipart upload)
│   ├── features.py           (Dynamics engine, KDE, centroid tracker, risk classifier)
│   ├── plot_results.py       (Automated 3-panel moving-average graph generation)
│   ├── collect.py            (Legacy dual-mode CLI/GUI collector)
│   ├── ui_controls.py        (Legacy OpenCV UI controls)
│   └── web_server.py         (Legacy thin REST bridge)
├── web/
│   └── index.html            (Full Tailwind research dashboard, ROI canvas, history drawer)
├── models/
│   ├── head_yolov8_openvino_model/   (OpenVINO FP16 optimized head detector)
│   └── head_yolov8.onnx              (ONNX export)
├── data/
│   ├── uploads/              (Uploaded video files via web interface)
│   ├── videos/               (Sample input footage e.g. easy.mp4, h1.mp4, h2.mp4)
│   ├── runs/                 (Isolated per-run artifact folders: run_YYYYMMDD_HHMMSS/)
│   │   └── run_20260820_214644/
│   │       ├── log.csv
│   │       ├── output_dynamics.mp4
│   │       ├── count_moving_average.png
│   │       ├── config.json
│   │       └── summary.json
│   └── outputs/              (Legacy single-run output directory)
└── codebase.md
```

---

## REST API Specification

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/config/schema` | Returns parameter schema (bounds, types, defaults) and preset profiles. |
| `GET` | `/api/state` | Returns live status (`IDLE`/`RUNNING`/`PAUSED`/`COMPLETED`), runtime timer, frame counters, metrics snapshot, active config, overlay flags, and recent stage event logs. |
| `POST` | `/api/config` | Validates and applies dynamic parameter updates on the fly. |
| `POST` | `/api/overlays` | Toggles rendering of bounding boxes, track IDs, and velocity arrows. |
| `POST` | `/api/upload` | Multipart form-data video upload with OpenCV resolution, FPS, duration, and size probing. |
| `POST` | `/api/session/start` | Starts an analysis session with specified source (file, webcam, RTSP, YouTube), parameters, and optional ROI polygon. |
| `POST` | `/api/session/pause` | Pauses video ingestion and inference loop. |
| `POST` | `/api/session/resume` | Resumes video ingestion and inference loop. |
| `POST` | `/api/session/stop` | Stops session cleanly, writes final CSV rows, saves output video, and auto-generates 3-panel plot and summary. |
| `GET` | `/stream` | Server-Sent Events (SSE) telemetry stream emitting real-time metrics at 10 Hz. |
| `GET` | `/video` | Multi-part JPEG (MJPEG) continuous video stream with active overlays and ROI visualizer. |
| `GET` | `/api/runs` | Returns array of all recorded historical runs with metadata and peak values. |
| `GET` | `/api/runs/{run_id}/download/{artifact}` | Downloads specific run artifact (`csv`, `video`, `plot`, `config`, `summary`). |

---

## Complete Source Code by File


---

### `requirements.txt`

```text
ultralytics>=8.4.0
opencv-python
numpy
openvino
onnxruntime
huggingface_hub
yt-dlp
matplotlib
fastapi>=0.100.0
uvicorn>=0.22.0
python-multipart>=0.0.6

```

---

### `.gitignore`

```gitignore
venv/
__pycache__/
*.pt
models/*.onnx
models/*_openvino_model/
data/outputs/
.vscode/
.idea/
*.log

```

---

### `setup_model.py`

```python
"""Download head-detection model and export to OpenVINO + ONNX."""
import os
import shutil
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


def setup():
    os.makedirs("models", exist_ok=True)
    target_vino = "models/head_yolov8_openvino_model"
    target_onnx = "models/head_yolov8.onnx"

    if os.path.exists(target_vino) and os.path.exists(target_onnx):
        print("Models already exist. Skipping.")
        return

    print("Downloading YOLOv8n head-detection weights...")
    try:
        pt_path = hf_hub_download(
            repo_id="keremberke/yolov8n-head-detection", filename="best.pt"
        )
    except Exception as e:
        print(f"Download failed: {e}")
        print("Manually place a head-detection .pt in models/ and export.")
        return

    model = YOLO(pt_path)

    if not os.path.exists(target_vino):
        print("Exporting to OpenVINO (imgsz=1280, half, dynamic)...")
        out = model.export(format="openvino", imgsz=1280, half=True, dynamic=True)
        if os.path.exists(out) and out != target_vino:
            if os.path.exists(target_vino):
                shutil.rmtree(target_vino)
            shutil.move(out, target_vino)

    if not os.path.exists(target_onnx):
        print("Exporting to ONNX (imgsz=1280, dynamic, simplified)...")
        out = model.export(format="onnx", imgsz=1280, dynamic=True, simplify=True, opset=12)
        if os.path.exists(out) and out != target_onnx:
            shutil.move(out, target_onnx)

    print("Done. Models ready in models/")


if __name__ == "__main__":
    setup()

```

---

### `main.py`

```python
"""CrowdShield L1 Vision Platform — Headless Unified Entrypoint.

Starts the full web platform on http://localhost:8000.
Browser manages video uploads, sources, ROI boundaries, live overlays, and parameter tuning.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src.config as C
from src.server import start_server


def main():
    port = getattr(C, "WEB_PORT", 8000)
    server = start_server(port=port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[CrowdShield] Server shutting down cleanly.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

```

---

### `src/__init__.py`

```python
"""CrowdShield L1 Vision Pipeline."""

```

---

### `src/config.py`

```python
"""Centralised pipeline configuration."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Input ──────────────────────────────────────────────
SOURCE = 0                         # Local Webcam (Device Index 0) or video path / stream URL
HEAD_MODEL = str(ROOT_DIR / "models" / "head_yolov8_openvino_model")

# ── Detection Sensitivity ──────────────────────────────
# Universal baseline confidence floor. Scale-Adaptive Confidence Gating in
# src/features.py dynamically raises this for close-up webcam heads (up to 0.50)
# while maintaining high recall (0.02) for distant crowd heads.
CONF = 0.02

IOU = 0.60                         # Prevents aggressive NMS suppression
IMGSZ = 1280                       # Full resolution inference
MAX_DET = 1500

# ── Timing & Logging ───────────────────────────────────
LOG_INTERVAL = 0.25                # 4 Hz sampling


LOG_CSV = str(ROOT_DIR / "data" / "outputs" / "log.csv")
CSV_FLUSH_SEC = 2.0
MAX_RUNTIME_SEC = 300.0            # Automatically stop after 5 minutes on live streams (None for infinite)

# ── Audit Video ────────────────────────────────────────
OUTPUT_VIDEO = str(ROOT_DIR / "data" / "outputs" / "output_dynamics.mp4")
VIDEO_CODEC_PREFERRED = "mp4v"     # Native, universal OpenCV mp4 codec
VIDEO_CODEC_FALLBACK = "mp4v"

# ── Live Stream Output Behaviour ───────────────────────
SAVE_LIVE_VIDEO = False
LIVE_VIDEO_FALLBACK_FPS = 25.0

# ── Perspective & Scales ───────────────────────────────
GRID_ROWS = 4
MIN_HEAD_HEIGHT = 2.0              # Lower floor for tiny/distant heads
ROLLING_WINDOW = 20

ROLLING_KEYS = [
    "count", "density_score", "peak_density", "danger_area_pct",
    "crowd_entropy", "max_pressure", "mean_momentum",
    "turbulence_index", "flow_coherence",
]

# ── Fast Kinematic Centroid Tracking ───────────────────
TRACK_MAX_DISPLACEMENT_SCALE = 4.0 # Generous search radius for running/fast crowd movement


# ── KDE / Dynamics ─────────────────────────────────────
KDE_SCALE_DOWN = 4
KDE_GAMMA = 0.8                    # Slightly broader Gaussian splat for tiny head signatures
DANGER_THRESH = 3.0
PRESSURE_NORM = 15.0

# ── Risk Classifier Thresholds ─────────────────────────
# Calibrated for high-completeness detection (~100-180 heads in dense view):
#   max_pressure p50 ~250k, p90 ~390k  -> P_CRIT = 450000 (saturates only during true violent surges)
#   peak_density p50 ~10,   p90 ~13    -> DENS_CRIT = 18.0 (crush threshold)
#   turbulence   p50 ~7k,   p90 ~23k   -> TURB_CRIT = 25000.0
#   sigmoid_k = 6.0 (smooth, non-saturating logistic gradient)
RISK_WINDOW_SEC  = 5.0
RISK_COUNT_FLOOR = 35
RISK_P_CRIT      = 450000.0
RISK_DENS_CRIT   = 18.0
RISK_TURB_CRIT   = 25000.0
RISK_SIGMOID_K   = 6.0
RISK_SIGMOID_MID = 0.45

# ── Runtime ────────────────────────────────────────────
HEADLESS = False
WEB_PORT = 8000
DISPLAY_SCALE = 100                # VIEW% trackbar (100 = native resolution)

```

---

### `src/config_schema.py`

```python
"""CrowdShield Parameter Schema, Dynamic Validation, and Presets."""
from typing import Any, Dict

PARAM_SCHEMA = {
    # ── Detection ──
    "CONF": {
        "type": "float",
        "min": 0.001,
        "max": 1.0,
        "default": 0.02,
        "group": "Detection",
        "desc": "Baseline detection confidence floor (0.001 - 1.000)",
    },
    "IOU": {
        "type": "float",
        "min": 0.01,
        "max": 1.0,
        "default": 0.60,
        "group": "Detection",
        "desc": "NMS Intersection-over-Union threshold",
    },
    "IMGSZ": {
        "type": "int",
        "allowed": [640, 768, 960, 1280, 1536, 1920],
        "default": 1280,
        "group": "Detection",
        "desc": "YOLO inference resolution",
    },
    "MAX_DET": {
        "type": "int",
        "min": 10,
        "max": 3000,
        "default": 1500,
        "group": "Detection",
        "desc": "Maximum allowed head detections per frame",
    },
    "MIN_HEAD_HEIGHT": {
        "type": "float",
        "min": 1.0,
        "max": 50.0,
        "default": 2.0,
        "group": "Detection",
        "desc": "Minimum head box pixel height",
    },
    "ENABLE_ADAPTIVE_CONF": {
        "type": "bool",
        "default": True,
        "group": "Detection",
        "desc": "Scale-adaptive confidence floor for close-up vs distant heads",
    },
    # ── Tracking ──
    "TRACK_MAX_DISPLACEMENT_SCALE": {
        "type": "float",
        "min": 0.5,
        "max": 10.0,
        "default": 4.0,
        "group": "Tracking",
        "desc": "Centroid search radius multiplier relative to head scale",
    },
    # ── Density & Physics ──
    "KDE_GAMMA": {
        "type": "float",
        "min": 0.1,
        "max": 3.0,
        "default": 0.8,
        "group": "Density & Physics",
        "desc": "Gaussian kernel standard deviation factor",
    },
    "KDE_SCALE_DOWN": {
        "type": "int",
        "allowed": [1, 2, 4, 8],
        "default": 4,
        "group": "Density & Physics",
        "desc": "Downscaling factor for continuous grid splatting",
    },
    "DANGER_THRESH": {
        "type": "float",
        "min": 0.5,
        "max": 10.0,
        "default": 3.0,
        "group": "Density & Physics",
        "desc": "Density threshold considered hazardous",
    },
    # ── Stampede Risk Classifier ──
    "RISK_COUNT_FLOOR": {
        "type": "int",
        "min": 0,
        "max": 500,
        "default": 35,
        "group": "Risk Model",
        "desc": "Minimum heads required to trigger Watch/Warning gates",
    },
    "RISK_P_CRIT": {
        "type": "float",
        "min": 10000.0,
        "max": 2000000.0,
        "default": 450000.0,
        "group": "Risk Model",
        "desc": "Helbing pressure saturation ceiling",
    },
    "RISK_DENS_CRIT": {
        "type": "float",
        "min": 1.0,
        "max": 50.0,
        "default": 18.0,
        "group": "Risk Model",
        "desc": "Continuous density crush limit",
    },
    "RISK_TURB_CRIT": {
        "type": "float",
        "min": 1000.0,
        "max": 100000.0,
        "default": 25000.0,
        "group": "Risk Model",
        "desc": "Momentum turbulence saturation factor",
    },
    "RISK_SIGMOID_K": {
        "type": "float",
        "min": 1.0,
        "max": 20.0,
        "default": 6.0,
        "group": "Risk Model",
        "desc": "Logistic curve transition steepness",
    },
    "RISK_SIGMOID_MID": {
        "type": "float",
        "min": 0.1,
        "max": 0.9,
        "default": 0.45,
        "group": "Risk Model",
        "desc": "Logistic curve midpoint",
    },
    # ── Runtime & Recording ──
    "MAX_RUNTIME_SEC": {
        "type": "float",
        "min": 30.0,
        "max": 864000.0,
        "default": 86400.0,  # Unlimited / 24h default
        "group": "Runtime",
        "desc": "Session auto-termination timeout in seconds",
    },
    "SAVE_LIVE_VIDEO": {
        "type": "bool",
        "default": True,
        "group": "Runtime",
        "desc": "Record annotated video artifact",
    },
}

PRESETS = {
    "High-Recall CCTV": {
        "CONF": 0.02,
        "IOU": 0.60,
        "IMGSZ": 1280,
        "MIN_HEAD_HEIGHT": 2.0,
        "TRACK_MAX_DISPLACEMENT_SCALE": 4.0,
        "KDE_GAMMA": 0.8,
        "RISK_COUNT_FLOOR": 35,
    },
    "Dense Crowd / Chokepoint": {
        "CONF": 0.05,
        "IOU": 0.50,
        "IMGSZ": 1280,
        "MIN_HEAD_HEIGHT": 2.0,
        "TRACK_MAX_DISPLACEMENT_SCALE": 5.5,
        "KDE_GAMMA": 1.0,
        "RISK_COUNT_FLOOR": 50,
        "RISK_P_CRIT": 400000.0,
    },
    "Webcam / Close-Range": {
        "CONF": 0.35,
        "IOU": 0.65,
        "IMGSZ": 640,
        "MIN_HEAD_HEIGHT": 15.0,
        "TRACK_MAX_DISPLACEMENT_SCALE": 3.0,
        "KDE_GAMMA": 0.6,
        "RISK_COUNT_FLOOR": 1,
    },
    "High-Precision (Strict)": {
        "CONF": 0.50,
        "IOU": 0.70,
        "IMGSZ": 1280,
        "MIN_HEAD_HEIGHT": 5.0,
        "TRACK_MAX_DISPLACEMENT_SCALE": 3.5,
        "KDE_GAMMA": 0.8,
    },
}


def validate_and_sanitize(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Validate incoming parameter updates against the schema."""
    cleaned = {}
    for k, val in updates.items():
        if k not in PARAM_SCHEMA:
            continue
        spec = PARAM_SCHEMA[k]
        try:
            if spec["type"] == "float":
                v = float(val)
                if "min" in spec and v < spec["min"]:
                    continue
                if "max" in spec and v > spec["max"]:
                    continue
                cleaned[k] = v
            elif spec["type"] == "int":
                v = int(val)
                if "allowed" in spec and v not in spec["allowed"]:
                    continue
                if "min" in spec and v < spec["min"]:
                    continue
                if "max" in spec and v > spec["max"]:
                    continue
                cleaned[k] = v
            elif spec["type"] == "bool":
                cleaned[k] = bool(val)
        except (ValueError, TypeError):
            continue
    return cleaned

```

---

### `src/session_manager.py`

```python
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

```

---

### `src/server.py`

```python
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

```

---

### `src/features.py`

```python
"""CrowdShield feature extraction — KLT-free high-speed build with Track IDs."""
import math
import time
import random
from collections import defaultdict, deque

import cv2
import numpy as np

try:
    import src.config as C
except ImportError:
    import config as C

EPS = 1e-6


# ═══════════════════════════════════════════════════════
# 1. ROLLING METRICS (scoped to ROLLING_KEYS only)
# ═══════════════════════════════════════════════════════
class RollingMetrics:
    def __init__(self, window_size=C.ROLLING_WINDOW, keys=C.ROLLING_KEYS):
        self.window = window_size
        self.keys = set(keys)
        self.history = defaultdict(lambda: deque(maxlen=window_size))

    def update(self, raw):
        out = {}
        for k, v in raw.items():
            out[k] = v
            if k not in self.keys or not isinstance(v, (int, float)) or math.isnan(v):
                continue
            h = self.history[k]
            h.append(v)
            if len(h) >= 2:
                sma = sum(h) / len(h)
                var = sum((x - sma) ** 2 for x in h) / (len(h) - 1)
                std = math.sqrt(var)
                out[k + "_sma"] = round(sma, 4)
                out[k + "_std"] = round(std, 4)
                out[k + "_z"] = round((v - sma) / (std + EPS), 4)
            else:
                out[k + "_sma"] = round(v, 4)
                out[k + "_std"] = 0.0
                out[k + "_z"] = 0.0
        return out


# ═══════════════════════════════════════════════════════
# 2. PERSPECTIVE SCALE FIELD S(y)
# ═══════════════════════════════════════════════════════
class PerspectiveGridEstimator:
    def __init__(self, rows=C.GRID_ROWS, alpha=0.1):
        self.rows = rows
        self.alpha = alpha
        self.baseline = np.linspace(
            C.MIN_HEAD_HEIGHT * 1.5, C.MIN_HEAD_HEIGHT * 4.0, rows
        )

    def update_and_get(self, heights, centers_y, frame_h):
        bins = np.linspace(0, frame_h, self.rows + 1)
        for i in range(self.rows):
            mask = (centers_y >= bins[i]) & (centers_y < bins[i + 1])
            if np.any(mask):
                obs = float(np.median(heights[mask]))
                self.baseline[i] = (1 - self.alpha) * self.baseline[i] + self.alpha * obs
        return np.maximum(self.baseline, C.MIN_HEAD_HEIGHT)


# ═══════════════════════════════════════════════════════
# 3. FAST KINEMATIC CENTROID TRACKER (with Multi-Frame Coasting)
# ═══════════════════════════════════════════════════════
class Track:
    """Individual track state with velocity prediction and occlusion memory."""
    def __init__(self, track_id, center, color, current_time):
        self.track_id = track_id
        self.center = center.astype(np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.color = color
        self.last_time = current_time
        self.misses = 0
        self.hits = 1

    def predict(self, dt):
        return self.center + self.velocity * dt


class FastCentroidTracker:
    """Scale-normalized bipartite centroid matching with Track Coasting.

    Key Features:
      1. Kinematic Lookahead: Projects track positions forward using prior velocity.
      2. Multi-Frame Coasting (max_misses = 6 ticks ~1.5s): Survives temporary head occlusions,
         lighting shifts, and detection dropouts in dense crowds without losing track identity.
      3. Velocity Smoothing: 80% instantaneous, 20% momentum for stable physical vectors.
      4. Dynamic Search Cones: Expands search radius for fast movers proportional to speed.
    """

    def __init__(self, max_displacement_scale=C.TRACK_MAX_DISPLACEMENT_SCALE, max_misses=6):
        self.tracks = {}  # track_id -> Track
        self.max_scale = max_displacement_scale
        self.max_misses = max_misses
        self.next_id = 0
        self.prev_time = None
        self.colors = {}

    def _get_color(self, track_id):
        if track_id not in self.colors:
            self.colors[track_id] = tuple(random.randint(80, 255) for _ in range(3))
        return self.colors[track_id]

    def update(self, centers, scales, current_time):
        # Re-read from config so trackbar / web slider changes take effect live
        self.max_scale = getattr(C, "TRACK_MAX_DISPLACEMENT_SCALE", self.max_scale)
        n = len(centers)

        if self.prev_time is None:
            self.prev_time = current_time
            ids = np.empty(n, dtype=np.int64)
            velocities = np.zeros((n, 2), dtype=np.float32)
            for i in range(n):
                tid = self.next_id
                self.next_id += 1
                color = self._get_color(tid)
                self.tracks[tid] = Track(tid, centers[i], color, current_time)
                ids[i] = tid
            return velocities, ids

        dt = current_time - self.prev_time
        self.prev_time = current_time

        if dt <= 0.001 or dt > 2.0:
            dt = 0.25

        active_track_ids = list(self.tracks.keys())
        m = len(active_track_ids)

        if m == 0:
            ids = np.empty(n, dtype=np.int64)
            velocities = np.zeros((n, 2), dtype=np.float32)
            for i in range(n):
                tid = self.next_id
                self.next_id += 1
                color = self._get_color(tid)
                self.tracks[tid] = Track(tid, centers[i], color, current_time)
                ids[i] = tid
            return velocities, ids

        if n == 0:
            # Coast all existing tracks forward
            for tid in active_track_ids:
                trk = self.tracks[tid]
                trk.center += trk.velocity * dt
                trk.misses += 1
                if trk.misses > self.max_misses:
                    del self.tracks[tid]
            return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int64)

        # ── Predict positions for all active tracks ──
        predicted_centers = np.array([self.tracks[tid].predict(dt) for tid in active_track_ids], dtype=np.float32)
        prev_speeds = np.array([np.linalg.norm(self.tracks[tid].velocity) for tid in active_track_ids], dtype=np.float32)

        # Pairwise distance matrix: N_detections × M_tracks
        diff = centers[:, np.newaxis, :] - predicted_centers[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)

        # Adaptive search threshold per detection-track pair
        base_allowed = scales[:, np.newaxis] * self.max_scale
        speed_cushion = prev_speeds[np.newaxis, :] * dt * 0.5
        max_allowed_dist = np.maximum(base_allowed, base_allowed * 0.7 + speed_cushion)

        matched_cur = set()
        matched_trk = set()
        det_ids = np.full(n, -1, dtype=np.int64)
        det_velocities = np.zeros((n, 2), dtype=np.float32)

        # Greedy bipartite matching
        sorted_indices = np.dstack(
            np.unravel_index(np.argsort(dist_matrix.ravel()), dist_matrix.shape)
        )[0]

        for cur_idx, trk_idx in sorted_indices:
            if cur_idx in matched_cur or trk_idx in matched_trk:
                continue
            if dist_matrix[cur_idx, trk_idx] <= max_allowed_dist[cur_idx, trk_idx]:
                matched_cur.add(cur_idx)
                matched_trk.add(trk_idx)
                
                tid = active_track_ids[trk_idx]
                trk = self.tracks[tid]
                
                # Displacement from actual prior location
                inst_v = (centers[cur_idx] - trk.center) / (dt * (trk.misses + 1))
                
                # Velocity smoothing
                trk.velocity = 0.8 * inst_v + 0.2 * trk.velocity
                trk.center = centers[cur_idx].astype(np.float32)
                trk.misses = 0
                trk.hits += 1
                trk.last_time = current_time
                
                det_ids[cur_idx] = tid
                det_velocities[cur_idx] = trk.velocity

        # Coast unmatched tracks
        for trk_idx, tid in enumerate(active_track_ids):
            if trk_idx not in matched_trk:
                trk = self.tracks[tid]
                trk.center += trk.velocity * dt
                trk.misses += 1
                if trk.misses > self.max_misses:
                    del self.tracks[tid]

        # Spawn new tracks for unmatched detections
        for i in range(n):
            if det_ids[i] == -1:
                tid = self.next_id
                self.next_id += 1
                color = self._get_color(tid)
                self.tracks[tid] = Track(tid, centers[i], color, current_time)
                det_ids[i] = tid
                det_velocities[i] = np.zeros(2, dtype=np.float32)

        # Prune colors
        if len(self.colors) > 3000:
            active_ids = set(self.tracks.keys())
            self.colors = {k: v for k, v in self.colors.items() if k in active_ids}

        return det_velocities, det_ids




# ═══════════════════════════════════════════════════════
# 4. SCALE-ADAPTIVE KDE (cached kernels)
# ═══════════════════════════════════════════════════════
class ScaleAdaptiveKDE:
    def __init__(self, frame_shape, scale_down=C.KDE_SCALE_DOWN, gamma=C.KDE_GAMMA):
        self.oh, self.ow = frame_shape[:2]
        self.sd = scale_down
        self.h = self.oh // scale_down
        self.w = self.ow // scale_down
        self.gamma = gamma
        self._kcache = {}

    def _kernel(self, sigma):
        key = max(round(sigma * 2.0) / 2.0, 1.0)
        hit = self._kcache.get(key)
        if hit is None:
            r = int(np.ceil(3.0 * key))
            ax = np.arange(-r, r + 1, dtype=np.float32)
            g = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2.0 * key * key))
            hit = (r, g.astype(np.float32))
            self._kcache[key] = hit
        return hit

    def splat(self, density, vx, vy, v2, centers, scales, velocities):
        # Re-read from config so trackbar / web slider changes take effect live
        self.gamma = getattr(C, "KDE_GAMMA", self.gamma)
        sc = centers / self.sd
        sig = np.maximum(scales * self.gamma / self.sd, 1.5)
        W, H = self.w, self.h
        for i in range(len(sc)):
            r, g = self._kernel(float(sig[i]))
            xi, yi = int(round(sc[i][0])), int(round(sc[i][1]))
            x0, y0 = xi - r, yi - r
            x1, y1 = xi + r + 1, yi + r + 1
            sx0, sy0 = max(0, -x0), max(0, -y0)
            ex, ey = min(W, x1) - x0, min(H, y1) - y0
            if ex <= sx0 or ey <= sy0:
                continue
            patch = g[sy0:ey, sx0:ex]
            ph, pw = patch.shape
            X, Y = max(0, x0), max(0, y0)
            density[Y:Y + ph, X:X + pw] += patch
            u, v = velocities[i]
            vx[Y:Y + ph, X:X + pw] += patch * u
            vy[Y:Y + ph, X:X + pw] += patch * v
            v2[Y:Y + ph, X:X + pw] += patch * (u * u + v * v)


# ═══════════════════════════════════════════════════════
# 5. CROWD DYNAMICS ENGINE
# ═══════════════════════════════════════════════════════
class CrowdDynamicsEngine:
    def __init__(self, frame_shape):
        self.kde = ScaleAdaptiveKDE(frame_shape)
        self.oh, self.ow = frame_shape[:2]

    def compute(self, centers, scales, velocities):
        h, w = self.kde.h, self.kde.w
        density = np.zeros((h, w), np.float32)
        if len(centers) == 0:
            z = density.copy()
            return density, z, z, z, z, z, z
        vx = np.zeros((h, w), np.float32)
        vy = np.zeros((h, w), np.float32)
        v2 = np.zeros((h, w), np.float32)
        self.kde.splat(density, vx, vy, v2, centers, scales, velocities)
        U = vx / (density + EPS)
        V = vy / (density + EPS)
        var_v = np.maximum(v2 / (density + EPS) - (U * U + V * V), 0.0)
        Mx = density * U
        My = density * V
        momentum = density * np.sqrt(U * U + V * V)
        pressure = density * var_v
        return density, U, V, Mx, My, momentum, pressure

    @staticmethod
    def metrics(density, Mx, My, momentum, pressure):
        peak = float(np.max(density)) if density.size else 0.0
        total = float(np.sum(density))
        danger_px = float(np.sum(density >= C.DANGER_THRESH))
        danger_pct = (danger_px / density.size * 100.0) if density.size else 0.0
        if total > 0:
            prob = density.ravel() / total
            prob = prob[prob > 0]
            entropy = float(-np.sum(prob * np.log2(prob + EPS)))
        else:
            entropy = 0.0
        max_p = float(np.max(pressure)) if pressure.size else 0.0
        mean_m = float(np.mean(momentum)) if momentum.size else 0.0
        turb = float(np.sum(pressure) / (total + EPS))
        m_total = float(np.sum(momentum))
        if m_total > EPS:
            net_mx, net_my = float(np.sum(Mx)), float(np.sum(My))
            coherence = min(math.sqrt(net_mx ** 2 + net_my ** 2) / (m_total + EPS), 1.0)
        else:
            coherence = 0.0
        return {
            "peak_density":    round(peak, 3),
            "danger_area_pct": round(danger_pct, 2),
            "crowd_entropy":   round(entropy, 3),
            "max_pressure":    round(max_p, 3),
            "mean_momentum":   round(mean_m, 4),
            "turbulence_index": round(turb, 4),
            "flow_coherence":  round(coherence, 4),
        }


# ═══════════════════════════════════════════════════════
# 6. DUAL-GATED 5s ALERT EVALUATOR
# ═══════════════════════════════════════════════════════
class StampedeRiskClassifier:
    """5-second smoothed, physics-gated stampede hazard classifier.

    Two mandatory gates must both pass before any score is emitted:
      1. Absolute count gate  — crowd too sparse → always NORMAL (no false alarms)
      2. Physics gate         — pressure AND density both negligible → NORMAL

    Risk score components:
      s_score = spatial density / RISK_DENS_CRIT          (how compressed)
      d_score = (pressure / RISK_P_CRIT)
                × (1 - flow_coherence)                    (chaotic momentum)
                × (1 + turbulence / RISK_TURB_CRIT)       (amplifier)

    Final score passed through a logistic (sigmoid) centred at 0.45
    so that the transition from WATCH→WARNING is steep and hysteresis-free.
    """

    def __init__(self):
        steps = max(1, int(C.RISK_WINDOW_SEC / C.LOG_INTERVAL))
        self.hist_count = deque(maxlen=steps)
        self.hist_p     = deque(maxlen=steps)
        self.hist_dens  = deque(maxlen=steps)
        self.hist_turb  = deque(maxlen=steps)
        self.hist_coh   = deque(maxlen=steps)

    def evaluate(self, metrics: dict) -> dict:
        # Use rolling SMA features directly (falling back to instantaneous if startup)
        count_sma = float(metrics.get("count_sma", metrics.get("count", 0)))
        p_sma     = float(metrics.get("max_pressure_sma", metrics.get("max_pressure", 0.0)))
        dens_sma  = float(metrics.get("peak_density_sma", metrics.get("peak_density", 0.0)))
        turb_sma  = float(metrics.get("turbulence_index_sma", metrics.get("turbulence_index", 0.0)))
        coh_sma   = float(metrics.get("flow_coherence_sma", metrics.get("flow_coherence", 0.0)))

        base = {"p_sma5s": round(p_sma, 1), "dens_sma5s": round(dens_sma, 2)}

        # ── Gate 1: minimum crowd size (evaluated on moving average count) ──
        if count_sma < C.RISK_COUNT_FLOOR:
            return {
                "risk_score_5s": 0.0,
                "risk_score_sma": 0.0,
                "alert_stage": 0,
                "alert_name": "NORMAL",
                **base,
            }

        # ── Gate 2: negligible physics signal ───────────────────────────
        if p_sma < C.RISK_P_CRIT * 0.2 and dens_sma < 3.0:
            return {
                "risk_score_5s": 0.0,
                "risk_score_sma": 0.0,
                "alert_stage": 0,
                "alert_name": "NORMAL",
                **base,
            }

        # ── Dimensionless risk scoring calculated on moving averages ────
        s_score = min(dens_sma / C.RISK_DENS_CRIT, 1.0)
        d_score = (
            min(p_sma / C.RISK_P_CRIT, 1.0)
            * (1.0 - coh_sma)
            * min(1.0 + turb_sma / C.RISK_TURB_CRIT, 2.0)
        )
        raw_risk = 0.35 * s_score + 0.65 * d_score

        # Logistic sigmoid centered at RISK_SIGMOID_MID
        mid = getattr(C, "RISK_SIGMOID_MID", 0.45)
        risk_score_sma = float(1.0 / (1.0 + math.exp(-C.RISK_SIGMOID_K * (raw_risk - mid))))

        if   risk_score_sma >= 0.80: stage, name = 3, "CRITICAL"
        elif risk_score_sma >= 0.55: stage, name = 2, "WARNING"
        elif risk_score_sma >= 0.30: stage, name = 1, "WATCH"
        else:                        stage, name = 0, "NORMAL"

        return {
            "risk_score_5s":  round(risk_score_sma, 4),
            "risk_score_sma": round(risk_score_sma, 4),
            "alert_stage":    stage,
            "alert_name":     name,
            **base,
        }



# ═══════════════════════════════════════════════════════
# 7. MASTER FEATURE EXTRACTION (Scale-Adaptive Gating)
# ═══════════════════════════════════════════════════════
def extract_features(boxes, confs, frame_h, frame_w, rolling, grid_est, tracker, dynamics, cur_t):
    sys_time = time.time()

    def empty():
        z = np.zeros((dynamics.kde.h, dynamics.kde.w), np.float32)
        raw = {
            "sys_time": sys_time, "count": 0, "density_score": 0.0,
            "peak_density": 0.0, "danger_area_pct": 0.0, "crowd_entropy": 0.0,
            "max_pressure": 0.0, "mean_momentum": 0.0,
            "turbulence_index": 0.0, "flow_coherence": 0.0,
        }
        return (rolling.update(raw), z, z.copy(), z.copy(), z.copy(),
                np.empty(0, np.int64), np.empty((0, 2), np.float32), np.empty((0, 4), np.float32))

    if len(boxes) == 0:
        tracker.update(np.empty((0, 2), np.float32), np.empty(0, np.float32), cur_t)
        return empty()

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    aspect_ratios = widths / (heights + EPS)

    # ── Scale-Adaptive Confidence Gating ────────────────────────────────
    # Tiny distant heads (h <= 10px in livestreams/CCTV) use high-recall threshold (CONF ~ 0.02)
    # Large close-up heads (h >= 100px in webcam/room views) require high confidence (0.22 - 0.50)
    # This automatically eliminates phantom furniture/background detections on webcams.
    h_norm = np.clip((heights - 10.0) / 90.0, 0.0, 1.0)
    adaptive_min_conf = np.maximum(C.CONF, 0.02 + 0.20 * h_norm)

    valid = (
        (heights >= C.MIN_HEAD_HEIGHT) &
        (aspect_ratios >= 0.40) & (aspect_ratios <= 2.30) &
        (confs >= adaptive_min_conf)
    )
    boxes = boxes[valid]
    heights = heights[valid]


    count = len(boxes)
    if count == 0:
        tracker.update(np.empty((0, 2), np.float32), np.empty(0, np.float32), cur_t)
        return empty()

    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    centers = np.column_stack((cx, cy))

    row_h = grid_est.update_and_get(heights, cy, frame_h)
    grid_y = np.clip((cy / frame_h * C.GRID_ROWS).astype(int), 0, C.GRID_ROWS - 1)
    scales = row_h[grid_y]

    density_score = float(np.sum((scales / row_h[-1]) ** 2))
    velocities, track_ids = tracker.update(centers, scales, cur_t)

    dens, U, V, Mx, My, mom, pres = dynamics.compute(centers, scales, velocities)
    dyn = dynamics.metrics(dens, Mx, My, mom, pres)

    raw = {
        "sys_time": sys_time,
        "count": count,
        "density_score": round(density_score, 3),
        **dyn,
    }
    return rolling.update(raw), dens, U, V, pres, track_ids, velocities, boxes


```

---

### `src/plot_results.py`

```python
"""Generates publication-quality time-series graphs of crowd dynamics and moving averages."""
import os
import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / script execution
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_CSV = str(ROOT_DIR / "data" / "outputs" / "log.csv")
OUTPUT_PLOT = str(ROOT_DIR / "data" / "outputs" / "count_moving_average.png")


def generate_plot(csv_path=LOG_CSV, output_png=OUTPUT_PLOT):
    if not os.path.exists(csv_path):
        print(f"[Plot] Error: Log file not found: {csv_path}")
        return False

    with open(csv_path, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("[Plot] Warning: CSV is empty.")
        return False

    time_sec = [float(r.get("time_sec", i * 0.25)) for i, r in enumerate(rows)]
    count_raw = [float(r.get("count", 0)) for r in rows]
    count_sma = [float(r.get("count_sma", r.get("count", 0))) for r in rows]
    risk_sma  = [float(r.get("risk_score_sma", r.get("risk_score_5s", 0.0))) for r in rows]
    p_sma     = [float(r.get("max_pressure_sma", r.get("max_pressure", 0.0))) for r in rows]
    dens_sma  = [float(r.get("peak_density_sma", r.get("peak_density", 0.0))) for r in rows]
    alerts    = [r.get("alert_name", "NORMAL") for r in rows]

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, dpi=150)
    fig.patch.set_facecolor("#0F172A")

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#1E293B")
        ax.tick_params(colors="#94A3B8", labelsize=10)
        ax.grid(True, linestyle="--", alpha=0.25, color="#64748B")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # ── Subplot 1: Crowd Count & Moving Average ───────────────────────
    ax1.plot(time_sec, count_raw, color="#38BDF8", alpha=0.35, linewidth=1.0, linestyle=":", label="Instantaneous Count")
    ax1.plot(time_sec, count_sma, color="#0284C7", linewidth=2.4, label="Count (Moving Average)")
    ax1.fill_between(time_sec, count_raw, count_sma, color="#38BDF8", alpha=0.10)

    # Annotate peak and median
    max_idx = int(np.argmax(count_sma))
    ax1.scatter([time_sec[max_idx]], [count_sma[max_idx]], color="#F43F5E", s=40, zorder=5)
    ax1.annotate(f"Peak SMA: {count_sma[max_idx]:.1f}",
                 xy=(time_sec[max_idx], count_sma[max_idx]),
                 xytext=(time_sec[max_idx] + 1.0, count_sma[max_idx] + 3.0),
                 color="#FDA4AF", fontsize=9, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#F43F5E", lw=1.2))

    ax1.set_ylabel("Crowd Count (Heads)", color="#E2E8F0", fontsize=11, fontweight="bold")
    ax1.set_title("Crowd Dynamics — Moving Average Analysis & Risk Forecasting",
                  color="#F8FAFC", fontsize=14, fontweight="bold", pad=12)
    ax1.legend(loc="upper right", facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0")

    # ── Subplot 2: Stampede Risk Score (Moving Average) ───────────────
    # Risk zones
    ax2.axhspan(0.00, 0.30, color="#10B981", alpha=0.12, label="NORMAL (<30%)")
    ax2.axhspan(0.30, 0.55, color="#FBBF24", alpha=0.12, label="WATCH (30-55%)")
    ax2.axhspan(0.55, 0.80, color="#FB923C", alpha=0.12, label="WARNING (55-80%)")
    ax2.axhspan(0.80, 1.00, color="#EF4444", alpha=0.15, label="CRITICAL (>80%)")

    ax2.plot(time_sec, risk_sma, color="#F59E0B", linewidth=2.6, label="Risk Score (Moving Average)")
    ax2.set_ylabel("Stampede Risk (SMA)", color="#E2E8F0", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(loc="upper right", facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0", ncol=2)

    # ── Subplot 3: Pressure & Density Moving Averages ─────────────────
    line1 = ax3.plot(time_sec, [p / 1000.0 for p in p_sma], color="#A855F7", linewidth=2.0, label="Pressure (SMA, k)")
    ax3.set_ylabel("Crowd Pressure (k)", color="#C084FC", fontsize=11, fontweight="bold")

    ax3_twin = ax3.twinx()
    ax3_twin.tick_params(colors="#94A3B8", labelsize=10)
    for spine in ax3_twin.spines.values():
        spine.set_color("#334155")
    line2 = ax3_twin.plot(time_sec, dens_sma, color="#F472B6", linewidth=2.0, linestyle="--", label="Peak Density (SMA)")
    ax3_twin.set_ylabel("Peak Density (SMA)", color="#F472B6", fontsize=11, fontweight="bold")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="upper right", facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0")

    ax3.set_xlabel("Elapsed Time (seconds)", color="#E2E8F0", fontsize=11, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, facecolor=fig.get_facecolor(), dpi=150)
    plt.close()
    print(f"[Plot] Graph successfully generated and saved -> {output_png}")
    return True


# Alias for seamless compatibility
plot_dynamics = generate_plot


if __name__ == "__main__":
    generate_plot()


```

---

### `src/ui_controls.py`

```python
"""Interactive OpenCV controls: live trackbars, draggable panels, source menu."""
import threading
import cv2

try:
    import src.config as C
except ImportError:
    import config as C

WINDOW = "CrowdShield Live"
_LOCK = threading.Lock()

UI = {
    "hud_pos": [12, 12],
    "param_pos": [12, 260],
    "hud_rect": (0, 0, 0, 0),
    "param_rect": (0, 0, 0, 0),
    "drag": None,
    "frame_size": (1280, 720),
}


def set_param(name, value):
    with _LOCK:
        setattr(C, name, value)


def get_param(name):
    with _LOCK:
        return getattr(C, name, None)


# (label, config attr, max_raw, to_value, to_raw)
_TRACKBARS = [
    ("CONF%",     "CONF",                         50,  lambda r: r / 100.0, lambda v: int(v * 100)),
    ("MINHEAD",   "MIN_HEAD_HEIGHT",              20,  float,               int),
    ("TRACKx10",  "TRACK_MAX_DISPLACEMENT_SCALE", 100, lambda r: r / 10.0,  lambda v: int(v * 10)),
    ("GAMMAx10",  "KDE_GAMMA",                    20,  lambda r: r / 10.0,  lambda v: int(v * 10)),
    ("PCRIT/10k", "RISK_P_CRIT",                  100, lambda r: r * 10000.0, lambda v: int(v / 10000)),
    ("DENSCRIT",  "RISK_DENS_CRIT",               40,  float,               int),
    ("COUNTFLR",  "RISK_COUNT_FLOOR",             200, float,               int),
]


def _view_cb(raw):
    set_param("DISPLAY_SCALE", float(raw))
    fw, fh = UI["frame_size"]
    try:
        cv2.resizeWindow(WINDOW, max(320, int(fw * raw / 100)), max(240, int(fh * raw / 100)))
    except Exception:
        pass


def _cb(attr, to_val):
    def fn(raw):
        set_param(attr, to_val(raw))
    return fn


def attach(window=WINDOW):
    """Create all trackbars on the named window."""
    for label, attr, max_raw, to_val, to_raw in _TRACKBARS:
        cur = get_param(attr)
        init = int(max(0, min(max_raw, to_raw(cur if cur is not None else 0))))
        cv2.createTrackbar(label, window, init, max_raw, _cb(attr, to_val))
    cv2.createTrackbar("VIEW%", window, 100, 100, _view_cb)


def _mouse(event, x, y, flags, _):
    """Mouse callback for dragging the HUD and param panels."""
    if event == cv2.EVENT_LBUTTONDOWN:
        for key in ("hud", "param"):
            rx, ry, rw, rh = UI[key + "_rect"]
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                UI["drag"] = (key, x - rx, y - ry)
                return
    elif event == cv2.EVENT_MOUSEMOVE and UI["drag"]:
        key, dx, dy = UI["drag"]
        UI[key + "_pos"] = [max(0, x - dx), max(0, y - dy)]
    elif event == cv2.EVENT_LBUTTONUP:
        UI["drag"] = None


def draw_param_panel(canvas):
    """Read-only live parameter readout (drag with mouse; tune via trackbars/web)."""
    x, y = UI["param_pos"]
    w, line_h = 240, 20
    rows = [
        ("CONF",        f"{get_param('CONF') or 0:.2f}"),
        ("MIN_HEAD",    f"{get_param('MIN_HEAD_HEIGHT') or 0:.0f}px"),
        ("TRACK_SCL",   f"{get_param('TRACK_MAX_DISPLACEMENT_SCALE') or 0:.1f}"),
        ("KDE_GAMMA",   f"{get_param('KDE_GAMMA') or 0:.1f}"),
        ("P_CRIT",      f"{(get_param('RISK_P_CRIT') or 0)/1000:.0f}k"),
        ("DENS_CRIT",   f"{get_param('RISK_DENS_CRIT') or 0:.0f}"),
        ("COUNT_FLR",   f"{get_param('RISK_COUNT_FLOOR') or 0:.0f}"),
    ]
    h = 26 + len(rows) * line_h + 8
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (120, 120, 120), 1)
    cv2.putText(canvas, "PARAMS (drag me)", (x + 8, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    ty = y + 26 + 12
    for label, val in rows:
        cv2.putText(canvas, label, (x + 10, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(canvas, val, (x + w - 10 - tw, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        ty += line_h
    UI["param_rect"] = (x, y, w, h)
    return canvas


def select_source(default=None):
    """Terminal-based interactive source picker."""
    try:
        print("\n=== SOURCE MENU ===  [1] Webcam  [2] RTSP/HTTP  [3] YouTube  [4] File  [5] Config default")
        ch = (input("Select [1-5]: ").strip() or "5")
    except EOFError:
        return default
    if ch == "1":
        return int(input("Webcam index [0]: ").strip() or "0")
    if ch == "2":
        return input("RTSP/HTTP URL: ").strip()
    if ch == "3":
        return input("YouTube URL: ").strip()
    if ch == "4":
        return input("Video file path: ").strip()
    return default

```

---

### `src/web_server.py`

```python
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

```

---

### `src/collect.py`

```python
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

```

---

### `web/index.html`

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CROWDSHIELD — Real-Time Vision Platform</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Raleway:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
  <style>
    * {
      font-family: 'Raleway', sans-serif !important;
      border-radius: 0px !important;
    }
    :root {
      --bg-black: #000000;
      --bg-panel: #111111;
      --bg-panel-alt: #1a1a1a;
      --border-white: #ffffff;
      --border-gray: #333333;
      --border-muted: #555555;
    }
    body {
      background-color: var(--bg-black);
      color: #ffffff;
    }
    .neo-box {
      background-color: var(--bg-panel);
      border: 2px solid var(--border-white);
      box-shadow: 4px 4px 0px #ffffff;
    }
    .neo-box-gray {
      background-color: var(--bg-panel);
      border: 2px solid var(--border-gray);
      box-shadow: 3px 3px 0px #333333;
    }
    .neo-btn {
      background-color: #ffffff;
      color: #000000;
      border: 2px solid #ffffff;
      font-weight: 800;
      text-transform: uppercase;
      box-shadow: 3px 3px 0px #888888;
      transition: transform 0.05s ease, box-shadow 0.05s ease;
    }
    .neo-btn:hover {
      background-color: #e5e5e5;
      box-shadow: 2px 2px 0px #888888;
      transform: translate(1px, 1px);
    }
    .neo-btn:active {
      box-shadow: 0px 0px 0px #888888;
      transform: translate(3px, 3px);
    }
    .neo-btn-dark {
      background-color: #1a1a1a;
      color: #ffffff;
      border: 2px solid var(--border-gray);
      font-weight: 700;
      text-transform: uppercase;
      box-shadow: 3px 3px 0px #333333;
      transition: transform 0.05s ease, box-shadow 0.05s ease;
    }
    .neo-btn-dark:hover {
      background-color: #262626;
      border-color: #ffffff;
      box-shadow: 2px 2px 0px #ffffff;
      transform: translate(1px, 1px);
    }
    .neo-input {
      background-color: #000000;
      border: 2px solid var(--border-gray);
      color: #ffffff;
    }
    .neo-input:focus {
      border-color: #ffffff;
      outline: none;
    }
    /* Neobrutalist Range Slider */
    input[type=range] {
      -webkit-appearance: none;
      background: #222222;
      height: 8px;
      border: 1px solid var(--border-muted);
    }
    input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none;
      height: 18px;
      width: 18px;
      background: #ffffff;
      border: 2px solid #000000;
      cursor: pointer;
    }
    input[type=range]::-moz-range-thumb {
      height: 18px;
      width: 18px;
      background: #ffffff;
      border: 2px solid #000000;
      cursor: pointer;
    }
    /* Strictly 2-color Risk Stages */
    .st-NORMAL { background: #000000; color: #ffffff; border: 2px solid #ffffff; }
    .st-WATCH { background: #262626; color: #ffffff; border: 2px solid #ffffff; }
    .st-WARNING { background: #525252; color: #ffffff; border: 2px solid #ffffff; }
    .st-CRITICAL { background: #ffffff; color: #000000; border: 2px solid #000000; font-weight: 900; }

    /* Custom Checkbox */
    input[type=checkbox] {
      appearance: none;
      width: 16px;
      height: 16px;
      border: 2px solid #ffffff;
      background: #000000;
      cursor: pointer;
      display: inline-grid;
      place-content: center;
    }
    input[type=checkbox]:checked {
      background: #ffffff;
    }
    input[type=checkbox]:checked::before {
      content: "";
      width: 8px;
      height: 8px;
      background: #000000;
    }
  </style>
</head>
<body class="p-4 md:p-6 min-h-screen overflow-y-auto flex flex-col gap-6">

  <!-- Top Navigation Header -->
  <header class="flex flex-wrap items-center justify-between gap-4 pb-4 border-b-2 border-white">
    <div class="flex items-center gap-4">
      <h1 class="text-2xl md:text-3xl font-black tracking-widest text-white uppercase">CROWDSHIELD</h1>
      <span id="badge_status" class="px-3 py-1 text-xs font-black uppercase tracking-wider border-2 border-white bg-black text-white">IDLE</span>
      <span id="timer_display" class="text-xs font-bold uppercase tracking-wider text-neutral-400">ELAPSED: 00:00</span>
    </div>
    
    <div class="flex items-center gap-3">
      <button id="btn_history" class="neo-btn-dark px-4 py-2 text-xs">History</button>
      <button id="btn_pause" class="neo-btn-dark px-4 py-2 text-xs hidden">Pause</button>
      <button id="btn_start" class="neo-btn px-5 py-2 text-xs">Start Analysis</button>
      <button id="btn_stop" class="neo-btn-dark px-4 py-2 text-xs border-white hidden">Stop</button>
    </div>
  </header>

  <!-- Main Grid Layout (Naturally Scrollable) -->
  <main class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
    
    <!-- Left Column: Source Selection, Video Canvas, Overlays Toolbar -->
    <section class="lg:col-span-7 flex flex-col gap-4">
      
      <!-- Source Ingestion Panel -->
      <div class="neo-box-gray p-4 flex flex-col gap-3">
        <div class="text-xs font-black uppercase tracking-wider text-neutral-400 border-b border-neutral-700 pb-2">
          Input Source
        </div>
        
        <div class="grid grid-cols-4 gap-2 text-xs font-bold uppercase">
          <button id="tab_file" class="p-2 border-2 border-white bg-white text-black text-center">Video File</button>
          <button id="tab_webcam" class="p-2 border-2 border-neutral-700 bg-black text-neutral-400 hover:border-neutral-400 text-center">Webcam</button>
          <button id="tab_rtsp" class="p-2 border-2 border-neutral-700 bg-black text-neutral-400 hover:border-neutral-400 text-center">RTSP</button>
          <button id="tab_yt" class="p-2 border-2 border-neutral-700 bg-black text-neutral-400 hover:border-neutral-400 text-center">YouTube</button>
        </div>

        <!-- File Upload Area -->
        <div id="src_file_area" class="flex flex-col gap-2">
          <div id="dropzone" class="border-2 border-dashed border-neutral-600 hover:border-white p-4 text-center cursor-pointer bg-black transition">
            <span id="file_label" class="text-xs font-bold uppercase tracking-wider text-neutral-300">
              Drag & Drop Video (MP4 / WebM) or <span class="text-white underline">Browse</span>
            </span>
            <input type="file" id="file_input" class="hidden" accept="video/*">
          </div>
          <div id="file_probe" class="text-xs font-mono text-neutral-400 uppercase hidden"></div>
        </div>

        <!-- Input Box for Webcam/RTSP/YouTube -->
        <div id="src_input_area" class="hidden flex gap-2">
          <input type="text" id="src_val" class="flex-1 neo-input px-3 py-2 text-xs font-mono" placeholder="Device index (0) or Stream URL">
        </div>
      </div>

      <!-- Live Video Viewport Container -->
      <div class="relative bg-black border-2 border-white w-full aspect-video flex items-center justify-center overflow-hidden">
        <img id="video_feed" class="w-full h-full object-contain" alt="Live Feed Viewport">
        <canvas id="roi_canvas" class="absolute inset-0 w-full h-full cursor-crosshair"></canvas>
        
        <!-- Live Hazard Banner (Strict 2-Color Neobrutalism) -->
        <div id="risk_banner" class="absolute top-3 left-3 px-4 py-1.5 text-xs font-black uppercase tracking-wider st-NORMAL">
          NORMAL · 0% (SMA)
        </div>
      </div>

      <!-- Decoupled Overlays & ROI Toolbar (Eliminates Glitching & Event Collisions) -->
      <div class="neo-box-gray p-3 flex flex-wrap items-center justify-between gap-3 text-xs uppercase font-bold">
        <div class="flex items-center gap-4">
          <span class="text-neutral-400">Overlays:</span>
          <label class="flex items-center gap-1.5 cursor-pointer">
            <input type="checkbox" id="chk_boxes" checked>
            <span>Boxes</span>
          </label>
          <label class="flex items-center gap-1.5 cursor-pointer">
            <input type="checkbox" id="chk_ids" checked>
            <span>IDs</span>
          </label>
          <label class="flex items-center gap-1.5 cursor-pointer">
            <input type="checkbox" id="chk_vel" checked>
            <span>Vectors</span>
          </label>
        </div>
        
        <div class="flex items-center gap-3">
          <span id="roi_status" class="text-[11px] text-neutral-400 lowercase">click canvas to add ROI polygon points</span>
          <button id="btn_clear_roi" class="px-2.5 py-1 bg-black border border-white text-white hover:bg-neutral-800 text-[11px] uppercase font-bold">
            Clear ROI
          </button>
        </div>
      </div>

    </section>

    <!-- Right Column: Real-Time Metrics & Detailed Parameter Controls -->
    <section class="lg:col-span-5 flex flex-col gap-4">
      
      <!-- Metrics Dashboard Panel -->
      <div class="neo-box p-4 flex flex-col gap-3">
        <div class="text-xs font-black uppercase tracking-wider text-white border-b-2 border-white pb-2 flex justify-between">
          <span>Telemetry Dynamics</span>
          <span class="text-neutral-400">SMA Smoothing</span>
        </div>
        
        <div class="grid grid-cols-3 gap-2 text-center">
          <div class="bg-black border border-neutral-700 p-2">
            <div class="text-[9px] font-bold text-neutral-400 uppercase tracking-wider">COUNT</div>
            <div id="m_count" class="text-xl font-black text-white mt-1">-</div>
          </div>
          <div class="bg-black border border-neutral-700 p-2">
            <div class="text-[9px] font-bold text-neutral-400 uppercase tracking-wider">DENSITY</div>
            <div id="m_dens" class="text-xl font-black text-white mt-1">-</div>
          </div>
          <div class="bg-black border border-neutral-700 p-2">
            <div class="text-[9px] font-bold text-neutral-400 uppercase tracking-wider">PRESSURE</div>
            <div id="m_pres" class="text-xl font-black text-white mt-1">-</div>
          </div>
          <div class="bg-black border border-neutral-700 p-2">
            <div class="text-[9px] font-bold text-neutral-400 uppercase tracking-wider">TURBULENCE</div>
            <div id="m_turb" class="text-xl font-black text-white mt-1">-</div>
          </div>
          <div class="bg-black border border-neutral-700 p-2">
            <div class="text-[9px] font-bold text-neutral-400 uppercase tracking-wider">COHERENCE</div>
            <div id="m_coh" class="text-xl font-black text-white mt-1">-</div>
          </div>
          <div class="bg-black border border-neutral-700 p-2">
            <div class="text-[9px] font-bold text-neutral-400 uppercase tracking-wider">RISK SCORE</div>
            <div id="m_risk" class="text-xl font-black text-white mt-1">0.00</div>
          </div>
        </div>
      </div>

      <!-- Presets Selector -->
      <div class="neo-box-gray p-3 flex items-center justify-between gap-3 text-xs">
        <span class="font-bold uppercase tracking-wider text-neutral-300">Preset Profile:</span>
        <select id="preset_select" class="neo-input px-3 py-1.5 text-xs uppercase font-bold text-white bg-black">
          <option value="">(Custom)</option>
          <option value="High-Recall CCTV">High-Recall CCTV</option>
          <option value="Dense Crowd / Chokepoint">Dense Crowd / Chokepoint</option>
          <option value="Webcam / Close-Range">Webcam / Close-Range</option>
          <option value="High-Precision (Strict)">High-Precision (Strict)</option>
        </select>
      </div>

      <!-- Full Parameter Controls with Clear Explanations -->
      <div class="neo-box p-4 flex flex-col gap-4 text-xs">
        <div class="text-xs font-black uppercase tracking-wider text-white border-b-2 border-white pb-2 flex justify-between">
          <span>Parameters & Sensitivity</span>
          <span class="text-neutral-400 text-[10px]">Real-Time Sync</span>
        </div>

        <!-- CONF Slider -->
        <div class="flex flex-col gap-1">
          <div class="flex items-center justify-between">
            <span class="font-bold uppercase tracking-wider text-white">Confidence Floor (CONF)</span>
            <input type="number" id="num_CONF" min="0.001" max="1" step="0.005" value="0.02" class="w-16 neo-input text-right px-1.5 py-0.5 font-mono text-xs">
          </div>
          <input type="range" id="rng_CONF" min="0.001" max="1" step="0.005" value="0.02" class="w-full">
          <p class="text-[11px] text-neutral-400">Baseline detection confidence floor. Lower values capture distant small heads; higher values eliminate false positives.</p>
        </div>

        <!-- IOU Slider -->
        <div class="flex flex-col gap-1 border-t border-neutral-800 pt-3">
          <div class="flex items-center justify-between">
            <span class="font-bold uppercase tracking-wider text-white">NMS IoU Threshold</span>
            <input type="number" id="num_IOU" min="0.01" max="1" step="0.05" value="0.60" class="w-16 neo-input text-right px-1.5 py-0.5 font-mono text-xs">
          </div>
          <input type="range" id="rng_IOU" min="0.01" max="1" step="0.05" value="0.60" class="w-full">
          <p class="text-[11px] text-neutral-400">Non-maximum suppression overlap threshold. Lower values aggressively merge duplicate overlapping head boxes.</p>
        </div>

        <!-- Centroid Search Scale -->
        <div class="flex flex-col gap-1 border-t border-neutral-800 pt-3">
          <div class="flex items-center justify-between">
            <span class="font-bold uppercase tracking-wider text-white">Kinematic Search Scale</span>
            <input type="number" id="num_TRACK_MAX_DISPLACEMENT_SCALE" min="0.5" max="10" step="0.5" value="4.0" class="w-16 neo-input text-right px-1.5 py-0.5 font-mono text-xs">
          </div>
          <input type="range" id="rng_TRACK_MAX_DISPLACEMENT_SCALE" min="0.5" max="10" step="0.5" value="4.0" class="w-full">
          <p class="text-[11px] text-neutral-400">Centroid search cone radius multiplier relative to head scale. Higher values maintain tracking during fast motion.</p>
        </div>

        <!-- KDE Gamma -->
        <div class="flex flex-col gap-1 border-t border-neutral-800 pt-3">
          <div class="flex items-center justify-between">
            <span class="font-bold uppercase tracking-wider text-white">KDE Splat Gamma</span>
            <input type="number" id="num_KDE_GAMMA" min="0.1" max="3" step="0.1" value="0.8" class="w-16 neo-input text-right px-1.5 py-0.5 font-mono text-xs">
          </div>
          <input type="range" id="rng_KDE_GAMMA" min="0.1" max="3" step="0.1" value="0.8" class="w-full">
          <p class="text-[11px] text-neutral-400">Gaussian kernel standard deviation factor for continuous spatial crowd density mapping.</p>
        </div>

        <!-- Pressure Limit -->
        <div class="flex flex-col gap-1 border-t border-neutral-800 pt-3">
          <div class="flex items-center justify-between">
            <span class="font-bold uppercase tracking-wider text-white">Pressure Ceiling P_CRIT (k)</span>
            <input type="number" id="num_RISK_P_CRIT" min="10" max="2000" step="10" value="450" class="w-16 neo-input text-right px-1.5 py-0.5 font-mono text-xs">
          </div>
          <input type="range" id="rng_RISK_P_CRIT" min="10" max="2000" step="10" value="450" class="w-full">
          <p class="text-[11px] text-neutral-400">Helbing crowd pressure saturation ceiling (P = ρ · Var(v), in thousands) representing severe mechanical compression.</p>
        </div>

        <!-- Density Crush Limit -->
        <div class="flex flex-col gap-1 border-t border-neutral-800 pt-3">
          <div class="flex items-center justify-between">
            <span class="font-bold uppercase tracking-wider text-white">Density Crush Limit (DENS_CRIT)</span>
            <input type="number" id="num_RISK_DENS_CRIT" min="2" max="50" step="1" value="18" class="w-16 neo-input text-right px-1.5 py-0.5 font-mono text-xs">
          </div>
          <input type="range" id="rng_RISK_DENS_CRIT" min="2" max="50" step="1" value="18" class="w-full">
          <p class="text-[11px] text-neutral-400">Critical continuous crowd density threshold for stampede risk calculation.</p>
        </div>

        <!-- Minimum Crowd Count Floor -->
        <div class="flex flex-col gap-1 border-t border-neutral-800 pt-3">
          <div class="flex items-center justify-between">
            <span class="font-bold uppercase tracking-wider text-white">Count Gate Floor</span>
            <input type="number" id="num_RISK_COUNT_FLOOR" min="0" max="200" step="5" value="35" class="w-16 neo-input text-right px-1.5 py-0.5 font-mono text-xs">
          </div>
          <input type="range" id="rng_RISK_COUNT_FLOOR" min="0" max="200" step="5" value="35" class="w-full">
          <p class="text-[11px] text-neutral-400">Minimum instantaneous crowd count required before activating Watch, Warning, and Critical hazard stages.</p>
        </div>

      </div>

    </section>
  </main>

  <!-- History Drawer Modal (Neobrutalist Black & White) -->
  <div id="history_modal" class="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4 hidden">
    <div class="neo-box bg-black w-full max-w-2xl max-h-[80vh] flex flex-col p-5">
      <div class="flex items-center justify-between pb-3 border-b-2 border-white">
        <h2 class="text-base font-black uppercase tracking-wider text-white">Run History & Artifacts</h2>
        <button id="btn_close_history" class="text-white hover:text-neutral-300 text-xl font-bold px-2">&times;</button>
      </div>
      <div id="history_list" class="flex-1 overflow-y-auto mt-4 flex flex-col gap-3">
        <p class="text-xs text-neutral-500 text-center py-4 uppercase">No completed runs recorded yet.</p>
      </div>
    </div>
  </div>

  <script>
    let activeSource = { type: 'file', val: '' };
    let roiPoints = [];
    let schemaPresets = {};
    let isRunning = false;
    const $ = id => document.getElementById(id);

    // Linked Sliders + Numbers
    function linkInputs(param, scale=1) {
      const rng = $('rng_' + param), num = $('num_' + param);
      if (!rng || !num) return;
      rng.addEventListener('input', () => {
        num.value = rng.value;
        pushParam(param, parseFloat(rng.value) * scale);
      });
      num.addEventListener('input', () => {
        rng.value = num.value;
        pushParam(param, parseFloat(num.value) * scale);
      });
    }

    linkInputs('CONF');
    linkInputs('IOU');
    linkInputs('TRACK_MAX_DISPLACEMENT_SCALE');
    linkInputs('KDE_GAMMA');
    linkInputs('RISK_P_CRIT', 1000);
    linkInputs('RISK_DENS_CRIT');
    linkInputs('RISK_COUNT_FLOOR');

    function pushParam(key, value) {
      fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ params: { [key]: value } })
      });
    }

    // Source Tabs (Neobrutalist Styling)
    function setSourceTab(type) {
      activeSource.type = type;
      const tabs = ['file', 'webcam', 'rtsp', 'yt'];
      tabs.forEach(t => {
        const btn = $('tab_' + t);
        if (t === type) {
          btn.className = 'p-2 border-2 border-white bg-white text-black text-center font-black';
        } else {
          btn.className = 'p-2 border-2 border-neutral-700 bg-black text-neutral-400 hover:border-neutral-400 text-center font-bold';
        }
      });

      $('src_file_area').classList.toggle('hidden', type !== 'file');
      $('src_input_area').classList.toggle('hidden', type === 'file');

      if (type === 'webcam') $('src_val').value = '0';
      if (type === 'rtsp') $('src_val').placeholder = 'rtsp://... or http://...';
      if (type === 'yt') $('src_val').placeholder = 'https://www.youtube.com/watch?v=...';
    }

    $('tab_file').onclick = () => setSourceTab('file');
    $('tab_webcam').onclick = () => setSourceTab('webcam');
    $('tab_rtsp').onclick = () => setSourceTab('rtsp');
    $('tab_yt').onclick = () => setSourceTab('yt');

    // Drag & Drop File Upload
    $('dropzone').onclick = () => $('file_input').click();
    $('file_input').onchange = e => handleUpload(e.target.files[0]);

    $('dropzone').ondragover = e => { e.preventDefault(); $('dropzone').classList.add('border-white'); };
    $('dropzone').ondragleave = () => $('dropzone').classList.remove('border-white');
    $('dropzone').ondrop = e => {
      e.preventDefault();
      $('dropzone').classList.remove('border-white');
      if (e.dataTransfer.files.length) handleUpload(e.dataTransfer.files[0]);
    };

    function handleUpload(file) {
      if (!file) return;
      const fd = new FormData();
      fd.append('file', file);
      $('file_label').textContent = 'Uploading ' + file.name + '...';
      fetch('/api/upload', { method: 'POST', body: fd })
        .then(r => r.json())
        .then(data => {
          activeSource.val = data.filepath;
          $('file_label').textContent = data.filename;
          $('file_probe').textContent = `${data.resolution} | ${data.fps} FPS | ${data.duration_sec}s | ${data.size_mb} MB`;
          $('file_probe').classList.remove('hidden');
        });
    }

    // ── Interactive Movable ROI Polygon Editor ──
    const roiCanvas = $('roi_canvas');
    const ctx = roiCanvas.getContext('2d');
    let draggedPointIndex = null;
    let hoveredPointIndex = null;
    const HANDLE_RADIUS = 14; // pixels hit radius

    function resizeCanvas() {
      roiCanvas.width = roiCanvas.parentElement.clientWidth;
      roiCanvas.height = roiCanvas.parentElement.clientHeight;
      redrawROI();
    }
    window.addEventListener('resize', resizeCanvas);
    setTimeout(resizeCanvas, 200);

    function getCanvasCoords(e) {
      const rect = roiCanvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const normX = Math.max(0, Math.min(1, px / roiCanvas.width));
      const normY = Math.max(0, Math.min(1, py / roiCanvas.height));
      return { px, py, normX, normY };
    }

    function findNearPointIndex(px, py) {
      const W = roiCanvas.width, H = roiCanvas.height;
      for (let i = 0; i < roiPoints.length; i++) {
        const ppx = roiPoints[i][0] * W;
        const ppy = roiPoints[i][1] * H;
        const dist = Math.hypot(px - ppx, py - ppy);
        if (dist <= HANDLE_RADIUS) return i;
      }
      return -1;
    }

    function redrawROI() {
      ctx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
      const W = roiCanvas.width, H = roiCanvas.height;
      if (roiPoints.length === 0) return;

      // Draw Polygon Path & Fill
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(roiPoints[0][0] * W, roiPoints[0][1] * H);
      for (let i = 1; i < roiPoints.length; i++) {
        ctx.lineTo(roiPoints[i][0] * W, roiPoints[i][1] * H);
      }
      if (roiPoints.length >= 3) {
        ctx.closePath();
        ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.fill();
      }
      ctx.stroke();

      // Draw Draggable Square Handles at each vertex
      for (let i = 0; i < roiPoints.length; i++) {
        const cx = roiPoints[i][0] * W;
        const cy = roiPoints[i][1] * H;
        const isHovered = (i === hoveredPointIndex || i === draggedPointIndex);

        const size = isHovered ? 12 : 9;
        ctx.fillStyle = isHovered ? '#ffffff' : '#e5e5e5';
        ctx.fillRect(cx - size/2, cy - size/2, size, size);
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#000000';
        ctx.strokeRect(cx - size/2, cy - size/2, size, size);

        // Point Label
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 10px monospace';
        ctx.fillText(`P${i+1}`, cx + 7, cy - 7);
      }
    }

    function syncROIBoundary() {
      const polygon = roiPoints.length >= 3 ? roiPoints : null;
      fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ roi_polygon: polygon })
      });
      $('roi_status').textContent = roiPoints.length >= 3 
        ? `ROI active (${roiPoints.length} points — drag handles to reshape)`
        : roiPoints.length > 0 
        ? `ROI setup (${roiPoints.length}/3 min points)` 
        : 'click canvas to add ROI points';
    }

    roiCanvas.addEventListener('mousedown', e => {
      if (e.button !== 0) return; // Left-click only
      const { px, py, normX, normY } = getCanvasCoords(e);
      const idx = findNearPointIndex(px, py);

      if (idx !== -1) {
        // Start dragging existing vertex
        draggedPointIndex = idx;
        roiCanvas.style.cursor = 'grabbing';
      } else {
        // Add new vertex point
        roiPoints.push([normX, normY]);
        draggedPointIndex = roiPoints.length - 1;
        redrawROI();
        syncROIBoundary();
      }
    });

    roiCanvas.addEventListener('mousemove', e => {
      const { px, py, normX, normY } = getCanvasCoords(e);

      if (draggedPointIndex !== null) {
        // Update dragging position in real time
        roiPoints[draggedPointIndex] = [normX, normY];
        redrawROI();
        roiCanvas.style.cursor = 'grabbing';
      } else {
        // Check hover over handles
        const idx = findNearPointIndex(px, py);
        hoveredPointIndex = idx;
        roiCanvas.style.cursor = idx !== -1 ? 'grab' : 'crosshair';
        redrawROI();
      }
    });

    const stopDragging = () => {
      if (draggedPointIndex !== null) {
        draggedPointIndex = null;
        roiCanvas.style.cursor = 'crosshair';
        redrawROI();
        syncROIBoundary();
      }
    };

    window.addEventListener('mouseup', stopDragging);
    roiCanvas.addEventListener('mouseleave', () => {
      hoveredPointIndex = null;
      redrawROI();
    });

    // Remove vertex on double-click or right-click
    roiCanvas.addEventListener('dblclick', e => {
      const { px, py } = getCanvasCoords(e);
      const idx = findNearPointIndex(px, py);
      if (idx !== -1) {
        roiPoints.splice(idx, 1);
        redrawROI();
        syncROIBoundary();
      }
    });

    roiCanvas.addEventListener('contextmenu', e => {
      e.preventDefault();
      const { px, py } = getCanvasCoords(e);
      const idx = findNearPointIndex(px, py);
      if (idx !== -1) {
        roiPoints.splice(idx, 1);
        redrawROI();
        syncROIBoundary();
      }
    });

    $('btn_clear_roi').onclick = () => {
      roiPoints = [];
      draggedPointIndex = null;
      hoveredPointIndex = null;
      redrawROI();
      syncROIBoundary();
    };

    // Overlay Toggles
    ['boxes', 'ids', 'vel'].forEach(k => {
      $(`chk_${k}`).onchange = () => {
        fetch('/api/overlays', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            boxes: $('chk_boxes').checked,
            track_ids: $('chk_ids').checked,
            velocity: $('chk_vel').checked
          })
        });
      };
    });

    // Session Controls (Start / Pause / Stop)
    $('btn_start').onclick = () => {
      const src = activeSource.type === 'file' ? (activeSource.val || 'data/videos/easy.mp4') : $('src_val').value;
      fetch('/api/session/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_type: activeSource.type,
          source: src,
          roi_polygon: roiPoints.length >= 3 ? roiPoints : null
        })
      }).then(r => r.json()).then(res => {
        if (!res.error) {
          $('video_feed').src = '/video?' + Date.now();
          setRunningState(true);
        } else {
          alert(res.error);
        }
      });
    };

    $('btn_pause').onclick = () => {
      const isPaused = $('btn_pause').textContent === 'Resume';
      fetch(isPaused ? '/api/session/resume' : '/api/session/pause', { method: 'POST' })
        .then(() => $('btn_pause').textContent = isPaused ? 'Pause' : 'Resume');
    };

    $('btn_stop').onclick = () => {
      fetch('/api/session/stop', { method: 'POST' }).then(() => setRunningState(false));
    };

    function setRunningState(running) {
      isRunning = running;
      $('btn_start').classList.toggle('hidden', running);
      $('btn_pause').classList.toggle('hidden', !running);
      $('btn_stop').classList.toggle('hidden', !running);
    }

    // Presets
    fetch('/api/config/schema').then(r => r.json()).then(data => {
      schemaPresets = data.presets || {};
    });

    $('preset_select').onchange = e => {
      const p = schemaPresets[e.target.value];
      if (!p) return;
      Object.keys(p).forEach(k => {
        const rng = $('rng_' + k), num = $('num_' + k);
        let val = p[k];
        if (k === 'RISK_P_CRIT') val = val / 1000;
        if (rng) rng.value = val;
        if (num) num.value = val;
      });
      fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ params: p })
      });
    };

    // History Modal
    $('btn_history').onclick = () => {
      $('history_modal').classList.remove('hidden');
      loadHistory();
    };
    $('btn_close_history').onclick = () => $('history_modal').classList.add('hidden');

    function loadHistory() {
      fetch('/api/runs').then(r => r.json()).then(runs => {
        const hl = $('history_list');
        if (!runs || runs.length === 0) {
          hl.innerHTML = '<p class="text-xs text-neutral-500 text-center py-4 uppercase">No completed runs recorded yet.</p>';
          return;
        }
        hl.innerHTML = runs.map(r => `
          <div class="bg-neutral-950 p-3 border-2 border-neutral-700 flex items-center justify-between text-xs">
            <div>
              <div class="font-black text-white uppercase">${r.run_id} <span class="text-[10px] text-neutral-400 font-normal">(${r.source})</span></div>
              <div class="text-[11px] text-neutral-400 mt-1 uppercase">Duration: ${r.elapsed_sec}s | Peak: ${r.peak_crowd} heads | Max Risk: ${(r.max_risk*100).toFixed(0)}%</div>
            </div>
            <div class="flex gap-3 font-black text-[11px] uppercase">
              <a href="/api/runs/${r.run_id}/download/csv" class="text-white hover:underline border-b border-white">CSV</a>
              <a href="/api/runs/${r.run_id}/download/plot" class="text-white hover:underline border-b border-white">Plot</a>
              <a href="/api/runs/${r.run_id}/download/video" class="text-white hover:underline border-b border-white">Video</a>
              <a href="/api/runs/${r.run_id}/download/summary" class="text-neutral-400 hover:underline border-b border-neutral-400">JSON</a>
            </div>
          </div>
        `).join('');
      });
    }

    // SSE Metrics Listener
    const sse = new EventSource('/stream');
    sse.onmessage = e => {
      const data = JSON.parse(e.data);
      const m = data.metrics || {};
      const status = data.status || 'IDLE';
      
      $('badge_status').textContent = status;
      $('badge_status').className = `px-3 py-1 text-xs font-black uppercase tracking-wider border-2 border-white ${
        status === 'RUNNING' ? 'bg-white text-black' :
        status === 'PAUSED' ? 'bg-neutral-800 text-white' :
        'bg-black text-white'
      }`;

      if (status === 'COMPLETED' && isRunning) {
        setRunningState(false);
      }

      // Elapsed Timer
      const elap = data.elapsed || 0;
      const elapM = String(Math.floor(elap / 60)).padStart(2, '0');
      const elapS = String(Math.floor(elap % 60)).padStart(2, '0');
      $('timer_display').textContent = `ELAPSED: ${elapM}:${elapS}`;

      // Metrics
      $('m_count').textContent = (m.count_sma || m.count || 0).toFixed(0);
      $('m_dens').textContent = (m.peak_density_sma || 0).toFixed(1);
      $('m_pres').textContent = ((m.max_pressure_sma || 0) / 1000).toFixed(0) + 'k';
      $('m_turb').textContent = (m.turbulence_index_sma || 0).toFixed(1);
      $('m_coh').textContent = (m.flow_coherence_sma || 0).toFixed(2);
      
      const risk = m.risk_score_sma || 0;
      $('m_risk').textContent = risk.toFixed(2);
      
      const banner = $('risk_banner');
      const stage = m.alert_name || 'NORMAL';
      banner.className = `absolute top-3 left-3 px-4 py-1.5 text-xs font-black uppercase tracking-wider st-${stage}`;
      banner.textContent = `${stage} · ${(risk * 100).toFixed(0)}% (SMA)`;
    };
  </script>
</body>
</html>

```

---

---

## Log Output Schema (`data/runs/{run_id}/log.csv`)

44 total structured time-series columns exported every 0.25s:

| Group | Columns | Description |
|---|---|---|
| **Time** | `time_sec`, `sys_time` | Video timestamp (s) and Unix epoch wall-clock timestamp. |
| **Instantaneous Physics** | `count`, `density_score`, `peak_density`, `danger_area_pct`, `crowd_entropy`, `max_pressure`, `mean_momentum`, `turbulence_index`, `flow_coherence` | Real-time continuous crowd state metrics. |
| **Rolling SMA (20-tick)** | `*_sma` for all 9 physics metrics | Moving average smoothing out transient high-frequency noise. |
| **Rolling STD (20-tick)** | `*_std` for all 9 physics metrics | Local variance measuring crowd instability over time. |
| **Rolling Z-Score** | `*_z` for all 9 physics metrics | Statistical anomaly indicator $z = (x - \mu)/\sigma$. |
| **Risk Classifier** | `risk_score_5s`, `risk_score_sma`, `alert_stage`, `alert_name`, `p_sma5s`, `dens_sma5s` | Dual-gated smoothed risk score $\in [0, 1]$ and categorical stage (`NORMAL`/`WATCH`/`WARNING`/`CRITICAL`). |
