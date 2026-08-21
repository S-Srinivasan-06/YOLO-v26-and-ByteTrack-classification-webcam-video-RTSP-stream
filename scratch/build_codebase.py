from pathlib import Path

overview = """# CrowdShield – Codebase Reference

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
  $$S_t(y) = (1 - \\alpha) S_{t-1}(y) + \\alpha \\cdot \\operatorname{median}(h_i)$$
- Ground-normalized density factor:
  $$\\text{density\\_score} = \\sum_{i} \\left(\\frac{S(y_i)}{S(y_{\\text{bottom}})}\\right)^2$$

### 2. Fast Kinematic Velocity-Projected Centroid Tracker (with Multi-Frame Coasting)
- Uses constant-velocity projection to predict expected positions:
  $$\\hat{\\mathbf{c}}_{t} = \\mathbf{c}_{t-\\Delta t} + \\mathbf{v}_{t-\\Delta t} \\cdot \\Delta t$$
- Matches current detections against projected positions $\\hat{\\mathbf{c}}$ rather than stale static positions.
- **Multi-Frame Track Coasting**: Tracks survive momentary occlusions or low-confidence detector drops (up to 6 ticks / 1.5s) without losing their unique track ID.
- **Dynamic Search Cones**: Expands search radius for fast movers proportional to speed:
  $$d_{\\text{max}} = \\max\\left(\\text{scale} \\cdot S(y), 0.7 \\cdot \\text{scale} \\cdot S(y) + 0.5 \\cdot \\|\\mathbf{v}_{prev}\\| \\cdot \\Delta t\\right)$$
- Applies exponential momentum smoothing on velocity vectors ($\\alpha = 0.8$):
  $$\\mathbf{v}_{t} = 0.8 \\cdot \\frac{\\mathbf{c}_t - \\mathbf{c}_{t-\\Delta t}}{\\Delta t} + 0.2 \\cdot \\mathbf{v}_{t-\\Delta t}$$

### 3. Scale-Adaptive Gaussian KDE
- Continuous spatial crowd density field:
  $$\\rho(\\mathbf{x}) = \\sum_i \\frac{1}{2\\pi \\sigma_i^2} \\exp\\left(-\\frac{\\|\\mathbf{x} - \\mathbf{c}_i\\|^2}{2\\sigma_i^2}\\right), \\quad \\sigma_i = \\max\\left(\\frac{\\gamma \\cdot S(y_i)}{\\text{scale\\_down}}, 1.5\\right)$$

### 4. Helbing Hydrodynamic Crowd Pressure
- Compressive mechanical force precursor to stampede:
  $$P(\\mathbf{x}) = \\rho(\\mathbf{x}) \\cdot \\operatorname{Var}(\\mathbf{v})(\\mathbf{x})$$
  $$\\operatorname{Var}(\\mathbf{v}) = \\frac{\\sum w_i \\|\\mathbf{v}_i\\|^2}{\\sum w_i} - \\left[\\left(\\frac{\\sum w_i u_i}{\\sum w_i}\\right)^2 + \\left(\\frac{\\sum w_i v_i}{\\sum w_i}\\right)^2\\right]$$

### 5. Local Momentum & Flow Coherence
- Vector momentum field $\\mathbf{M}(\\mathbf{x}) = (M_x, M_y) = (\\rho U, \\rho V)$.
- Flow Coherence index $\\Phi \\in [0, 1]$:
  $$\\Phi = \\frac{\\sqrt{\\left(\\sum M_x\\right)^2 + \\left(\\sum M_y\\right)^2}}{\\sum \\|\\mathbf{M}\\| + \\epsilon}$$

### 6. Dual-Gated Moving-Average Stampede Risk Classifier
- Evaluated directly on smoothed moving-average features (`count_sma`, `peak_density_sma`, `max_pressure_sma`, `turbulence_index_sma`, `flow_coherence_sma`):
  $$s_{\\text{score}} = \\min\\left(\\frac{\\text{peak\\_density\\_sma}}{\\text{RISK\\_DENS\\_CRIT}}, 1.0\\right)$$
  $$d_{\\text{score}} = \\min\\left(\\frac{\\text{max\\_pressure\\_sma}}{\\text{RISK\\_P\\_CRIT}}, 1.0\\right) \\cdot (1 - \\text{flow\\_coherence\\_sma}) \\cdot \\min\\left(1 + \\frac{\\text{turbulence\\_index\\_sma}}{\\text{RISK\\_TURB\\_CRIT}}, 2.0\\right)$$
  $$\\text{raw\\_risk} = 0.35 \\cdot s_{\\text{score}} + 0.65 \\cdot d_{\\text{score}}$$
  $$\\text{risk\\_score\\_sma} = \\frac{1}{1 + \\exp\\left(-k \\cdot (\\text{raw\\_risk} - \\text{mid})\\right)}$$
- Alert Stage Mapping:
  - `0 (NORMAL)`: $\\text{risk\\_score} < 0.30$
  - `1 (WATCH)`: $0.30 \\le \\text{risk\\_score} < 0.55$
  - `2 (WARNING)`: $0.55 \\le \\text{risk\\_score} < 0.80$
  - `3 (CRITICAL)`: $\\text{risk\\_score} \\ge 0.80$

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
"""

files = [
    ('requirements.txt', 'text'),
    ('.gitignore', 'gitignore'),
    ('setup_model.py', 'python'),
    ('main.py', 'python'),
    ('src/__init__.py', 'python'),
    ('src/config.py', 'python'),
    ('src/config_schema.py', 'python'),
    ('src/session_manager.py', 'python'),
    ('src/server.py', 'python'),
    ('src/features.py', 'python'),
    ('src/plot_results.py', 'python'),
    ('src/ui_controls.py', 'python'),
    ('src/web_server.py', 'python'),
    ('src/collect.py', 'python'),
    ('web/index.html', 'html'),
]

doc_parts = [overview]

for rel_path, lang in files:
    content = Path(rel_path).read_text(encoding='utf-8')
    doc_parts.append(f"### `{rel_path}`\n\n```{lang}\n{content}\n```")

footer = r"""---

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
"""

doc_parts.append(footer)

full_text = "\n\n---\n\n".join(doc_parts)
Path("codebase.md").write_text(full_text, encoding="utf-8")
print("SUCCESS: codebase.md completely generated with all 15 source files.")
