# CrowdShield - Real-Time Crowd Dynamics & Early Stampede Warning Platform

Production-grade, CPU-optimized computer vision platform for real-time crowd density estimation, hydrodynamic pressure mapping, kinematic motion tracking, and early stampede precursor detection.

CrowdShield operates as a headless, unified web platform where video ingestion, interactive movable ROI polygon boundaries, continuous parameter calibration, live telemetry, and per-run artifact archiving are managed through a single browser interface.

---

## 1. System Architecture

```text
+-----------------------------------------------------------------------------------------+
|                                    BROWSER INTERFACE                                    |
|   - Video Upload (Drag & Drop)       - Movable ROI Polygon Editor                       |
|   - Stream Ingestion (Webcam/RTSP/YT)- Continuous Parameter Sliders                     |
|   - Real-time Physics Telemetry      - Run History & Artifact Downloads                 |
+--------------------------------------------+--------------------------------------------+
                                             | HTTP REST + SSE + MJPEG
                                             v
+-----------------------------------------------------------------------------------------+
|                                    UNIFIED BACKEND                                      |
|                                                                                         |
|   [ HTTP REST API & Streaming Engine ] (src/server.py)                                  |
|     * Multipart Upload Handler with OpenCV Video Probe                                  |
|     * Server-Sent Events (SSE) 10 Hz Telemetry Broadcaster                              |
|     * Continuous MJPEG Video Frame Streamer                                             |
|                                                                                         |
|   [ Session & Lifecycle Manager ] (src/session_manager.py)                              |
|     * Threaded Multi-Source Ingestion (File, Webcam DirectShow, RTSP, YouTube via yt-dlp)|
|     * Movable ROI Polygon In-Point Testing (cv2.pointPolygonTest)                       |
|     * Per-Run Isolated Artifact Storage (data/runs/run_YYYYMMDD_HHMMSS/)                |
|                                                                                         |
|   [ Async Inference & Physics Engine ] (src/features.py)                                |
|     * YOLOv8 Head Detector (OpenVINO FP16 CPU Accelerated)                              |
|     * Fast Kinematic Lookahead Centroid Tracker with Multi-Frame Coasting               |
|     * Scale-Adaptive Gaussian Kernel Density Estimation (KDE)                           |
|     * Helbing Hydrodynamic Crowd Pressure Mapping (P = rho * Var(v))                    |
|     * Local Momentum Vector Field (Mx, My) & Flow Coherence Index                       |
|     * Dual-Gated Moving-Average Stampede Risk Classifier                                |
|     * Automated 3-Panel Matplotlib Dynamics Plotter (src/plot_results.py)               |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Core Mathematical and Physical Models

### Perspective Scale Field S(y)
Decouples perspective distortion from head bounding boxes using median vertical head heights smoothed across horizontal grid rows via Exponential Moving Average (EMA):
$$S_t(y) = (1 - \alpha) S_{t-1}(y) + \alpha \cdot \operatorname{median}(h_i)$$

Ground-normalized density factor:
$$\text{density\_score} = \sum_{i} \left(\frac{S(y_i)}{S(y_{\text{bottom}})}\right)^2$$

### Fast Kinematic Centroid Tracker with Multi-Frame Coasting
Employs constant-velocity kinematic projection to estimate expected head positions:
$$\hat{\mathbf{c}}_{t} = \mathbf{c}_{t-\Delta t} + \mathbf{v}_{t-\Delta t} \cdot \Delta t$$
Detections are associated with projected positions rather than stale coordinates. Tracks survive temporary occlusions or low-confidence detector drops for up to 6 ticks (1.5 seconds) without losing identity. Search cones expand dynamically based on velocity:
$$d_{\text{max}} = \max\left(\text{scale} \cdot S(y), 0.7 \cdot \text{scale} \cdot S(y) + 0.5 \cdot \|\mathbf{v}_{\text{prev}}\| \cdot \Delta t\right)$$

### Scale-Adaptive Continuous Gaussian KDE
Computes smooth continuous crowd density without discrete grid artifacts:
$$\rho(\mathbf{x}) = \sum_i \frac{1}{2\pi \sigma_i^2} \exp\left(-\frac{\|\mathbf{x} - \mathbf{c}_i\|^2}{2\sigma_i^2}\right), \quad \sigma_i = \max\left(\frac{\gamma \cdot S(y_i)}{\text{scale\_down}}, 1.5\right)$$

### Helbing Hydrodynamic Crowd Pressure Field
Calculates mechanical compressive force, a critical precursor to physical crushing and stampedes:
$$P(\mathbf{x}) = \rho(\mathbf{x}) \cdot \operatorname{Var}(\mathbf{v})(\mathbf{x})$$
$$\operatorname{Var}(\mathbf{v}) = \frac{\sum w_i \|\mathbf{v}_i\|^2}{\sum w_i} - \left[\left(\frac{\sum w_i u_i}{\sum w_i}\right)^2 + \left(\frac{\sum w_i v_i}{\sum w_i}\right)^2\right]$$

### Local Momentum and Flow Coherence
Evaluates directional crowd alignment:
$$\mathbf{M}(\mathbf{x}) = (M_x, M_y) = (\rho U, \rho V)$$
$$\Phi = \frac{\sqrt{\left(\sum M_x\right)^2 + \left(\sum M_y\right)^2}}{\sum \|\mathbf{M}\| + \epsilon}$$
- $\Phi \approx 1.0$: Laminar, unified directional motion.
- $\Phi \to 0.0$: Turbulent, opposing, or chaotic crowd collisions.

### Dual-Gated Moving-Average Stampede Risk Classifier
Evaluates smoothed 20-tick rolling physics indicators against critical saturation limits:
$$s_{\text{score}} = \min\left(\frac{\text{peak\_density\_sma}}{\text{RISK\_DENS\_CRIT}}, 1.0\right)$$
$$d_{\text{score}} = \min\left(\frac{\text{max\_pressure\_sma}}{\text{RISK\_P\_CRIT}}, 1.0\right) \cdot (1 - \text{flow\_coherence\_sma}) \cdot \min\left(1 + \frac{\text{turbulence\_index\_sma}}{\text{RISK\_TURB\_CRIT}}, 2.0\right)$$
$$\text{raw\_risk} = 0.35 \cdot s_{\text{score}} + 0.65 \cdot d_{\text{score}}$$
$$\text{risk\_score\_sma} = \frac{1}{1 + \exp\left(-k \cdot (\text{raw\_risk} - \text{mid})\right)}$$

Alert stages:
- `NORMAL`: $\text{risk\_score} < 0.30$
- `WATCH`: $0.30 \le \text{risk\_score} < 0.55$
- `WARNING`: $0.55 \le \text{risk\_score} < 0.80$
- `CRITICAL`: $\text{risk\_score} \ge 0.80$

---

## 3. Repository Structure

```text
crowd_prototype/v0.3/
|-- main.py                   # Root headless server launcher
|-- setup_model.py            # Automated OpenVINO model downloader & exporter
|-- requirements.txt          # Python runtime dependencies
|-- .gitignore                # Git exclusions for models, logs, and artifacts
|-- README.md                 # Technical documentation
|-- codebase.md               # Full codebase reference and synchronization log
|-- src/
|   |-- __init__.py
|   |-- config.py             # Default paths and core parameters
|   |-- config_schema.py      # Parameter schema, bounds validation, and presets
|   |-- session_manager.py    # Session lifecycle, ROI filtering, per-run manager
|   |-- server.py             # HTTP REST API, SSE telemetry, MJPEG streaming
|   |-- features.py           # Physics engine (KDE, Helbing pressure, tracker, classifier)
|   |-- plot_results.py       # 3-panel matplotlib graph generator
|   |-- collect.py            # Legacy standalone pipeline
|   |-- ui_controls.py        # Legacy OpenCV UI controls
|   `-- web_server.py         # Legacy thin bridge
|-- web/
|   `-- index.html            # Neobrutalist dashboard, ROI canvas, telemetry cards
|-- models/
|   |-- .gitkeep
|   `-- head_yolov8_openvino_model/   # Generated by setup_model.py
`-- data/
    |-- uploads/              # Uploaded video storage
    |-- videos/               # Sample input videos
    `-- runs/                 # Per-run artifact folders (run_YYYYMMDD_HHMMSS/)
```

---

## 4. Installation and Setup

### Prerequisites
- Python 3.9, 3.10, or 3.11
- Operating System: Linux, macOS, or Windows
- Hardware: Standard x86_64 CPU (OpenVINO optimized)

### Step 1: Clone and Create Virtual Environment
```bash
git clone https://github.com/S-Srinivasan-06/CrowdShield.git
cd CrowdShield

python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download and Export Head Detection Model
```bash
python setup_model.py
```
This downloads `head_yolov8.pt` from HuggingFace and exports an optimized FP16 OpenVINO model into `models/head_yolov8_openvino_model/`.

---

## 5. Running the Application

Start the headless server:
```bash
python main.py
```

Access the dashboard in any modern web browser:
```
http://localhost:8000
```

---

## 6. Web Dashboard Features

- **Source Selection**:
  - Video Upload: Drag and drop MP4 or WebM files with automatic video probing (resolution, FPS, duration, file size).
  - Webcam: USB or DirectShow camera index (e.g., `0`).
  - RTSP: Low-latency RTSP / HTTP network stream URL.
  - YouTube Live: Direct YouTube stream URL (resolved via `yt-dlp`).
- **Interactive Movable ROI Polygon**:
  - Click anywhere on the video canvas to create boundary vertices.
  - Drag vertex handles (`P1`, `P2`, `P3`, ...) to reshape the polygon in real-time.
  - Double-click or right-click any handle to remove a vertex.
  - Model detects and tracks individuals exclusively within the defined polygon.
- **Real-Time Dynamics Cards**:
  - `COUNT`: Smoothed crowd headcount.
  - `DENSITY`: Spatial peak density.
  - `PRESSURE`: Helbing crowd pressure (in thousands).
  - `TURBULENCE`: Momentum variance index.
  - `COHERENCE`: Directional uniformity $\Phi \in [0, 1]$.
  - `RISK SCORE`: Dual-gated sigmoid stampede risk score $\in [0.00, 1.00]$.
- **Parameter Controls & Presets**:
  - Presets: `High-Recall CCTV`, `Dense Crowd / Chokepoint`, `Webcam / Close-Range`, `High-Precision (Strict)`.
  - Continuous sliders and numeric inputs for `CONF` (0.001 - 1.000), `IOU`, `TRACK_MAX_DISPLACEMENT_SCALE`, `KDE_GAMMA`, `RISK_P_CRIT`, `RISK_DENS_CRIT`, and `RISK_COUNT_FLOOR`.
- **Run History & Artifact Archiving**:
  - Every completed session is preserved under `data/runs/run_YYYYMMDD_HHMMSS/`.
  - Direct download links for `log.csv`, `output_dynamics.mp4`, `count_moving_average.png`, `config.json`, and `summary.json`.

---

## 7. REST & Streaming API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/config/schema` | Returns parameter schema, bounds, and preset dictionaries. |
| `GET` | `/api/state` | Returns live session status, elapsed time, frame count, and physics metrics. |
| `POST` | `/api/config` | Updates hyperparameters dynamically on the active session. |
| `POST` | `/api/overlays` | Toggles rendering of bounding boxes, track IDs, and velocity arrows. |
| `POST` | `/api/upload` | Multipart form-data video file upload with metadata probe. |
| `POST` | `/api/session/start` | Starts an analysis session with specified source and optional ROI polygon. |
| `POST` | `/api/session/pause` | Pauses video ingestion and inference. |
| `POST` | `/api/session/resume` | Resumes active session. |
| `POST` | `/api/session/stop` | Stops session, flushes CSV, and generates summary and plots. |
| `GET` | `/stream` | Server-Sent Events (SSE) telemetry stream emitting at 10 Hz. |
| `GET` | `/video` | Multi-part JPEG (MJPEG) video stream with active visual overlays. |
| `GET` | `/api/runs` | Returns list of all historical runs from `data/runs/`. |
| `GET` | `/api/runs/{run_id}/download/{artifact}` | Downloads run artifact (`csv`, `video`, `plot`, `config`, `summary`). |

---

## 8. Log Feature Schema (`log.csv`)

Each row in `data/runs/{run_id}/log.csv` contains 44 structured time-series features logged every 0.25 seconds:

| Category | Columns | Description |
|---|---|---|
| **Time** | `time_sec`, `sys_time` | Video relative timestamp and Unix epoch wall-clock time. |
| **Instantaneous Physics** | `count`, `density_score`, `peak_density`, `danger_area_pct`, `crowd_entropy`, `max_pressure`, `mean_momentum`, `turbulence_index`, `flow_coherence` | Real-time continuous crowd state variables. |
| **Rolling SMA (20-tick)** | `*_sma` for all 9 physics metrics | Moving average smoothing high-frequency noise. |
| **Rolling STD (20-tick)** | `*_std` for all 9 physics metrics | Local standard deviation measuring temporal instability. |
| **Rolling Z-Score** | `*_z` for all 9 physics metrics | Statistical anomaly indicator $z = (x - \mu)/\sigma$. |
| **Risk Classifier** | `risk_score_5s`, `risk_score_sma`, `alert_stage`, `alert_name`, `p_sma5s`, `dens_sma5s` | Smoothed risk probability and categorical alert stage (`NORMAL`, `WATCH`, `WARNING`, `CRITICAL`). |

---

## 9. Production Deployment Guidelines

### Linux Systemd Service Setup
Create `/etc/systemd/system/crowdshield.service`:

```ini
[Unit]
Description=CrowdShield Vision Platform
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/crowdshield
ExecStart=/opt/crowdshield/venv/bin/python main.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable crowdshield
sudo systemctl start crowdshield
sudo systemctl status crowdshield
```

### Nginx Reverse Proxy Configuration
Add to `/etc/nginx/sites-available/crowdshield`:

```nginx
server {
    listen 80;
    server_name crowdshield.local;

    client_max_body_size 500M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding off;
    }
}
```

Enable and reload Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/crowdshield /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```
