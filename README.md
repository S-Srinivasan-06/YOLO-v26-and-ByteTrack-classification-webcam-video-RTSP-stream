# Real-Time CPU Head Detection & KLT Motion Tracking Pipeline

A lightweight, CPU-optimized computer vision pipeline for real-time crowd density estimation and motion dynamics logging.

The system performs direct **single-class YOLO head detection** combined with **sparse Lucas-Kanade (KLT) optical flow tracking** to extract per-second density metrics, spatial distribution grids, and crowd velocity vectors without the overhead of heavy multi-object tracking algorithms (like DeepSORT or ByteTrack).

---

## Key Features

- **Direct Head Detection Counting**: Counts actual head detection bounding boxes per frame rather than tracking survivor IDs, preventing drift in dense crowds.
- **CPU-Optimized Adaptive Inference**: Built-in adaptive detection governor automatically adjusts detection intervals (`DETECT_EVERY_N` up to `DETECT_EVERY_N_MAX`) under heavy CPU loads while enforcing a freshness deadline (`MAX_COUNT_AGE = 1.0s`).
- **Feature-Preserved KLT Motion**: Runs optical flow at half resolution (`KLT_SCALE = 0.5`) seeded with Shi-Tomasi corners (`cv2.goodFeaturesToTrack`) inside detected head boxes to compute speed, mean movement vectors, and directional consistency.
- **Spatial Grid Distribution**: Computes 4×4 spatial distribution density grids (`g0` to `g15`) across the camera frame.
- **1 Hz CSV Logging**: Periodically exports structured feature vectors to `log.csv` for downstream analysis or time-series forecasting.
- **Visual Audit Video**: Generates an annotated MP4 video (`output_head_klt.mp4`) displaying detected head bounding boxes, optical flow motion vectors, and real-time status overlays at native video framerate.

---

## Directory Structure

```text
.
├── config.py          # Pipeline hyperparameters, detection intervals & KLT parameters
├── features.py        # Density metrics, spatial grid calculation & velocity store
├── collect.py         # Main execution script: video capture, YOLO inference, KLT & logging
├── requirements.txt   # Python package dependencies
├── codebase.md        # Full codebase reference documentation
└── README.md          # Project overview and instructions
```

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/S-Srinivasan-06/YOLO-v26-and-ByteTrack-classification-webcam-video-RTSP-stream.git
   cd YOLO-v26-and-ByteTrack-classification-webcam-video-RTSP-stream
   ```

2. **Create and activate a virtual environment (optional)**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Model Setup

Download a pretrained single-class YOLOv8/YOLO11 head detection checkpoint (e.g. trained on SCUT-HEAD or CrowdHuman-heads) and place it in the root directory as `head_yolov8.pt`, or configure `HEAD_MODEL` in `config.py`.

> **Note**: The pipeline validates the model on load and requires single-class head detection (`names == {0: "head"}`). Standard COCO 80-class models are not accepted.

---

## Configuration (`config.py`)

Key parameters can be adjusted directly in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `SOURCE` | `"r1.mp4"` | Input video file path, webcam index, or RTSP stream URL |
| `HEAD_MODEL` | `"head_yolov8.pt"` | Path to single-class YOLO head weights |
| `CONF` | `0.05` | Confidence threshold for head detection |
| `IOU` | `0.70` | NMS IoU threshold |
| `IMGSZ` | `832` | Detection image resolution (preserves distant heads) |
| `DETECT_EVERY_N` | `30` | Base frame interval between detector passes |
| `DETECT_EVERY_N_MAX` | `60` | Maximum frame interval under CPU throttle |
| `MAX_COUNT_AGE` | `1.0` | Max seconds before a fresh detection is forced |
| `KLT_SCALE` | `0.5` | Resolution scale factor for KLT optical flow |
| `KLT_FEATURES_PER_HEAD` | `4` | Target optical flow corner points per head |
| `OUTPUT_VIDEO` | `"output_head_klt.mp4"` | Filepath for visual audit MP4 |
| `OUTPUT_EVERY_N` | `1` | Frame write cadence for the output video |
| `LOG_INTERVAL` | `1.0` | CSV log interval in seconds (1 Hz) |
| `LOG_CSV` | `"log.csv"` | Output CSV log path |

---

## Usage

Run the main pipeline:

```bash
python collect.py
```

During execution, the script processes the video stream, calculates real-time motion and density features, logs rows to `log.csv`, and saves the annotated audit video.

Upon completion, benchmark telemetry is printed:
```text
benchmark: wall=73.47s video=61.53s ratio=1.194 detection_frames=62 klt_frames=1846 rows=62 klt_min=149 klt_median=233 video=output_head_klt.mp4
```

---

## Output Data Format (`log.csv`)

The exported `log.csv` contains uncalibrated density and motion features:

| Column | Description |
|---|---|
| `t` | Video timestamp in seconds |
| `count` | Total detected head count |
| `cov` | Bounding box area coverage ratio ($0.0 - 1.0$) |
| `wcount` | Vertical perspective depth-weighted head count |
| `speed` | Mean optical flow displacement speed (px/sec) |
| `dirx`, `diry` | Normalized aggregate motion direction unit vectors |
| `consist` | Directional consistency magnitude ($0.0 = \text{chaotic}, 1.0 = \text{uniform}$) |
| `g0` ... `g15` | 4×4 spatial distribution grid cell counts |
