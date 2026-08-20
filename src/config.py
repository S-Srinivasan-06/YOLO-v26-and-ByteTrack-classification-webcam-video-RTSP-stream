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
