"""CPU-only settings for the head-counting pipeline."""
import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Input & Model Paths
# SOURCE = str(ROOT_DIR / "data" / "videos" / "easy.mp4")
# SOURCE = str(ROOT_DIR / "data" / "videos" / "medium.mp4")
#SOURCE = str(ROOT_DIR / "data" / "videos" / "h1.mp4")
SOURCE = str(ROOT_DIR / "data" / "videos" / "h2.mp4")
#SOURCE = str(ROOT_DIR / "data" / "videos" / "jagganath.mp4")
HEAD_MODEL = str(ROOT_DIR / "models" / "head_yolov8.onnx")

# Detection
CONF = 0.05
IOU = 0.75
IMGSZ = 512  # preserve detail for small/still distant heads; KLT remains half-resolution
MAX_DET = 500

# Faster crowd -> refresh detection more frequently
DETECT_EVERY_N = 60
DETECT_EVERY_N_MAX = 120
MAX_COUNT_AGE = 2.0

# Spatial distribution
GRID = 4

# Logging
LOG_INTERVAL = 1.0
LOG_CSV = str(ROOT_DIR / "data" / "outputs" / "log.csv")

# Visual audit
OUTPUT_VIDEO = str(ROOT_DIR / "data" / "outputs" / "output_head_klt.mp4")
OUTPUT_EVERY_N = 3

# KLT motion tracking
KLT_SCALE = 0.4
KLT_MAX_DISPLACEMENT = 80.0
KLT_FEATURES_PER_HEAD = 4
KLT_QUALITY_LEVEL = 0.01
KLT_MIN_DISTANCE = 3.0
