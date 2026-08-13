"""CPU-only settings for the head-counting pipeline."""

SOURCE = "r1.mp4"

HEAD_MODEL = "head_yolov8.pt"

# Detection
CONF = 0.05
IOU = 0.70
IMGSZ = 832  # preserve detail for small/still distant heads; KLT remains half-resolution
MAX_DET = 500

# Faster crowd -> refresh detection more frequently
DETECT_EVERY_N = 30
DETECT_EVERY_N_MAX = 60
MAX_COUNT_AGE = 1.0

# Spatial distribution
GRID = 4

# Logging
LOG_INTERVAL = 1.0
LOG_CSV = "log.csv"

# Visual audit
OUTPUT_VIDEO = "output_head_klt.mp4"
OUTPUT_EVERY_N = 1

# KLT motion tracking
KLT_SCALE = 0.5
KLT_MAX_DISPLACEMENT = 80.0
KLT_FEATURES_PER_HEAD = 4
KLT_QUALITY_LEVEL = 0.01
KLT_MIN_DISTANCE = 3.0
