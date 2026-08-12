# config.py
MODEL_NAME   = "yolo26n.pt"     # auto-downloads; fallback "yolov8n.pt"
SOURCE       = "video.mp4"      # video file path, or 0 / "rtsp://..." for live
TRACKER      = "bytetrack.yaml" # built into ultralytics (needs `lap`)
PERSON_CLASS = 0

GRID         = 4                # 4x4 spatial grid features
HIST_WINDOW  = 180              # seconds of history kept for lag features
LOG_INTERVAL = 1.0              # seconds between logged rows (1 Hz)
PRED_INTERVAL= 5.0              # seconds between live predictions

HORIZON      = 600              # forecast horizon: 10 minutes
TARGET       = "count"          # what to forecast: "count" | "wcount" | "cov"

LOG_CSV      = "log.csv"
MODEL_FILE   = "crowd_gbm.joblib"
SHOW         = True             # set False for headless servers
SAVE_VIDEO   = True             # save annotated output video
OUTPUT_VIDEO = "output.mp4"     # output video path
