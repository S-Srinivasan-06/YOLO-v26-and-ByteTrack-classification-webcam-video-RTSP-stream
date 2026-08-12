# Crowd Density Forecast (YOLO26 → LightGBM)

Real-time people detection + tracking, converted into uncalibrated crowd
features, used to forecast crowd density **10 minutes ahead** with LightGBM.
No camera calibration required — works on a fixed camera with raw pixel boxes.

## Pipeline

```
video/webcam ─▶ YOLO26 detect ─▶ ByteTrack track ─▶ features (1 Hz)
        ─▶ log.csv ─▶ label t+600s ─▶ LightGBM ─▶ live forecast
```

## Install

```bash
pip install -r requirements.txt
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"   # verify
```

## Usage

```bash
python collect.py   # 1) gather data  -> log.csv (run on hours of footage)
python train.py     # 2) train model  -> crowd_gbm.joblib
python predict.py   # 3) live current density + 10-min forecast
```

Set `SOURCE` in `config.py` (video path, `0` for webcam, RTSP URL).
Set `SHOW=False` on headless servers.

## Features (28, all uncalibrated)

| Group | Features | Meaning |
|---|---|---|
| Density | `count`, `cov`, `wcount` | raw count; box-pixel coverage; depth-weighted count (boxes lower in frame weigh more) |
| Motion | `speed`, `dirx`, `diry`, `consist` | mean track speed (px/s); mean direction unit vector; 0=chaotic, 1=orderly flow |
| History | `lag10..lag120`, `trend` | counts 10/30/60/120 s ago; count change over 60 s |
| Space | `g0..g15` | 4×4 grid counts — where the crowd is |

## Assumptions & limits

- **Fixed camera, no zoom/PTZ** (pixel-space velocity breaks otherwise).
- Speed is relative (px/s), not m/s — fine for forecasting.
- First ~2 minutes after startup have partial lag history (warm-up).
- Data: collect **several hours** covering different crowd states
  (empty → busy). A 10-min horizon needs the model to see real inflow/outflow
  patterns, not just one static scene.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'lap'` / tracker error | `pip install lap` |
| `yolo26n.pt` not found | `pip install -U ultralytics`, or use `yolov8n.pt` |
| No display (server) | `SHOW=False` in config |
| Slow inference | use `n` model + GPU, or lower video FPS |

## Extending

- Inflow/outflow counts at ROI edges (strongest predictor of density change).
- Time-of-day / day-of-week columns for long deployments.
- Forecast each grid cell (`g0..g15`) instead of global count.
- Add `overlap_ratio` (mean pairwise IoU) as an occlusion/saturation proxy.
