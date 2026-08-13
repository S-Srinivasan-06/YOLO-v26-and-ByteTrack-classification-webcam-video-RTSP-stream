# CPU Head-Count Pipeline

This pipeline counts **head-detector boxes**, not tracker survivors. KLT optical
flow runs at half resolution (0.5 scale, 25% pixel area) to supply speed and direction features.

## Model setup

Download/export a pretrained single-class YOLOv8 or YOLO11 `.pt` checkpoint
trained for heads (for example, from the [SCUT-HEAD Roboflow Universe
project](https://universe.roboflow.com/viet-hoang-head/scut-head-part-a)), place
it beside the scripts as `head_yolov8.pt`, or update `HEAD_MODEL` in `config.py`.
The program rejects every checkpoint unless its classes are exactly
`{0: "head"}`; do not substitute a COCO person detector.

## Run

```bash
pip install -r requirements.txt
python collect.py
```

`log.csv` has `t` plus `count, cov, wcount, speed, dirx, diry, consist, g0..g15`.
Each row uses detector density features from a head detection no older than
0.5 seconds. If inference becomes expensive after five seconds of source video,
the governor doubles the scheduled detection interval (up to 40 frames), while
still forcing a detection before a count could become stale.
