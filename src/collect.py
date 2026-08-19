"""CPU-only head detection plus KLT motion logging; no tracker or video output."""
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    import src.config as C
    from src.features import FEATURE_COLS, TrackStore, detection_features, motion_features
except ImportError:
    import config as C
    from features import FEATURE_COLS, TrackStore, detection_features, motion_features

KLT_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def load_head_model():
    """Load only an exact single-class head model, never a COCO person model."""
    if not os.path.exists(C.HEAD_MODEL):
        raise FileNotFoundError(
            f"Head checkpoint not found: {C.HEAD_MODEL}. Download a pretrained "
            "single-class SCUT-HEAD/CrowdHuman-heads YOLO checkpoint or export "
            "an OpenVINO/ONNX model first."
        )

    model = YOLO(C.HEAD_MODEL)
    names = {int(key): str(value).strip().lower() for key, value in model.names.items()}

    if names != {0: "head"}:
        raise ValueError(
            f"Refusing {C.HEAD_MODEL}: expected model.names == {{0: 'head'}}, got {names}."
        )

    return model


def make_gray(frame):
    return cv2.cvtColor(
        cv2.resize(frame, None, fx=C.KLT_SCALE, fy=C.KLT_SCALE, interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY,
    )


def advance_klt(previous_gray, gray, points, point_ids):
    """Advance KLT points and discard failed or implausibly displaced points."""
    if previous_gray is None or len(points) == 0:
        return np.empty((0, 2), np.float32), np.empty(0, np.int64)
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, gray, points.reshape(-1, 1, 2), None, **KLT_PARAMS
    )
    if next_points is None or status is None:
        return np.empty((0, 2), np.float32), np.empty(0, np.int64)
    next_points = next_points.reshape(-1, 2)
    valid = status.reshape(-1).astype(bool)
    displacement = np.linalg.norm(next_points - points, axis=1)
    valid &= displacement <= C.KLT_MAX_DISPLACEMENT * C.KLT_SCALE
    return next_points[valid].astype(np.float32), point_ids[valid]


def reseed_klt(gray, points, point_ids, boxes, next_point_id):
    """Keep KLT motion samples in head boxes and replenish textured features."""
    scaled_boxes = np.round(boxes * C.KLT_SCALE).astype(int)
    height, width = gray.shape
    if len(points) and len(scaled_boxes):
        in_any_head = np.zeros(len(points), dtype=bool)
        for x1, y1, x2, y2 in scaled_boxes:
            in_any_head |= ((points[:, 0] >= x1) & (points[:, 0] <= x2)
                            & (points[:, 1] >= y1) & (points[:, 1] <= y2))
        points, point_ids = points[in_any_head], point_ids[in_any_head]
    else:
        points, point_ids = np.empty((0, 2), np.float32), np.empty(0, np.int64)

    additions = []
    for x1, y1, x2, y2 in scaled_boxes:
        x1, x2 = np.clip(sorted((x1, x2)), 0, width - 1)
        y1, y2 = np.clip(sorted((y1, y2)), 0, height - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        already_here = (((points[:, 0] >= x1) & (points[:, 0] <= x2)
                         & (points[:, 1] >= y1) & (points[:, 1] <= y2)).sum())
        needed = max(C.KLT_FEATURES_PER_HEAD - already_here, 0)
        if not needed:
            continue
        mask = np.zeros_like(gray)
        mask[y1:y2 + 1, x1:x2 + 1] = 255
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=needed, qualityLevel=C.KLT_QUALITY_LEVEL,
            minDistance=C.KLT_MIN_DISTANCE, mask=mask, blockSize=3,
        )
        if corners is not None:
            additions.extend(corners.reshape(-1, 2))
        # Smooth or blurred heads can have no corner: retain a local fallback
        # sample so every detector entity continues to contribute motion data.
        found = 0 if corners is None else len(corners)
        if found < needed:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            offsets = ((0.0, 0.0), (-2.0, 0.0), (2.0, 0.0))
            additions.extend((cx + dx, cy + dy) for dx, dy in offsets[:needed - found])
    if additions:
        added_points = np.asarray(additions, dtype=np.float32).reshape(-1, 2)
        add_ids = np.arange(next_point_id, next_point_id + len(added_points), dtype=np.int64)
        points = np.vstack((points, added_points))
        point_ids = np.concatenate((point_ids, add_ids))
        next_point_id += len(added_points)
    return points, point_ids, next_point_id


def detect(model, frame):
    result = model(frame, conf=C.CONF, iou=C.IOU, imgsz=C.IMGSZ, max_det=C.MAX_DET,
                   device="cpu", verbose=False)[0]
    return (result.boxes.xyxy.cpu().numpy().astype(np.float32)
            if result.boxes is not None else np.empty((0, 4), np.float32))


def draw_overlay(frame, boxes, points, point_ids, previous_lookup, count):
    """Render head detections and KLT motion for a low-rate audit video."""
    canvas = frame.copy()
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
    for point, point_id in zip(points, point_ids):
        x, y = np.round(point / C.KLT_SCALE).astype(int)
        prior = previous_lookup.get(int(point_id))
        if prior is not None:
            px, py = np.round(prior / C.KLT_SCALE).astype(int)
            cv2.arrowedLine(canvas, (px, py), (x, y), (0, 220, 255), 1, tipLength=0.25)
        cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)
    cv2.putText(canvas, f"people (heads)={count}", (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, f"people (heads)={count}", (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    text = f"KLT motion samples={len(points)} (not people)"
    # Same thickness for both to avoid OpenCV character spacing bug on fractional font scales
    cv2.putText(canvas, text, (17, 71), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, text, (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    torch.set_num_threads(os.cpu_count() or 1)
    model = load_head_model()
    cap = cv2.VideoCapture(C.SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {C.SOURCE}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_total / fps if frame_total else 0.0

    # Ensure output directories exist
    os.makedirs(os.path.dirname(os.path.abspath(C.LOG_CSV)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(C.OUTPUT_VIDEO)), exist_ok=True)

    store = TrackStore()
    previous_gray = None
    points, point_ids = np.empty((0, 2), np.float32), np.empty(0, np.int64)
    next_point_id = 0
    last_detection_time = float("-inf")
    latest_detection = detection_features(np.empty((0, 4), np.float32), 1, 1)
    latest_boxes = np.empty((0, 4), np.float32)
    detect_every = C.DETECT_EVERY_N
    detection_frames = klt_frames = frame_index = 0
    klt_counts = []
    rows, last_log_time = [], float("-inf")
    wall_start = time.perf_counter()
    writer = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        gray = make_gray(frame)
        previous_lookup = {int(point_id): point.copy() for point_id, point in zip(point_ids, points)}
        points, point_ids = advance_klt(previous_gray, gray, points, point_ids)
        klt_frames += 1
        store.update_points(video_time, point_ids, points / C.KLT_SCALE)

        must_refresh_for_log = video_time - last_detection_time > C.MAX_COUNT_AGE
        scheduled_detection = frame_index == 1 or frame_index % detect_every == 0
        if scheduled_detection or must_refresh_for_log:
            boxes = detect(model, frame)
            latest_boxes = boxes
            latest_detection = detection_features(boxes, frame.shape[0], frame.shape[1])
            last_detection_time = video_time
            points, point_ids, next_point_id = reseed_klt(
                gray, points, point_ids, boxes, next_point_id
            )
            store.update_points(video_time, point_ids, points / C.KLT_SCALE)
            detection_frames += 1
        klt_counts.append(len(points))

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                C.OUTPUT_VIDEO, fourcc, fps,
                (frame.shape[1], frame.shape[0]),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create output video: {C.OUTPUT_VIDEO}")
        if frame_index % C.OUTPUT_EVERY_N == 0:
            # Boxes remain detector boxes; KLT exists only as a motion overlay.
            writer.write(draw_overlay(frame, latest_boxes, points, point_ids, previous_lookup,
                                      latest_detection["count"]))

        if video_time - last_log_time >= C.LOG_INTERVAL:
            # Detection is refreshed above if needed: count is always <=0.5 s old.
            row = {**latest_detection, **motion_features(store, point_ids)}
            rows.append([video_time] + [row[column] for column in FEATURE_COLS])
            last_log_time = video_time

        if video_time >= 5.0:
            wall_per_video = (time.perf_counter() - wall_start) / max(video_time, 1e-6)
            if wall_per_video > 0.8 and detect_every < C.DETECT_EVERY_N_MAX:
                detect_every = min(detect_every * 2, C.DETECT_EVERY_N_MAX)
        previous_gray = gray

    cap.release()
    if writer is not None:
        writer.release()
    wall_time = time.perf_counter() - wall_start
    with open(C.LOG_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["t"] + FEATURE_COLS)
        writer.writerows(rows)
    ratio = wall_time / duration if duration else 0.0
    klt_min = min(klt_counts) if klt_counts else 0
    klt_median = int(np.median(klt_counts)) if klt_counts else 0
    print(
        f"benchmark: wall={wall_time:.2f}s video={duration:.2f}s ratio={ratio:.3f} "
        f"detection_frames={detection_frames} klt_frames={klt_frames} rows={len(rows)} "
        f"klt_min={klt_min} klt_median={klt_median} video={C.OUTPUT_VIDEO}"
    )

# --- NEW AUTOMATION CODE BEGINS HERE ---
    import subprocess
    print("\n--- Generating ML Metrics (metric.csv) ---")
    try:
        # sys.executable ensures it uses your active (venv) Python
        subprocess.run([sys.executable, "tools/log_to_metrics.py"], check=True)
    except Exception as e:
        print(f"Failed to generate metric.csv automatically: {e}")
    # --- NEW AUTOMATION CODE ENDS HERE ---

if __name__ == "__main__":
    main()
