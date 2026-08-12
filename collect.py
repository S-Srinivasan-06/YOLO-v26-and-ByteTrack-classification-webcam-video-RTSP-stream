# collect.py
"""STEP 1 — run YOLO26 + ByteTrack, log 1 Hz feature rows to log.csv."""
import time, csv
import cv2
import numpy as np
from ultralytics import YOLO
import config as C
from features import TrackStore, FeatureBuffer, frame_features, FEATURE_COLS


def main():
    model = YOLO(C.MODEL_NAME)
    cap = cv2.VideoCapture(C.SOURCE)
    is_video = isinstance(C.SOURCE, str)
    store, buf = TrackStore(), FeatureBuffer(C.HIST_WINDOW)
    H = W = 1
    last_log, rows = 0.0, []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if is_video else time.time()

        res = model.track(frame, persist=True, tracker=C.TRACKER, verbose=False)[0]
        boxes, ids = np.empty((0, 4)), np.empty(0, int)
        if res.boxes is not None and res.boxes.id is not None:
            m = res.boxes.cls.cpu().numpy() == C.PERSON_CLASS
            boxes = res.boxes.xyxy.cpu().numpy()[m]
            ids = res.boxes.id.cpu().numpy()[m].astype(int)

        store.update(t, boxes, ids)
        feat = frame_features(t, boxes, ids, store, H, W)
        buf.push(feat, t)

        if t - last_log >= C.LOG_INTERVAL:
            rows.append([t] + buf.vector(t))
            last_log = t

        if C.SHOW:
            for b in boxes.astype(int):
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cv2.putText(frame, f"n={feat['count']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("collect", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if C.SHOW:
        cv2.destroyAllWindows()
    with open(C.LOG_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t"] + FEATURE_COLS)
        w.writerows(rows)
    print(f"saved {len(rows)} rows -> {C.LOG_CSV}")


if __name__ == "__main__":
    main()
