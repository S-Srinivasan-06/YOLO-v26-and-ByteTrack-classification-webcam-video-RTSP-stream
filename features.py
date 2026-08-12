# features.py
"""Uncalibrated crowd features from YOLO person boxes + track IDs."""
import numpy as np
from collections import deque
from config import GRID

FEATURE_COLS = (
    ["count", "cov", "wcount", "speed", "dirx", "diry", "consist"]
    + [f"lag{s}" for s in (10, 30, 60, 120)]
    + ["trend"]
    + [f"g{i}" for i in range(GRID * GRID)]
)
LAGS = (10, 30, 60, 120)


class TrackStore:
    """Keeps short centroid history per track ID for velocity estimation."""

    def __init__(self, maxlen=30, max_age=10.0):
        self.tracks, self.last_seen = {}, {}
        self.maxlen, self.max_age = maxlen, max_age

    def update(self, t, boxes, ids):
        for b, tid in zip(boxes, ids):
            dq = self.tracks.get(tid)
            if dq is None:
                dq = self.tracks[tid] = deque(maxlen=self.maxlen)
            dq.append((t, (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0))
            self.last_seen[tid] = t
        for k in [k for k, ts in self.last_seen.items() if t - ts > self.max_age]:
            del self.tracks[k], self.last_seen[k]

    def velocity(self, tid, min_dt=0.2):
        h = self.tracks.get(tid)
        if not h or len(h) < 2:
            return None
        t0, x0, y0 = h[0]
        t1, x1, y1 = h[-1]
        dt = t1 - t0
        if dt < min_dt:
            return None
        return (x1 - x0) / dt, (y1 - y0) / dt


def frame_features(t, boxes, ids, store, H, W):
    """Scalar + grid features for one frame. No camera calibration needed."""
    f = {"count": len(boxes)}
    if len(boxes) == 0:
        f.update(cov=0.0, wcount=0.0, speed=0.0, dirx=0.0, diry=0.0, consist=0.0)
        f.update({f"g{i}": 0 for i in range(GRID * GRID)})
        return f

    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0

    # density proxies (uncalibrated)
    f["cov"]    = float(((boxes[:, 2] - boxes[:, 0]) *
                         (boxes[:, 3] - boxes[:, 1])).sum() / (H * W))
    f["wcount"] = float((cy / H).sum())          # depth-weighted count

    # 4x4 spatial grid counts
    g = np.zeros(GRID * GRID)
    gx = np.clip((cx / W * GRID).astype(int), 0, GRID - 1)
    gy = np.clip((cy / H * GRID).astype(int), 0, GRID - 1)
    for x, y in zip(gx, gy):
        g[y * GRID + x] += 1
    f.update({f"g{i}": float(g[i]) for i in range(GRID * GRID)})

    # crowd velocity (pixel space; valid for fixed camera)
    v = [store.velocity(tid) for tid in ids]
    v = np.array([x for x in v if x is not None])
    if len(v):
        s  = np.linalg.norm(v, axis=1)
        uv = v / (s[:, None] + 1e-6)
        f.update(speed=float(s.mean()),
                 dirx=float(uv[:, 0].mean()),
                 diry=float(uv[:, 1].mean()),
                 consist=float(np.linalg.norm(uv.mean(axis=0))))  # 0 chaos..1 orderly
    else:
        f.update(speed=0.0, dirx=0.0, diry=0.0, consist=0.0)
    return f


class FeatureBuffer:
    """Rolling history; flattens it into one model-ready vector."""

    def __init__(self, window):
        self.hist, self.window = deque(), window

    def push(self, feat, t):
        self.hist.append({**feat, "t": t})
        while self.hist and t - self.hist[0]["t"] > self.window:
            self.hist.popleft()

    def ready(self):
        return len(self.hist) >= 5

    def vector(self, t):
        now = self.hist[-1]

        def past(sec):
            for f in reversed(self.hist):
                if t - f["t"] >= sec:
                    return f
            return self.hist[0]

        v  = [now[k] for k in ("count", "cov", "wcount", "speed", "dirx", "diry", "consist")]
        v += [past(s)["count"] for s in LAGS]
        v += [now["count"] - past(60)["count"]]                 # 1-min trend
        v += [now[f"g{i}"] for i in range(GRID * GRID)]
        return v
