"""Detection density features and KLT-only motion features."""
from collections import deque

import numpy as np

try:
    from src.config import GRID
except ImportError:
    from config import GRID

FEATURE_COLS = (
    ["count", "cov", "wcount", "speed", "dirx", "diry", "consist"]
    + [f"g{i}" for i in range(GRID * GRID)]
)


class TrackStore:
    """Centroid history for KLT points; never used to determine the count."""

    def __init__(self, maxlen=30, max_age=10.0):
        self.tracks, self.last_seen = {}, {}
        self.maxlen, self.max_age = maxlen, max_age

    def update_points(self, t, point_ids, points):
        for point_id, (x, y) in zip(point_ids, points):
            self.tracks.setdefault(int(point_id), deque(maxlen=self.maxlen)).append(
                (t, float(x), float(y))
            )
            self.last_seen[int(point_id)] = t
        stale = [key for key, seen in self.last_seen.items() if t - seen > self.max_age]
        for key in stale:
            del self.tracks[key]
            del self.last_seen[key]

    def velocity(self, point_id, min_dt=0.2):
        history = self.tracks.get(int(point_id))
        if history is None or len(history) < 2:
            return None
        t0, x0, y0 = history[0]
        t1, x1, y1 = history[-1]
        dt = t1 - t0
        if dt < min_dt:
            return None
        return (x1 - x0) / dt, (y1 - y0) / dt


def detection_features(boxes, height, width):
    """Count and spatial values from the most recent head-detector boxes."""
    feature = {"count": int(len(boxes))}
    if len(boxes) == 0:
        feature.update(cov=0.0, wcount=0.0)
        feature.update({f"g{i}": 0.0 for i in range(GRID * GRID)})
        return feature

    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    feature["cov"] = float(areas.sum() / (height * width))
    feature["wcount"] = float((centers_y / height).sum())

    grid = np.zeros(GRID * GRID, dtype=float)
    grid_x = np.clip((centers_x / width * GRID).astype(int), 0, GRID - 1)
    grid_y = np.clip((centers_y / height * GRID).astype(int), 0, GRID - 1)
    np.add.at(grid, grid_y * GRID + grid_x, 1)
    feature.update({f"g{i}": float(grid[i]) for i in range(GRID * GRID)})
    return feature


def motion_features(store, live_ids):
    """Speed and direction from KLT point displacement history only."""
    velocities = [store.velocity(point_id) for point_id in live_ids]
    velocities = np.asarray([v for v in velocities if v is not None], dtype=float)
    if len(velocities) == 0:
        return {"speed": 0.0, "dirx": 0.0, "diry": 0.0, "consist": 0.0}
    speed = np.linalg.norm(velocities, axis=1)
    unit = velocities / (speed[:, None] + 1e-6)
    mean_unit = unit.mean(axis=0)
    return {
        "speed": float(speed.mean()),
        "dirx": float(mean_unit[0]),
        "diry": float(mean_unit[1]),
        "consist": float(np.linalg.norm(mean_unit)),
    }
