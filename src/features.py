"""CrowdShield feature extraction — KLT-free high-speed build with Track IDs."""
import math
import time
import random
from collections import defaultdict, deque

import cv2
import numpy as np

try:
    import src.config as C
except ImportError:
    import config as C

EPS = 1e-6


# ═══════════════════════════════════════════════════════
# 1. ROLLING METRICS (scoped to ROLLING_KEYS only)
# ═══════════════════════════════════════════════════════
class RollingMetrics:
    def __init__(self, window_size=C.ROLLING_WINDOW, keys=C.ROLLING_KEYS):
        self.window = window_size
        self.keys = set(keys)
        self.history = defaultdict(lambda: deque(maxlen=window_size))

    def update(self, raw):
        out = {}
        for k, v in raw.items():
            out[k] = v
            if k not in self.keys or not isinstance(v, (int, float)) or math.isnan(v):
                continue
            h = self.history[k]
            h.append(v)
            if len(h) >= 2:
                sma = sum(h) / len(h)
                var = sum((x - sma) ** 2 for x in h) / (len(h) - 1)
                std = math.sqrt(var)
                out[k + "_sma"] = round(sma, 4)
                out[k + "_std"] = round(std, 4)
                out[k + "_z"] = round((v - sma) / (std + EPS), 4)
            else:
                out[k + "_sma"] = round(v, 4)
                out[k + "_std"] = 0.0
                out[k + "_z"] = 0.0
        return out


# ═══════════════════════════════════════════════════════
# 2. PERSPECTIVE SCALE FIELD S(y)
# ═══════════════════════════════════════════════════════
class PerspectiveGridEstimator:
    def __init__(self, rows=C.GRID_ROWS, alpha=0.1):
        self.rows = rows
        self.alpha = alpha
        self.baseline = np.linspace(
            C.MIN_HEAD_HEIGHT * 1.5, C.MIN_HEAD_HEIGHT * 4.0, rows
        )

    def update_and_get(self, heights, centers_y, frame_h):
        bins = np.linspace(0, frame_h, self.rows + 1)
        for i in range(self.rows):
            mask = (centers_y >= bins[i]) & (centers_y < bins[i + 1])
            if np.any(mask):
                obs = float(np.median(heights[mask]))
                self.baseline[i] = (1 - self.alpha) * self.baseline[i] + self.alpha * obs
        return np.maximum(self.baseline, C.MIN_HEAD_HEIGHT)


# ═══════════════════════════════════════════════════════
# 3. FAST KINEMATIC CENTROID TRACKER (with Multi-Frame Coasting)
# ═══════════════════════════════════════════════════════
class Track:
    """Individual track state with velocity prediction and occlusion memory."""
    def __init__(self, track_id, center, color, current_time):
        self.track_id = track_id
        self.center = center.astype(np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.color = color
        self.last_time = current_time
        self.misses = 0
        self.hits = 1

    def predict(self, dt):
        return self.center + self.velocity * dt


class FastCentroidTracker:
    """Scale-normalized bipartite centroid matching with Track Coasting.

    Key Features:
      1. Kinematic Lookahead: Projects track positions forward using prior velocity.
      2. Multi-Frame Coasting (max_misses = 6 ticks ~1.5s): Survives temporary head occlusions,
         lighting shifts, and detection dropouts in dense crowds without losing track identity.
      3. Velocity Smoothing: 80% instantaneous, 20% momentum for stable physical vectors.
      4. Dynamic Search Cones: Expands search radius for fast movers proportional to speed.
    """

    def __init__(self, max_displacement_scale=C.TRACK_MAX_DISPLACEMENT_SCALE, max_misses=6):
        self.tracks = {}  # track_id -> Track
        self.max_scale = max_displacement_scale
        self.max_misses = max_misses
        self.next_id = 0
        self.prev_time = None
        self.colors = {}

    def _get_color(self, track_id):
        if track_id not in self.colors:
            self.colors[track_id] = tuple(random.randint(80, 255) for _ in range(3))
        return self.colors[track_id]

    def update(self, centers, scales, current_time):
        # Re-read from config so trackbar / web slider changes take effect live
        self.max_scale = getattr(C, "TRACK_MAX_DISPLACEMENT_SCALE", self.max_scale)
        n = len(centers)

        if self.prev_time is None:
            self.prev_time = current_time
            ids = np.empty(n, dtype=np.int64)
            velocities = np.zeros((n, 2), dtype=np.float32)
            for i in range(n):
                tid = self.next_id
                self.next_id += 1
                color = self._get_color(tid)
                self.tracks[tid] = Track(tid, centers[i], color, current_time)
                ids[i] = tid
            return velocities, ids

        dt = current_time - self.prev_time
        self.prev_time = current_time

        if dt <= 0.001 or dt > 2.0:
            dt = 0.25

        active_track_ids = list(self.tracks.keys())
        m = len(active_track_ids)

        if m == 0:
            ids = np.empty(n, dtype=np.int64)
            velocities = np.zeros((n, 2), dtype=np.float32)
            for i in range(n):
                tid = self.next_id
                self.next_id += 1
                color = self._get_color(tid)
                self.tracks[tid] = Track(tid, centers[i], color, current_time)
                ids[i] = tid
            return velocities, ids

        if n == 0:
            # Coast all existing tracks forward
            for tid in active_track_ids:
                trk = self.tracks[tid]
                trk.center += trk.velocity * dt
                trk.misses += 1
                if trk.misses > self.max_misses:
                    del self.tracks[tid]
            return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int64)

        # ── Predict positions for all active tracks ──
        predicted_centers = np.array([self.tracks[tid].predict(dt) for tid in active_track_ids], dtype=np.float32)
        prev_speeds = np.array([np.linalg.norm(self.tracks[tid].velocity) for tid in active_track_ids], dtype=np.float32)

        # Pairwise distance matrix: N_detections × M_tracks
        diff = centers[:, np.newaxis, :] - predicted_centers[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)

        # Adaptive search threshold per detection-track pair
        base_allowed = scales[:, np.newaxis] * self.max_scale
        speed_cushion = prev_speeds[np.newaxis, :] * dt * 0.5
        max_allowed_dist = np.maximum(base_allowed, base_allowed * 0.7 + speed_cushion)

        matched_cur = set()
        matched_trk = set()
        det_ids = np.full(n, -1, dtype=np.int64)
        det_velocities = np.zeros((n, 2), dtype=np.float32)

        # Greedy bipartite matching
        sorted_indices = np.dstack(
            np.unravel_index(np.argsort(dist_matrix.ravel()), dist_matrix.shape)
        )[0]

        for cur_idx, trk_idx in sorted_indices:
            if cur_idx in matched_cur or trk_idx in matched_trk:
                continue
            if dist_matrix[cur_idx, trk_idx] <= max_allowed_dist[cur_idx, trk_idx]:
                matched_cur.add(cur_idx)
                matched_trk.add(trk_idx)
                
                tid = active_track_ids[trk_idx]
                trk = self.tracks[tid]
                
                # Displacement from actual prior location
                inst_v = (centers[cur_idx] - trk.center) / (dt * (trk.misses + 1))
                
                # Velocity smoothing
                trk.velocity = 0.8 * inst_v + 0.2 * trk.velocity
                trk.center = centers[cur_idx].astype(np.float32)
                trk.misses = 0
                trk.hits += 1
                trk.last_time = current_time
                
                det_ids[cur_idx] = tid
                det_velocities[cur_idx] = trk.velocity

        # Coast unmatched tracks
        for trk_idx, tid in enumerate(active_track_ids):
            if trk_idx not in matched_trk:
                trk = self.tracks[tid]
                trk.center += trk.velocity * dt
                trk.misses += 1
                if trk.misses > self.max_misses:
                    del self.tracks[tid]

        # Spawn new tracks for unmatched detections
        for i in range(n):
            if det_ids[i] == -1:
                tid = self.next_id
                self.next_id += 1
                color = self._get_color(tid)
                self.tracks[tid] = Track(tid, centers[i], color, current_time)
                det_ids[i] = tid
                det_velocities[i] = np.zeros(2, dtype=np.float32)

        # Prune colors
        if len(self.colors) > 3000:
            active_ids = set(self.tracks.keys())
            self.colors = {k: v for k, v in self.colors.items() if k in active_ids}

        return det_velocities, det_ids




# ═══════════════════════════════════════════════════════
# 4. SCALE-ADAPTIVE KDE (cached kernels)
# ═══════════════════════════════════════════════════════
class ScaleAdaptiveKDE:
    def __init__(self, frame_shape, scale_down=C.KDE_SCALE_DOWN, gamma=C.KDE_GAMMA):
        self.oh, self.ow = frame_shape[:2]
        self.sd = scale_down
        self.h = self.oh // scale_down
        self.w = self.ow // scale_down
        self.gamma = gamma
        self._kcache = {}

    def _kernel(self, sigma):
        key = max(round(sigma * 2.0) / 2.0, 1.0)
        hit = self._kcache.get(key)
        if hit is None:
            r = int(np.ceil(3.0 * key))
            ax = np.arange(-r, r + 1, dtype=np.float32)
            g = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2.0 * key * key))
            hit = (r, g.astype(np.float32))
            self._kcache[key] = hit
        return hit

    def splat(self, density, vx, vy, v2, centers, scales, velocities):
        # Re-read from config so trackbar / web slider changes take effect live
        self.gamma = getattr(C, "KDE_GAMMA", self.gamma)
        sc = centers / self.sd
        sig = np.maximum(scales * self.gamma / self.sd, 1.5)
        W, H = self.w, self.h
        for i in range(len(sc)):
            r, g = self._kernel(float(sig[i]))
            xi, yi = int(round(sc[i][0])), int(round(sc[i][1]))
            x0, y0 = xi - r, yi - r
            x1, y1 = xi + r + 1, yi + r + 1
            sx0, sy0 = max(0, -x0), max(0, -y0)
            ex, ey = min(W, x1) - x0, min(H, y1) - y0
            if ex <= sx0 or ey <= sy0:
                continue
            patch = g[sy0:ey, sx0:ex]
            ph, pw = patch.shape
            X, Y = max(0, x0), max(0, y0)
            density[Y:Y + ph, X:X + pw] += patch
            u, v = velocities[i]
            vx[Y:Y + ph, X:X + pw] += patch * u
            vy[Y:Y + ph, X:X + pw] += patch * v
            v2[Y:Y + ph, X:X + pw] += patch * (u * u + v * v)


# ═══════════════════════════════════════════════════════
# 5. CROWD DYNAMICS ENGINE
# ═══════════════════════════════════════════════════════
class CrowdDynamicsEngine:
    def __init__(self, frame_shape):
        self.kde = ScaleAdaptiveKDE(frame_shape)
        self.oh, self.ow = frame_shape[:2]

    def compute(self, centers, scales, velocities):
        h, w = self.kde.h, self.kde.w
        density = np.zeros((h, w), np.float32)
        if len(centers) == 0:
            z = density.copy()
            return density, z, z, z, z, z, z
        vx = np.zeros((h, w), np.float32)
        vy = np.zeros((h, w), np.float32)
        v2 = np.zeros((h, w), np.float32)
        self.kde.splat(density, vx, vy, v2, centers, scales, velocities)
        U = vx / (density + EPS)
        V = vy / (density + EPS)
        var_v = np.maximum(v2 / (density + EPS) - (U * U + V * V), 0.0)
        Mx = density * U
        My = density * V
        momentum = density * np.sqrt(U * U + V * V)
        pressure = density * var_v
        return density, U, V, Mx, My, momentum, pressure

    @staticmethod
    def metrics(density, Mx, My, momentum, pressure):
        peak = float(np.max(density)) if density.size else 0.0
        total = float(np.sum(density))
        danger_px = float(np.sum(density >= C.DANGER_THRESH))
        danger_pct = (danger_px / density.size * 100.0) if density.size else 0.0
        if total > 0:
            prob = density.ravel() / total
            prob = prob[prob > 0]
            entropy = float(-np.sum(prob * np.log2(prob + EPS)))
        else:
            entropy = 0.0
        max_p = float(np.max(pressure)) if pressure.size else 0.0
        mean_m = float(np.mean(momentum)) if momentum.size else 0.0
        turb = float(np.sum(pressure) / (total + EPS))
        m_total = float(np.sum(momentum))
        if m_total > EPS:
            net_mx, net_my = float(np.sum(Mx)), float(np.sum(My))
            coherence = min(math.sqrt(net_mx ** 2 + net_my ** 2) / (m_total + EPS), 1.0)
        else:
            coherence = 0.0
        return {
            "peak_density":    round(peak, 3),
            "danger_area_pct": round(danger_pct, 2),
            "crowd_entropy":   round(entropy, 3),
            "max_pressure":    round(max_p, 3),
            "mean_momentum":   round(mean_m, 4),
            "turbulence_index": round(turb, 4),
            "flow_coherence":  round(coherence, 4),
        }


# ═══════════════════════════════════════════════════════
# 6. DUAL-GATED 5s ALERT EVALUATOR
# ═══════════════════════════════════════════════════════
class StampedeRiskClassifier:
    """5-second smoothed, physics-gated stampede hazard classifier.

    Two mandatory gates must both pass before any score is emitted:
      1. Absolute count gate  — crowd too sparse → always NORMAL (no false alarms)
      2. Physics gate         — pressure AND density both negligible → NORMAL

    Risk score components:
      s_score = spatial density / RISK_DENS_CRIT          (how compressed)
      d_score = (pressure / RISK_P_CRIT)
                × (1 - flow_coherence)                    (chaotic momentum)
                × (1 + turbulence / RISK_TURB_CRIT)       (amplifier)

    Final score passed through a logistic (sigmoid) centred at 0.45
    so that the transition from WATCH→WARNING is steep and hysteresis-free.
    """

    def __init__(self):
        steps = max(1, int(C.RISK_WINDOW_SEC / C.LOG_INTERVAL))
        self.hist_count = deque(maxlen=steps)
        self.hist_p     = deque(maxlen=steps)
        self.hist_dens  = deque(maxlen=steps)
        self.hist_turb  = deque(maxlen=steps)
        self.hist_coh   = deque(maxlen=steps)

    def evaluate(self, metrics: dict) -> dict:
        # Use rolling SMA features directly (falling back to instantaneous if startup)
        count_sma = float(metrics.get("count_sma", metrics.get("count", 0)))
        p_sma     = float(metrics.get("max_pressure_sma", metrics.get("max_pressure", 0.0)))
        dens_sma  = float(metrics.get("peak_density_sma", metrics.get("peak_density", 0.0)))
        turb_sma  = float(metrics.get("turbulence_index_sma", metrics.get("turbulence_index", 0.0)))
        coh_sma   = float(metrics.get("flow_coherence_sma", metrics.get("flow_coherence", 0.0)))

        base = {"p_sma5s": round(p_sma, 1), "dens_sma5s": round(dens_sma, 2)}

        # ── Gate 1: minimum crowd size (evaluated on moving average count) ──
        if count_sma < C.RISK_COUNT_FLOOR:
            return {
                "risk_score_5s": 0.0,
                "risk_score_sma": 0.0,
                "alert_stage": 0,
                "alert_name": "NORMAL",
                **base,
            }

        # ── Gate 2: negligible physics signal ───────────────────────────
        if p_sma < C.RISK_P_CRIT * 0.2 and dens_sma < 3.0:
            return {
                "risk_score_5s": 0.0,
                "risk_score_sma": 0.0,
                "alert_stage": 0,
                "alert_name": "NORMAL",
                **base,
            }

        # ── Dimensionless risk scoring calculated on moving averages ────
        s_score = min(dens_sma / C.RISK_DENS_CRIT, 1.0)
        d_score = (
            min(p_sma / C.RISK_P_CRIT, 1.0)
            * (1.0 - coh_sma)
            * min(1.0 + turb_sma / C.RISK_TURB_CRIT, 2.0)
        )
        raw_risk = 0.35 * s_score + 0.65 * d_score

        # Logistic sigmoid centered at RISK_SIGMOID_MID
        mid = getattr(C, "RISK_SIGMOID_MID", 0.45)
        risk_score_sma = float(1.0 / (1.0 + math.exp(-C.RISK_SIGMOID_K * (raw_risk - mid))))

        if   risk_score_sma >= 0.80: stage, name = 3, "CRITICAL"
        elif risk_score_sma >= 0.55: stage, name = 2, "WARNING"
        elif risk_score_sma >= 0.30: stage, name = 1, "WATCH"
        else:                        stage, name = 0, "NORMAL"

        return {
            "risk_score_5s":  round(risk_score_sma, 4),
            "risk_score_sma": round(risk_score_sma, 4),
            "alert_stage":    stage,
            "alert_name":     name,
            **base,
        }



# ═══════════════════════════════════════════════════════
# 7. MASTER FEATURE EXTRACTION (Scale-Adaptive Gating)
# ═══════════════════════════════════════════════════════
def extract_features(boxes, confs, frame_h, frame_w, rolling, grid_est, tracker, dynamics, cur_t):
    sys_time = time.time()

    def empty():
        z = np.zeros((dynamics.kde.h, dynamics.kde.w), np.float32)
        raw = {
            "sys_time": sys_time, "count": 0, "density_score": 0.0,
            "peak_density": 0.0, "danger_area_pct": 0.0, "crowd_entropy": 0.0,
            "max_pressure": 0.0, "mean_momentum": 0.0,
            "turbulence_index": 0.0, "flow_coherence": 0.0,
        }
        return (rolling.update(raw), z, z.copy(), z.copy(), z.copy(),
                np.empty(0, np.int64), np.empty((0, 2), np.float32), np.empty((0, 4), np.float32))

    if len(boxes) == 0:
        tracker.update(np.empty((0, 2), np.float32), np.empty(0, np.float32), cur_t)
        return empty()

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    aspect_ratios = widths / (heights + EPS)

    # ── Scale-Adaptive Confidence Gating ────────────────────────────────
    # Tiny distant heads (h <= 10px in livestreams/CCTV) use high-recall threshold (CONF ~ 0.02)
    # Large close-up heads (h >= 100px in webcam/room views) require high confidence (0.22 - 0.50)
    # This automatically eliminates phantom furniture/background detections on webcams.
    h_norm = np.clip((heights - 10.0) / 90.0, 0.0, 1.0)
    adaptive_min_conf = np.maximum(C.CONF, 0.02 + 0.20 * h_norm)

    valid = (
        (heights >= C.MIN_HEAD_HEIGHT) &
        (aspect_ratios >= 0.40) & (aspect_ratios <= 2.30) &
        (confs >= adaptive_min_conf)
    )
    boxes = boxes[valid]
    heights = heights[valid]


    count = len(boxes)
    if count == 0:
        tracker.update(np.empty((0, 2), np.float32), np.empty(0, np.float32), cur_t)
        return empty()

    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    centers = np.column_stack((cx, cy))

    row_h = grid_est.update_and_get(heights, cy, frame_h)
    grid_y = np.clip((cy / frame_h * C.GRID_ROWS).astype(int), 0, C.GRID_ROWS - 1)
    scales = row_h[grid_y]

    density_score = float(np.sum((scales / row_h[-1]) ** 2))
    velocities, track_ids = tracker.update(centers, scales, cur_t)

    dens, U, V, Mx, My, mom, pres = dynamics.compute(centers, scales, velocities)
    dyn = dynamics.metrics(dens, Mx, My, mom, pres)

    raw = {
        "sys_time": sys_time,
        "count": count,
        "density_score": round(density_score, 3),
        **dyn,
    }
    return rolling.update(raw), dens, U, V, pres, track_ids, velocities, boxes

