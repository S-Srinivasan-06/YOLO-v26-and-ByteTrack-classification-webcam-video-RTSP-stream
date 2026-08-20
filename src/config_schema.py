"""CrowdShield Parameter Schema, Dynamic Validation, and Presets."""
from typing import Any, Dict

PARAM_SCHEMA = {
    # ── Detection ──
    "CONF": {
        "type": "float",
        "min": 0.001,
        "max": 1.0,
        "default": 0.02,
        "group": "Detection",
        "desc": "Baseline detection confidence floor (0.001 - 1.000)",
    },
    "IOU": {
        "type": "float",
        "min": 0.01,
        "max": 1.0,
        "default": 0.60,
        "group": "Detection",
        "desc": "NMS Intersection-over-Union threshold",
    },
    "IMGSZ": {
        "type": "int",
        "allowed": [640, 768, 960, 1280, 1536, 1920],
        "default": 1280,
        "group": "Detection",
        "desc": "YOLO inference resolution",
    },
    "MAX_DET": {
        "type": "int",
        "min": 10,
        "max": 3000,
        "default": 1500,
        "group": "Detection",
        "desc": "Maximum allowed head detections per frame",
    },
    "MIN_HEAD_HEIGHT": {
        "type": "float",
        "min": 1.0,
        "max": 50.0,
        "default": 2.0,
        "group": "Detection",
        "desc": "Minimum head box pixel height",
    },
    "ENABLE_ADAPTIVE_CONF": {
        "type": "bool",
        "default": True,
        "group": "Detection",
        "desc": "Scale-adaptive confidence floor for close-up vs distant heads",
    },
    # ── Tracking ──
    "TRACK_MAX_DISPLACEMENT_SCALE": {
        "type": "float",
        "min": 0.5,
        "max": 10.0,
        "default": 4.0,
        "group": "Tracking",
        "desc": "Centroid search radius multiplier relative to head scale",
    },
    # ── Density & Physics ──
    "KDE_GAMMA": {
        "type": "float",
        "min": 0.1,
        "max": 3.0,
        "default": 0.8,
        "group": "Density & Physics",
        "desc": "Gaussian kernel standard deviation factor",
    },
    "KDE_SCALE_DOWN": {
        "type": "int",
        "allowed": [1, 2, 4, 8],
        "default": 4,
        "group": "Density & Physics",
        "desc": "Downscaling factor for continuous grid splatting",
    },
    "DANGER_THRESH": {
        "type": "float",
        "min": 0.5,
        "max": 10.0,
        "default": 3.0,
        "group": "Density & Physics",
        "desc": "Density threshold considered hazardous",
    },
    # ── Stampede Risk Classifier ──
    "RISK_COUNT_FLOOR": {
        "type": "int",
        "min": 0,
        "max": 500,
        "default": 35,
        "group": "Risk Model",
        "desc": "Minimum heads required to trigger Watch/Warning gates",
    },
    "RISK_P_CRIT": {
        "type": "float",
        "min": 10000.0,
        "max": 2000000.0,
        "default": 450000.0,
        "group": "Risk Model",
        "desc": "Helbing pressure saturation ceiling",
    },
    "RISK_DENS_CRIT": {
        "type": "float",
        "min": 1.0,
        "max": 50.0,
        "default": 18.0,
        "group": "Risk Model",
        "desc": "Continuous density crush limit",
    },
    "RISK_TURB_CRIT": {
        "type": "float",
        "min": 1000.0,
        "max": 100000.0,
        "default": 25000.0,
        "group": "Risk Model",
        "desc": "Momentum turbulence saturation factor",
    },
    "RISK_SIGMOID_K": {
        "type": "float",
        "min": 1.0,
        "max": 20.0,
        "default": 6.0,
        "group": "Risk Model",
        "desc": "Logistic curve transition steepness",
    },
    "RISK_SIGMOID_MID": {
        "type": "float",
        "min": 0.1,
        "max": 0.9,
        "default": 0.45,
        "group": "Risk Model",
        "desc": "Logistic curve midpoint",
    },
    # ── Runtime & Recording ──
    "MAX_RUNTIME_SEC": {
        "type": "float",
        "min": 30.0,
        "max": 864000.0,
        "default": 86400.0,  # Unlimited / 24h default
        "group": "Runtime",
        "desc": "Session auto-termination timeout in seconds",
    },
    "SAVE_LIVE_VIDEO": {
        "type": "bool",
        "default": True,
        "group": "Runtime",
        "desc": "Record annotated video artifact",
    },
}

PRESETS = {
    "High-Recall CCTV": {
        "CONF": 0.02,
        "IOU": 0.60,
        "IMGSZ": 1280,
        "MIN_HEAD_HEIGHT": 2.0,
        "TRACK_MAX_DISPLACEMENT_SCALE": 4.0,
        "KDE_GAMMA": 0.8,
        "RISK_COUNT_FLOOR": 35,
    },
    "Dense Crowd / Chokepoint": {
        "CONF": 0.05,
        "IOU": 0.50,
        "IMGSZ": 1280,
        "MIN_HEAD_HEIGHT": 2.0,
        "TRACK_MAX_DISPLACEMENT_SCALE": 5.5,
        "KDE_GAMMA": 1.0,
        "RISK_COUNT_FLOOR": 50,
        "RISK_P_CRIT": 400000.0,
    },
    "Webcam / Close-Range": {
        "CONF": 0.35,
        "IOU": 0.65,
        "IMGSZ": 640,
        "MIN_HEAD_HEIGHT": 15.0,
        "TRACK_MAX_DISPLACEMENT_SCALE": 3.0,
        "KDE_GAMMA": 0.6,
        "RISK_COUNT_FLOOR": 1,
    },
    "High-Precision (Strict)": {
        "CONF": 0.50,
        "IOU": 0.70,
        "IMGSZ": 1280,
        "MIN_HEAD_HEIGHT": 5.0,
        "TRACK_MAX_DISPLACEMENT_SCALE": 3.5,
        "KDE_GAMMA": 0.8,
    },
}


def validate_and_sanitize(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Validate incoming parameter updates against the schema."""
    cleaned = {}
    for k, val in updates.items():
        if k not in PARAM_SCHEMA:
            continue
        spec = PARAM_SCHEMA[k]
        try:
            if spec["type"] == "float":
                v = float(val)
                if "min" in spec and v < spec["min"]:
                    continue
                if "max" in spec and v > spec["max"]:
                    continue
                cleaned[k] = v
            elif spec["type"] == "int":
                v = int(val)
                if "allowed" in spec and v not in spec["allowed"]:
                    continue
                if "min" in spec and v < spec["min"]:
                    continue
                if "max" in spec and v > spec["max"]:
                    continue
                cleaned[k] = v
            elif spec["type"] == "bool":
                cleaned[k] = bool(val)
        except (ValueError, TypeError):
            continue
    return cleaned
