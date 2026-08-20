"""Interactive OpenCV controls: live trackbars, draggable panels, source menu."""
import threading
import cv2

try:
    import src.config as C
except ImportError:
    import config as C

WINDOW = "CrowdShield Live"
_LOCK = threading.Lock()

UI = {
    "hud_pos": [12, 12],
    "param_pos": [12, 260],
    "hud_rect": (0, 0, 0, 0),
    "param_rect": (0, 0, 0, 0),
    "drag": None,
    "frame_size": (1280, 720),
}


def set_param(name, value):
    with _LOCK:
        setattr(C, name, value)


def get_param(name):
    with _LOCK:
        return getattr(C, name, None)


# (label, config attr, max_raw, to_value, to_raw)
_TRACKBARS = [
    ("CONF%",     "CONF",                         50,  lambda r: r / 100.0, lambda v: int(v * 100)),
    ("MINHEAD",   "MIN_HEAD_HEIGHT",              20,  float,               int),
    ("TRACKx10",  "TRACK_MAX_DISPLACEMENT_SCALE", 100, lambda r: r / 10.0,  lambda v: int(v * 10)),
    ("GAMMAx10",  "KDE_GAMMA",                    20,  lambda r: r / 10.0,  lambda v: int(v * 10)),
    ("PCRIT/10k", "RISK_P_CRIT",                  100, lambda r: r * 10000.0, lambda v: int(v / 10000)),
    ("DENSCRIT",  "RISK_DENS_CRIT",               40,  float,               int),
    ("COUNTFLR",  "RISK_COUNT_FLOOR",             200, float,               int),
]


def _view_cb(raw):
    set_param("DISPLAY_SCALE", float(raw))
    fw, fh = UI["frame_size"]
    try:
        cv2.resizeWindow(WINDOW, max(320, int(fw * raw / 100)), max(240, int(fh * raw / 100)))
    except Exception:
        pass


def _cb(attr, to_val):
    def fn(raw):
        set_param(attr, to_val(raw))
    return fn


def attach(window=WINDOW):
    """Create all trackbars on the named window."""
    for label, attr, max_raw, to_val, to_raw in _TRACKBARS:
        cur = get_param(attr)
        init = int(max(0, min(max_raw, to_raw(cur if cur is not None else 0))))
        cv2.createTrackbar(label, window, init, max_raw, _cb(attr, to_val))
    cv2.createTrackbar("VIEW%", window, 100, 100, _view_cb)


def _mouse(event, x, y, flags, _):
    """Mouse callback for dragging the HUD and param panels."""
    if event == cv2.EVENT_LBUTTONDOWN:
        for key in ("hud", "param"):
            rx, ry, rw, rh = UI[key + "_rect"]
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                UI["drag"] = (key, x - rx, y - ry)
                return
    elif event == cv2.EVENT_MOUSEMOVE and UI["drag"]:
        key, dx, dy = UI["drag"]
        UI[key + "_pos"] = [max(0, x - dx), max(0, y - dy)]
    elif event == cv2.EVENT_LBUTTONUP:
        UI["drag"] = None


def draw_param_panel(canvas):
    """Read-only live parameter readout (drag with mouse; tune via trackbars/web)."""
    x, y = UI["param_pos"]
    w, line_h = 240, 20
    rows = [
        ("CONF",        f"{get_param('CONF') or 0:.2f}"),
        ("MIN_HEAD",    f"{get_param('MIN_HEAD_HEIGHT') or 0:.0f}px"),
        ("TRACK_SCL",   f"{get_param('TRACK_MAX_DISPLACEMENT_SCALE') or 0:.1f}"),
        ("KDE_GAMMA",   f"{get_param('KDE_GAMMA') or 0:.1f}"),
        ("P_CRIT",      f"{(get_param('RISK_P_CRIT') or 0)/1000:.0f}k"),
        ("DENS_CRIT",   f"{get_param('RISK_DENS_CRIT') or 0:.0f}"),
        ("COUNT_FLR",   f"{get_param('RISK_COUNT_FLOOR') or 0:.0f}"),
    ]
    h = 26 + len(rows) * line_h + 8
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (120, 120, 120), 1)
    cv2.putText(canvas, "PARAMS (drag me)", (x + 8, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    ty = y + 26 + 12
    for label, val in rows:
        cv2.putText(canvas, label, (x + 10, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(canvas, val, (x + w - 10 - tw, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        ty += line_h
    UI["param_rect"] = (x, y, w, h)
    return canvas


def select_source(default=None):
    """Terminal-based interactive source picker."""
    try:
        print("\n=== SOURCE MENU ===  [1] Webcam  [2] RTSP/HTTP  [3] YouTube  [4] File  [5] Config default")
        ch = (input("Select [1-5]: ").strip() or "5")
    except EOFError:
        return default
    if ch == "1":
        return int(input("Webcam index [0]: ").strip() or "0")
    if ch == "2":
        return input("RTSP/HTTP URL: ").strip()
    if ch == "3":
        return input("YouTube URL: ").strip()
    if ch == "4":
        return input("Video file path: ").strip()
    return default
