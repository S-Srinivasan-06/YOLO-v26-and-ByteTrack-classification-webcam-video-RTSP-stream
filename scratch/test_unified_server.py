import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import threading
import time
import urllib.request
from src.server import start_server

PORT = 8899
BASE = f"http://127.0.0.1:{PORT}"

server = start_server(port=PORT, host="127.0.0.1")
t = threading.Thread(target=server.serve_forever, daemon=True)
t.start()
time.sleep(0.5)

print("[Test] Server started on", BASE)

# 1. Test Schema
req = urllib.request.Request(f"{BASE}/api/config/schema")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    assert "schema" in data and "presets" in data
    print("  [Pass] GET /api/config/schema")

# 2. Test State
req = urllib.request.Request(f"{BASE}/api/state")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    assert data["status"] == "IDLE"
    print("  [Pass] GET /api/state (status=IDLE)")

# 3. Test Config Update
payload = json.dumps({"params": {"CONF": 0.05, "TRACK_MAX_DISPLACEMENT_SCALE": 4.5}}).encode()
req = urllib.request.Request(f"{BASE}/api/config", data=payload, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    assert data["config"]["CONF"] == 0.05
    print("  [Pass] POST /api/config (CONF=0.05)")

# 4. Test Session Start
video_path = str(Path("data/videos/easy.mp4").resolve())
start_payload = json.dumps({
    "source_type": "file",
    "source": video_path,
    "params": {"MAX_RUNTIME_SEC": 3.0},
    "roi_polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
}).encode()

req = urllib.request.Request(f"{BASE}/api/session/start", data=start_payload, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    run_id = data["run_id"]
    print(f"  [Pass] POST /api/session/start (run_id={run_id})")

# Wait for a couple inference cycles
time.sleep(1.5)

# 5. Check active state
req = urllib.request.Request(f"{BASE}/api/state")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    print(f"  [Pass] Live State: status={data['status']}, frames={data['frames']}, inferences={data['inferences']}")

# 6. Stop session
req = urllib.request.Request(f"{BASE}/api/session/stop", data=b"{}", headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    assert data["status"] == "stopped"
    print("  [Pass] POST /api/session/stop")

time.sleep(0.5)

# 7. Check History
req = urllib.request.Request(f"{BASE}/api/runs")
with urllib.request.urlopen(req) as resp:
    runs = json.loads(resp.read().decode())
    assert len(runs) > 0
    print(f"  [Pass] GET /api/runs (found {len(runs)} runs, latest={runs[0]['run_id']})")

# 8. Check Run Artifacts on Disk
run_dir = Path("data/runs") / run_id
assert (run_dir / "log.csv").exists()
assert (run_dir / "summary.json").exists()
assert (run_dir / "config.json").exists()
print(f"  [Pass] Run artifacts verified in {run_dir}")

server.shutdown()
server.server_close()
print("\nALL SERVER & SESSION TESTS PASSED PERFECTLY!")
