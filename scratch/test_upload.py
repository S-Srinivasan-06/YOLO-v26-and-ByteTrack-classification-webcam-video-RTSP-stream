import io
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.server import start_server

PORT = 8898
BASE = f"http://127.0.0.1:{PORT}"

server = start_server(port=PORT, host="127.0.0.1")
t = threading.Thread(target=server.serve_forever, daemon=True)
t.start()
time.sleep(0.5)

# Prepare multipart body
boundary = "---------------------------974767299852498929531610575"
video_path = Path("data/videos/easy.mp4")
video_bytes = video_path.read_bytes()

body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="easy.mp4"\r\n'
    "Content-Type: video/mp4\r\n\r\n"
).encode("latin-1") + video_bytes + f"\r\n--{boundary}--\r\n".encode("latin-1")

req = urllib.request.Request(
    f"{BASE}/api/upload",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
)

with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode())
    print("Upload Response:", res)
    assert res["status"] == "uploaded"
    assert res["filename"] == "easy.mp4"
    assert "fps" in res and "resolution" in res

print("\nUPLOAD TEST PASSED!")
server.shutdown()
server.server_close()
