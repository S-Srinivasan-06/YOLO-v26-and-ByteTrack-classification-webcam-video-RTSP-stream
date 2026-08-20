"""CrowdShield L1 Vision Platform — Headless Unified Entrypoint.

Starts the full web platform on http://localhost:8000.
Browser manages video uploads, sources, ROI boundaries, live overlays, and parameter tuning.
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src.config as C
from src.server import start_server


def main():
    port = int(os.environ.get("PORT", getattr(C, "WEB_PORT", 8000)))
    server = start_server(port=port, host="0.0.0.0")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[CrowdShield] Server shutting down cleanly.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
