from huggingface_hub import hf_hub_download, snapshot_download
import shutil
import glob
import os
from ultralytics import YOLO

repos = [
    "keremberke/yolov8s-nlf-head-detection",
    "keremberke/yolov8s-head-detection",
    "keremberke/yolov8n-head-detection"
]
filenames = ["best.pt", "model.pt", "weights/best.pt", "weights/last.pt", "yolov8s-head-detection.pt", "yolov8s-nlf-head-detection.pt", "yolov8n-head-detection.pt"]
target = "models/head_yolov8s_keremberke.pt"

success = False
for repo in repos:
    if success: break
    for fn in filenames:
        try:
            print(f"Trying to download {fn} from {repo}...")
            path = hf_hub_download(repo_id=repo, filename=fn)
            shutil.copy(path, target)
            model = YOLO(target)
            names = {int(k): str(v).strip().lower() for k, v in model.names.items()}
            if names == {0: "head"}:
                print("SUCCESS downloaded and validated:", target, "from", repo, fn)
                success = True
                break
            else:
                print(f"Invalid names for {fn}:", names)
        except Exception as e:
            pass
    if success: break
    
    try:
        print(f"Trying snapshot download for {repo}...")
        path = snapshot_download(repo_id=repo)
        pt_files = glob.glob(os.path.join(path, "**", "*.pt"), recursive=True)
        for pt in pt_files:
            shutil.copy(pt, target)
            model = YOLO(target)
            names = {int(k): str(v).strip().lower() for k, v in model.names.items()}
            if names == {0: "head"}:
                print("SUCCESS downloaded and validated via snapshot:", target, "from", repo, pt)
                success = True
                break
    except Exception as e:
        pass

if not success:
    print("FAILED all downloads. Using existing models/head_yolov8.pt")
