"""Download head-detection model and export to OpenVINO + ONNX."""
import os
import shutil
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


def setup():
    os.makedirs("models", exist_ok=True)
    target_vino = "models/head_yolov8_openvino_model"
    target_onnx = "models/head_yolov8.onnx"

    if os.path.exists(target_vino) and os.path.exists(target_onnx):
        print("Models already exist. Skipping.")
        return

    print("Downloading YOLOv8n head-detection weights...")
    try:
        pt_path = hf_hub_download(
            repo_id="keremberke/yolov8n-head-detection", filename="best.pt"
        )
    except Exception as e:
        print(f"Download failed: {e}")
        print("Manually place a head-detection .pt in models/ and export.")
        return

    model = YOLO(pt_path)

    if not os.path.exists(target_vino):
        print("Exporting to OpenVINO (imgsz=1280, half, dynamic)...")
        out = model.export(format="openvino", imgsz=1280, half=True, dynamic=True)
        if os.path.exists(out) and out != target_vino:
            if os.path.exists(target_vino):
                shutil.rmtree(target_vino)
            shutil.move(out, target_vino)

    if not os.path.exists(target_onnx):
        print("Exporting to ONNX (imgsz=1280, dynamic, simplified)...")
        out = model.export(format="onnx", imgsz=1280, dynamic=True, simplify=True, opset=12)
        if os.path.exists(out) and out != target_onnx:
            shutil.move(out, target_onnx)

    print("Done. Models ready in models/")


if __name__ == "__main__":
    setup()
