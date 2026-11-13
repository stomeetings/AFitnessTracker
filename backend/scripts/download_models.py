"""Download pose model weights into backend/models/ (idempotent)."""
import sys
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"
MP_BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
MP_FILES = {
    "pose_landmarker_lite.task": f"{MP_BASE}/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task": f"{MP_BASE}/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}


def main():
    MODELS.mkdir(exist_ok=True)
    for name, url in MP_FILES.items():
        dst = MODELS / name
        if dst.exists():
            print("have", dst.name, dst.stat().st_size // 1024, "KB")
            continue
        print("downloading", name)
        urllib.request.urlretrieve(url, dst)
        print("saved", dst.name, dst.stat().st_size // 1024, "KB")

    yolo = MODELS / "yolo11n-pose.pt"
    if not yolo.exists():
        from ultralytics import YOLO
        print("downloading yolo11n-pose.pt")
        YOLO(str(yolo))  # ultralytics downloads the asset to this path
    print("have", yolo.name, yolo.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    sys.exit(main())
