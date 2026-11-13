"""Live webcam window with skeleton, bounding box and FPS. Press q to quit.

Usage: python scripts/webcam_baseline.py --engine mediapipe [--variant lite|full]
       python scripts/webcam_baseline.py --engine yolo [--device cuda:0|cpu]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.engines import make_engine  # noqa: E402
from app.pipeline.annotate import draw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["mediapipe", "yolo"], default="mediapipe")
    ap.add_argument("--variant", choices=["lite", "full"], default="full")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()

    kw = {"variant": a.variant} if a.engine == "mediapipe" else {"device": a.device}
    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW)
    assert cap.isOpened(), f"cannot open camera {a.camera}"
    t0 = prev = time.perf_counter()
    fps = 0.0
    with make_engine(a.engine, **kw) as engine:
        print("running", engine.name, "- press q to quit")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = engine.infer(frame, int((time.perf_counter() - t0) * 1000))
            if res:
                draw(frame, res)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - prev, 1e-3)  # EMA of loop FPS
            prev = now
            cv2.putText(frame, f"{engine.name}  {fps:.1f} FPS  {'pose' if res else 'no pose'}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("webcam baseline", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
