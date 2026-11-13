"""P0 environment check: MediaPipe import, CUDA availability, tiny CUDA matmul, webcam open."""
import sys

import mediapipe as mp
import torch
import cv2

print("python     ", sys.version.split()[0])
print("mediapipe  ", mp.__version__)
print("torch      ", torch.__version__, "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available"
print("gpu        ", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))

a = torch.randn(512, 512, device="cuda")
c = (a @ a).sum().item()
assert c == c, "matmul produced NaN"  # NaN != NaN
print("cuda matmul OK, sum =", round(c, 2))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ok, frame = cap.read()
cap.release()
print("webcam     ", "OK" if ok else "FAILED", frame.shape if ok else "")
print("ALL CHECKS PASSED" if ok else "WEBCAM CHECK FAILED")
