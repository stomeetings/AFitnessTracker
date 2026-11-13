# Pose engine benchmark (2026-11-13)

- Source: `0`, 100 frames at 640x480, 5 warm-up frames excluded
- CPU: Intel64 Family 6 Model 158 Stepping 9, GenuineIntel | GPU: NVIDIA GeForce GTX 1050 | torch 2.14.1+cu126 | Python 3.13.5
- Latency is inference only (no capture, no drawing). Hit rate = frames with a detected pose.

| Engine | FPS | Median ms | p95 ms | Hit rate % | Peak VRAM MB |
|---|---|---|---|---|---|
| mediapipe-lite | 57.7 | 16.3 | 21.6 | 59 | 0 |
| mediapipe-full | 48.2 | 20.8 | 26.9 | 59 | 0 |
| yolo11n-pose-cuda | 45.6 | 20.4 | 29.2 | 76 | 82 |
| yolo11n-pose-cpu | 16.1 | 60.5 | 71.9 | 76 | 0 |
