"""Benchmark pose engines on the same N frames. Writes docs/benchmark_results.md.

Usage: python scripts/benchmark_engines.py [--frames 100] [--source 0|video.mp4]
"""
import argparse
import platform
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.engines import make_engine  # noqa: E402

DOCS = Path(__file__).resolve().parents[2] / "docs"
CONFIGS = [
    ("mediapipe", {"variant": "lite"}),
    ("mediapipe", {"variant": "full"}),
    ("yolo", {"device": "cuda:0"}),
    ("yolo", {"device": "cpu"}),
]


def grab_frames(source, n):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source, cv2.CAP_DSHOW if str(source).isdigit() else 0)
    assert cap.isOpened(), f"cannot open {source}"
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    assert frames, "no frames captured"
    return frames


def bench(name, kw, frames, warmup=5):
    engine = make_engine(name, **kw)
    engine.load()
    uses_cuda = "cuda" in getattr(engine, "device", "")
    if uses_cuda:
        torch.cuda.reset_peak_memory_stats()
    for i in range(min(warmup, len(frames))):
        engine.infer(frames[i], i * 33)
    lat, hits = [], 0
    for i, f in enumerate(frames):
        t0 = time.perf_counter()
        r = engine.infer(f, (warmup + i) * 33)
        if uses_cuda:
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
        hits += r is not None
    engine.close()
    lat.sort()
    return {
        "engine": engine.name,
        "fps": 1000 / statistics.mean(lat),
        "median_ms": statistics.median(lat),
        "p95_ms": lat[int(0.95 * (len(lat) - 1))],
        "hit_rate": 100 * hits / len(frames),
        "vram_mb": torch.cuda.max_memory_allocated() / 2**20 if uses_cuda else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--source", default="0")
    ap.add_argument("--out", default=str(DOCS / "benchmark_results.md"))
    a = ap.parse_args()

    frames = grab_frames(a.source, a.frames)
    h, w = frames[0].shape[:2]
    print(f"captured {len(frames)} frames at {w}x{h} from {a.source}")

    rows = []
    for name, kw in CONFIGS:
        print("benchmarking", name, kw, "...", flush=True)
        rows.append(bench(name, kw, frames))
        print(f"  {rows[-1]['engine']}: {rows[-1]['fps']:.1f} FPS, median {rows[-1]['median_ms']:.1f} ms")

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    lines = [
        f"# Pose engine benchmark ({date.today()})",
        "",
        f"- Source: `{a.source}`, {len(frames)} frames at {w}x{h}, 5 warm-up frames excluded",
        f"- CPU: {platform.processor()} | GPU: {gpu} | torch {torch.__version__} | Python {platform.python_version()}",
        "- Latency is inference only (no capture, no drawing). Hit rate = frames with a detected pose.",
        "",
        "| Engine | FPS | Median ms | p95 ms | Hit rate % | Peak VRAM MB |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| {r['engine']} | {r['fps']:.1f} | {r['median_ms']:.1f} | {r['p95_ms']:.1f} | {r['hit_rate']:.0f} | {r['vram_mb']:.0f} |" for r in rows]
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[6:]))
    print("saved", a.out)


if __name__ == "__main__":
    main()
