"""Synthetic-signal checks for angles, rep FSM / ROM, and fall transitions. No framework.

Run from repo root:  backend\\.venv\\Scripts\\python tests\\test_analytics.py
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.engines import BBox, Keypoint, PoseResult  # noqa: E402
from app.pipeline.core import Pipeline  # noqa: E402
from app.pipeline.fall_detector import CONFIRMED, NORMAL, SUSPECTED, FallDetector  # noqa: E402
from app.pipeline.kinematics import EMA, angle_deg, joint_angles  # noqa: E402
from app.pipeline.rep_counter import RepCounter  # noqa: E402

FPS = 30


def test_angle():
    assert abs(angle_deg((0, 1), (0, 0), (1, 0)) - 90) < 1e-6
    assert abs(angle_deg((0, 0, 1), (0, 0, 0), (0, 0, -1)) - 180) < 1e-6
    assert angle_deg((0, 0), (0, 0), (1, 1)) is None


def _sine_angles(key_rest_peak, cycles, period_s, jitter=0.0, seed=0):
    """Yields (t, {key: angle}) sweeping rest -> peak -> rest per cycle for every (key, rest, peak)."""
    rng = random.Random(seed)
    n = int(cycles * period_s * FPS)
    for i in range(n):
        t = i / FPS
        phase = (1 - math.cos(2 * math.pi * t / period_s)) / 2  # 0..1..0
        yield t, {k: rest + (peak - rest) * phase + rng.uniform(-jitter, jitter) for k, rest, peak in key_rest_peak}


def test_rep_counter_clean_and_noisy():
    for jitter in (0.0, 4.0):
        rc = RepCounter("Elbow Curl")
        ema = EMA(0.35)
        for t, a in _sine_angles([("l_elbow", 160, 45), ("r_elbow", 160, 60)], cycles=5, period_s=2.0, jitter=jitter):
            rc.update(ema.update(a), t)
        snap = rc.snapshot()
        assert snap["L"]["reps"] == 5, (jitter, snap)
        assert snap["R"]["reps"] == 5, (jitter, snap)
        assert abs(snap["L"]["best_rom"] - 115) < 12, snap["L"]
        assert abs(snap["R"]["best_rom"] - 100) < 12, snap["R"]


def test_rep_counter_rejects_partial_and_too_fast():
    rc = RepCounter("Knee Flexion")
    for t, a in _sine_angles([("l_knee", 170, 150)], cycles=5, period_s=2.0):  # only 30% of the way: no rep
        rc.update(a, t)
    assert rc.snapshot()["L"]["reps"] == 0
    rc = RepCounter("Knee Flexion")
    for t, a in _sine_angles([("l_knee", 170, 95)], cycles=5, period_s=0.2):  # 0.2 s reps: below MIN_REP_S
        rc.update(a, t)
    assert rc.snapshot()["L"]["reps"] == 0


def test_sit_to_stand_and_raise_direction():
    rc = RepCounter("Sit-to-Stand")
    for t, a in _sine_angles([("knee_mean", 95, 172)], cycles=3, period_s=3.0):
        rc.update(a, t)
    assert rc.snapshot()["B"]["reps"] == 3
    rc = RepCounter("Shoulder Raise")
    for t, a in _sine_angles([("l_shoulder", 15, 100), ("r_shoulder", 15, 100)], cycles=4, period_s=2.0):
        rc.update(a, t)
    assert rc.snapshot()["L"]["reps"] == 4 and rc.snapshot()["R"]["reps"] == 4


def _pose(hip_y, tilt_deg, torso=100.0, ts=0, lying_box=False, seated=False):
    """Stick figure in a 640x480 frame; shoulders rotated by tilt_deg from vertical about the hip.
    seated: thigh horizontal, shank vertical (hip one thigh-length above the ankles)."""
    cx = 320.0
    rad = math.radians(tilt_deg)
    sh = (cx + torso * math.sin(rad), hip_y - torso * math.cos(rad))
    kp = {}
    for s, dx in (("l", -15), ("r", 15)):
        kp[f"{s}_shoulder"] = Keypoint(sh[0] + dx, sh[1], 0.95)
        kp[f"{s}_hip"] = Keypoint(cx + dx, hip_y, 0.95)
        if seated:
            kp[f"{s}_knee"] = Keypoint(cx + dx + 0.9 * torso, hip_y, 0.95)
            kp[f"{s}_ankle"] = Keypoint(cx + dx + 0.9 * torso, hip_y + 0.9 * torso, 0.95)
        else:
            kp[f"{s}_knee"] = Keypoint(cx + dx, hip_y + 0.9 * torso, 0.95)
            kp[f"{s}_ankle"] = Keypoint(cx + dx, hip_y + 1.8 * torso, 0.95)
        kp[f"{s}_elbow"] = Keypoint(sh[0] + dx * 2, sh[1] + 0.4 * torso, 0.95)
        kp[f"{s}_wrist"] = Keypoint(sh[0] + dx * 2, sh[1] + 0.8 * torso, 0.95)
    bbox = BBox(cx - 150, hip_y - 20, 300, 80, 0.9) if lying_box else BBox(cx - 40, sh[1] - 20, 80, 2.1 * torso, 0.9)
    return PoseResult(kp, bbox, ts)


def test_fall_sequence():
    fd = FallDetector()
    seen = []
    t = 0.0

    def step(res, dt=1 / FPS):
        nonlocal t
        t += dt
        st = fd.update(res, t)["status"]
        if not seen or seen[-1] != st:
            seen.append(st)

    for _ in range(FPS):                       # 1 s standing upright
        step(_pose(240, 0))
    assert fd.status == NORMAL
    for i in range(1, 10):                     # 0.3 s rapid drop + rotate to horizontal
        step(_pose(240 + i * 20, i * 9, lying_box=i > 5))
    assert fd.status == SUSPECTED, (fd.status, fd.snapshot(t))
    for _ in range(int(1.0 * FPS)):            # lying still 1.0 s: not yet confirmed
        step(_pose(420, 85, lying_box=True))
    assert fd.status == SUSPECTED
    for _ in range(int(0.8 * FPS)):            # +0.8 s -> confirmed (>1.5 s total)
        step(_pose(420, 85, lying_box=True))
    assert fd.status == CONFIRMED
    for _ in range(int(1.3 * FPS)):            # stands back up for >1 s -> recovery
        step(_pose(240, 0))
    assert fd.status == NORMAL
    assert seen == [NORMAL, SUSPECTED, CONFIRMED, NORMAL], seen


def test_fall_pose_lost_after_drop_confirms():
    fd = FallDetector()
    t = 0.0
    for _ in range(FPS):
        t += 1 / FPS; fd.update(_pose(240, 0), t)
    for i in range(1, 8):
        t += 1 / FPS; fd.update(_pose(240 + i * 25, i * 5), t)
    assert fd.status == SUSPECTED
    for _ in range(int(1.6 * FPS)):             # tracker loses the body on the floor
        t += 1 / FPS; fd.update(None, t)
    assert fd.status == CONFIRMED


def test_bend_over_stays_normal():
    fd = FallDetector()
    t = 0.0
    for _ in range(FPS):
        t += 1 / FPS; fd.update(_pose(240, 0), t)
    for i in range(1, 8):                       # bend forward to 70 deg, wide box, hips stay high
        t += 1 / FPS; assert fd.update(_pose(240, i * 10, lying_box=i > 4), t)["status"] == NORMAL
    for _ in range(int(3 * FPS)):               # hold the bend (picking something up)
        t += 1 / FPS; assert fd.update(_pose(240, 70, lying_box=True), t)["status"] == NORMAL


def test_sitting_down_stays_normal():
    for sit_s in (1.5, 0.5):                    # slow and quick sit onto a chair
        fd = FallDetector()
        t = 0.0
        for _ in range(FPS):
            t += 1 / FPS; fd.update(_pose(240, 0), t)
        n = int(sit_s * FPS)
        for i in range(1, n + 1):               # hip drops 0.9 torso, leans forward 35 deg on the way down
            t += 1 / FPS
            hip = 240 + 90 * i / n
            tilt = 35 * math.sin(math.pi * i / n)
            assert fd.update(_pose(hip, tilt, seated=i > n // 2), t)["status"] == NORMAL, (sit_s, i, fd.snapshot(t))
        for _ in range(int(4 * FPS)):           # sits still, slouched
            t += 1 / FPS; assert fd.update(_pose(330, 25, seated=True), t)["status"] == NORMAL, fd.snapshot(t)


def test_fall_from_chair_confirms():
    fd = FallDetector()
    t = 0.0
    for _ in range(FPS):
        t += 1 / FPS; fd.update(_pose(240, 0), t)
    for _ in range(int(20 * FPS)):              # seated 20 s (baseline window still remembers standing)
        t += 1 / FPS; fd.update(_pose(330, 10, seated=True), t)
    assert fd.status == NORMAL
    for i in range(1, 8):                       # topples sideways onto the floor
        t += 1 / FPS; fd.update(_pose(330 + i * 13, 10 + i * 11, lying_box=i > 3), t)
    for _ in range(int(1.9 * FPS)):             # 0.3 s velocity window to settle + 1.5 s still
        t += 1 / FPS; fd.update(_pose(420, 85, lying_box=True), t)
    assert fd.status == CONFIRMED, fd.snapshot(t)


def test_pipeline_end_to_end_angles():
    p = Pipeline("Knee Flexion")
    out = p.process(_pose(240, 0), 0.0)
    assert out["posture"] == "standing" and out["fall"]["status"] == NORMAL
    assert 170 < out["angles"]["l_knee"] <= 180, out["angles"]
    assert joint_angles(_pose(240, 0))["l_elbow"] > 150


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL", fn.__name__, "->", repr(e))
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
