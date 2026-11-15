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
from app.pipeline.escalation import ACKNOWLEDGED, ALERTED, COUNTDOWN, IDLE  # noqa: E402
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
    for _ in range(int(4.5 * FPS)):            # +4.5 s -> confirmed (down for the whole 5 s dwell)
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
    for _ in range(int(5.2 * FPS)):             # tracker loses the body on the floor for the dwell time
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
    for _ in range(int(5.5 * FPS)):             # stays down for the 5 s dwell
        t += 1 / FPS; fd.update(_pose(420, 85, lying_box=True), t)
    assert fd.status == CONFIRMED, fd.snapshot(t)


def _floor_pose(drop=(), box=(200, 380, 320, 60), **override):
    """Person lying on their side across the frame, legs out straight. `drop`: joints (e.g. "ankle") the
    tracker is unsure about; `override`: joint -> (x, y) to reshape the pose (applied to both sides)."""
    xy = {"shoulder": (220, 400), "hip": (320, 405), "knee": (410, 408), "ankle": (500, 410),
          "elbow": (250, 410), "wrist": (290, 412), **override}
    kp = {f"{s}_{j}": Keypoint(x, y + dy, 0.2 if j in drop else 0.95)
          for s, dy in (("l", -8), ("r", 8)) for j, (x, y) in xy.items()}
    return PoseResult(kp, BBox(*box, 0.9), 0)


def _status_after(res, seconds):
    """Final status when tracking starts with the person already in this pose (no standing baseline)."""
    fd, t = FallDetector(), 0.0
    for _ in range(int(seconds * FPS)):
        t += 1 / FPS; fd.update(res, t)
    return fd.status, fd.snapshot(t)


def test_already_on_floor_confirms_without_standing_baseline():
    assert _status_after(_floor_pose(), 1.0)[0] == SUSPECTED                       # flagged at once, not yet confirmed
    assert _status_after(_floor_pose(), 5.5)[0] == CONFIRMED                       # all joints tracked
    st, snap = _status_after(_floor_pose(drop=("ankle",)), 5.5)                    # feet not tracked: body is flat
    assert st == CONFIRMED, snap
    st, snap = _status_after(_floor_pose(drop=("hip",)), 5.5)                      # torso not tracked: wide box
    assert st == CONFIRMED, snap
    fd, t = FallDetector(), 0.0                                                    # still down after the 30 s baseline expires
    for _ in range(40 * FPS):
        t += 1 / FPS; fd.update(_floor_pose(drop=("ankle",)), t)
    assert fd.status == CONFIRMED and fd.snapshot(t)["on_floor"], fd.snapshot(t)


def test_floor_lookalikes_stay_normal():
    # on hands and knees: torso horizontal and box wide, but the hips are a thigh-length above the knees
    all_fours = _floor_pose(box=(200, 300, 230, 130), shoulder=(220, 330), hip=(320, 330), knee=(320, 420), ankle=(410, 425))
    st, snap = _status_after(all_fours, 5.0)
    assert st == NORMAL, snap
    # sitting close to the camera: only head and shoulders in view, hips not tracked, box wider than tall
    kp = {"l_shoulder": Keypoint(200, 400, 0.95), "r_shoulder": Keypoint(440, 400, 0.95),
          "l_hip": Keypoint(230, 470, 0.2), "r_hip": Keypoint(410, 470, 0.2)}
    st, snap = _status_after(PoseResult(kp, BBox(180, 300, 280, 170, 0.9), 0), 5.0)
    assert st == NORMAL, snap


# Keypoints read off docs/sample/image_9379a9.jpg: patient flat on their back, feet toward a raised camera.
# On screen the skeleton is upright (tilt ~3 deg, box taller than wide) and looks like a standing person.
_REF_LYING = {"l_shoulder": (270, 262), "r_shoulder": (335, 265), "l_hip": (283, 337), "r_hip": (322, 337),
              "l_knee": (280, 370), "r_knee": (318, 372), "l_ankle": (270, 410), "r_ankle": (328, 412)}
# The same person standing on that spot (feet in the same place, about twice as tall on screen) ...
_REF_STANDING = {"l_shoulder": (270, 70), "r_shoulder": (335, 70), "l_hip": (283, 195), "r_hip": (322, 195),
                 "l_knee": (280, 300), "r_knee": (318, 300), "l_ankle": (270, 410), "r_ankle": (328, 412)}
# ... and standing further back in the room: smaller, and higher up the image.
_REF_FAR = {"l_shoulder": (280, 90), "r_shoulder": (325, 90), "l_hip": (288, 170), "r_hip": (317, 170),
            "l_knee": (287, 240), "r_knee": (316, 240), "l_ankle": (283, 300), "r_ankle": (321, 300)}


def _blend(a, b, f):
    """Pose a fraction f of the way from keypoint map a to b, with a box around the joints."""
    kp = {n: Keypoint(a[n][0] + (b[n][0] - a[n][0]) * f, a[n][1] + (b[n][1] - a[n][1]) * f, 0.95) for n in a}
    xs, ys = [k.x for k in kp.values()], [k.y for k in kp.values()]
    return PoseResult(kp, BBox(min(xs) - 40, min(ys) - 50, max(xs) - min(xs) + 80, max(ys) - min(ys) + 90, 0.9), 0)


def _move(p, t, a, b, seconds):
    """Runs the pipeline while the pose moves from a to b over `seconds`. Returns (t, last output)."""
    n = int(seconds * FPS)
    for i in range(1, n + 1):
        t += 1 / FPS
        out = p.process(_blend(a, b, i / n), t)
    return t, out


def test_reference_image_lying_feet_toward_camera():
    p = Pipeline("Elbow Curl")
    t, out = _move(p, 0.0, _REF_STANDING, _REF_STANDING, 1.0)
    assert out["fall"]["status"] == NORMAL, out["fall"]
    t, out = _move(p, t, _REF_STANDING, _REF_LYING, 3.0)             # lowers to the floor slowly: no fast drop
    assert out["fall"]["status"] != NORMAL and out["fall"]["on_floor"], out["fall"]   # flagged as soon as they are down
    t, out = _move(p, t, _REF_LYING, _REF_LYING, 2.0)                # lies still, as in the image
    assert out["fall"]["status"] == SUSPECTED and out["fall"]["alert"]["state"] == IDLE, out["fall"]   # inside the dwell
    t, out = _move(p, t, _REF_LYING, _REF_LYING, 3.0)                # still down after 5 s
    assert out["fall"]["status"] == CONFIRMED and out["posture"] == "lying", out["fall"]
    assert out["fall"]["alert"]["state"] == COUNTDOWN and not out["fall"]["impact"], out["fall"]
    t, out = _move(p, t, _REF_LYING, _REF_STANDING, 2.0)             # gets back up
    t, out = _move(p, t, _REF_STANDING, _REF_STANDING, 1.5)
    assert out["fall"]["status"] == NORMAL and out["fall"]["alert"]["state"] == IDLE, out["fall"]


def test_walking_away_from_camera_stays_normal():
    p = Pipeline("Elbow Curl")                                       # shrinking on screen, but the feet move up it
    t, out = _move(p, 0.0, _REF_STANDING, _REF_STANDING, 1.0)
    for a, b, seconds in ((_REF_STANDING, _REF_FAR, 3.0), (_REF_FAR, _REF_FAR, 5.0), (_REF_FAR, _REF_STANDING, 3.0)):
        n = int(seconds * FPS)
        for i in range(1, n + 1):
            t += 1 / FPS
            assert p.process(_blend(a, b, i / n), t)["fall"]["status"] == NORMAL, (i, p.fall.snapshot(t))


def test_lying_head_toward_camera_confirms():
    upside_down = _floor_pose(box=(230, 220, 140, 220), shoulder=(300, 400), hip=(300, 330), knee=(300, 290),
                              ankle=(300, 250), elbow=(260, 390), wrist=(260, 360))
    st, snap = _status_after(upside_down, 5.5)                      # no standing reference needed
    assert st == CONFIRMED, snap


def test_pipeline_end_to_end_angles():
    p = Pipeline("Knee Flexion")
    out = p.process(_pose(240, 0), 0.0)
    assert out["posture"] == "standing" and out["fall"]["status"] == NORMAL
    assert 170 < out["angles"]["l_knee"] <= 180, out["angles"]
    assert joint_angles(_pose(240, 0))["l_elbow"] > 150


def test_new_angles_hip_abduction_and_elbow_mean():
    res = _pose(240, 0)
    a = joint_angles(res)
    assert abs(a["l_hip_abd"] - 90) < 1 and abs(a["r_hip_abd"] - 90) < 1, a      # standing: thigh square to the pelvis
    assert abs(a["elbow_mean"] - (a["l_elbow"] + a["r_elbow"]) / 2) < 1e-6
    res.keypoints["l_knee"] = Keypoint(305 - 45, 240 + 78, 0.95)                 # left leg lifted 30 deg out to the side
    a = joint_angles(res)
    assert abs(a["l_hip_abd"] - 120) < 1 and abs(a["r_hip_abd"] - 90) < 1, a


def test_new_exercises_count_reps():
    cases = [
        ("Overhead Reach", [("l_shoulder", 18, 152), ("r_shoulder", 18, 152)], ("L", "R")),
        ("Seated Knee Extension", [("l_knee", 93, 168)], ("L",)),
        ("Mini Squat", [("knee_mean", 172, 112)], ("B",)),
        ("Side Leg Raise", [("r_hip_abd", 90, 122)], ("R",)),
        ("Wall Push-up", [("elbow_mean", 167, 92)], ("B",)),
    ]
    for name, sweep, sides in cases:
        rc = RepCounter(name)
        for t, a in _sine_angles(sweep, cycles=3, period_s=3.0):
            rc.update(a, t)
        for side in sides:
            assert rc.snapshot()[side]["reps"] == 3, (name, rc.snapshot())


def _fall(p, t=0.0):
    """Stands 1 s, then drops to the floor in 0.3 s (same motion as test_fall_sequence). Returns t."""
    for _ in range(FPS):
        t += 1 / FPS; p.process(_pose(240, 0), t)
    for i in range(1, 10):
        t += 1 / FPS; p.process(_pose(240 + i * 20, i * 9, lying_box=i > 5), t)
    return t


def _hold(p, t, res, seconds):
    """Feeds one pose for `seconds`. Returns (t, last alert dict, alert states in order, times the alert fired)."""
    seen, fired, alert = [], 0, None
    for _ in range(int(seconds * FPS)):
        t += 1 / FPS
        alert = p.process(res, t)["fall"]["alert"]
        fired += p.escalation.fired
        if not seen or seen[-1] != alert["state"]:
            seen.append(alert["state"])
    return t, alert, seen, fired


LYING, STANDING = _pose(420, 85, lying_box=True), _pose(240, 0)


def test_escalation_countdown_expires_and_alerts_once():
    p = Pipeline("Knee Flexion")
    t, alert, seen, fired = _hold(p, _fall(p), LYING, 5.5)          # confirmed after the 5 s dwell on the floor
    assert seen == [IDLE, COUNTDOWN] and fired == 0, seen
    assert alert["impact"] and 8.5 < alert["seconds_left"] < 10.0, alert  # 10 s countdown, ~0.7 s already gone
    t, alert, seen, fired = _hold(p, t, LYING, 10.5)
    assert seen == [COUNTDOWN, ALERTED] and fired == 1, (seen, fired)
    t, alert, seen, fired = _hold(p, t, STANDING, 2.0)              # getting up clears a raised alert by itself
    assert seen == [ALERTED, IDLE] and fired == 0 and p.fall.status == NORMAL, seen


def test_escalation_recovery_cancels_countdown():
    p = Pipeline("Knee Flexion")
    t, alert, seen, fired = _hold(p, _fall(p), LYING, 3.0)          # up again inside the dwell: nothing starts
    assert alert["state"] == IDLE and p.fall.status == SUSPECTED, alert
    t, alert, seen, fired = _hold(p, t, STANDING, 1.0)
    assert seen == [IDLE] and p.fall.status == NORMAL, seen
    t, alert, seen, fired = _hold(p, _fall(p, t), LYING, 5.5)       # falls again and stays down: countdown starts
    assert alert["state"] == COUNTDOWN
    t, alert, seen, fired = _hold(p, t, STANDING, 1.5)              # up again for > RECOVER_S
    assert seen == [COUNTDOWN, IDLE] and fired == 0, seen
    t, alert, seen, fired = _hold(p, t, STANDING, 10.0)             # and nothing fires later
    assert seen == [IDLE] and fired == 0, seen


def test_escalation_acknowledge():
    p = Pipeline("Knee Flexion")
    p.escalation.ack()                                              # stale "I'm OK" before any fall is ignored
    t, alert, seen, fired = _hold(p, _fall(p), LYING, 5.5)
    assert alert["state"] == COUNTDOWN, alert
    p.escalation.ack()
    t, alert, seen, fired = _hold(p, t, LYING, 12.0)                # still on the floor, but answered: no alert
    assert seen == [ACKNOWLEDGED] and fired == 0, seen
    t, alert, seen, fired = _hold(p, t, STANDING, 1.5)
    assert seen == [ACKNOWLEDGED, IDLE], seen


def test_escalation_slow_descent_also_escalates():
    p = Pipeline("Knee Flexion")
    t = 0.0
    for _ in range(FPS):
        t += 1 / FPS; p.process(STANDING, t)
    n = 3 * FPS
    for i in range(1, n + 1):                                       # lowers to the floor over 3 s: no fast drop
        t += 1 / FPS; p.process(_pose(240 + 180 * i / n, 85 * i / n, lying_box=i > 2 * n // 3), t)
    t, alert, seen, fired = _hold(p, t, LYING, 5.5)
    assert alert["state"] == COUNTDOWN and not alert["impact"], (alert, p.fall.snapshot(t))
    assert alert["seconds_left"] > 5.5, alert                       # 10 s countdown, ~1.5 s already gone
    t, alert, seen, fired = _hold(p, t, LYING, 9.5)
    assert alert["state"] == ALERTED and fired == 1, (alert, fired)


if __name__ == "__main__":
    tests =[v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
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
