"""Synthetic-signal checks for angles, rep FSM / ROM, and fall transitions. No framework.

Run from repo root:  backend\\.venv\\Scripts\\python tests\\test_analytics.py
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.engines import BBox, Keypoint, PoseEngine, PoseResult  # noqa: E402
from app.inference import InferenceService  # noqa: E402
from app.pipeline.core import Pipeline  # noqa: E402
from app.pipeline.escalation import ACKNOWLEDGED, ALERTED, COUNTDOWN, IDLE  # noqa: E402
from app.pipeline.fall_detector import CONFIRMED, NORMAL, SUSPECTED, FallDetector  # noqa: E402
from app.pipeline.kinematics import EMA, angle_deg, joint_angles  # noqa: E402
from app.pipeline.rep_counter import RepCounter  # noqa: E402
from app.sessions import EXERCISE_NAMES, Session, SessionLimitError, SessionManager, to_reference  # noqa: E402
from app.voice.intent import classify, parse_intent  # noqa: E402
import os  # noqa: E402

from app.voice import groq_stt, recorder, settings  # noqa: E402

# The voice tests must not depend on the developer's own backend/.env or environment variables.
settings.ENV_FILE = Path(__file__).with_name("no-such-settings.env")
for _name in ("VOICE_MIN_RMS", "VOICE_TONE_SHARE", "VOICE_TRAIL_S", "VOICE_MIN_ON_S", "STT_INPUT_DEVICE"):
    os.environ.pop(_name, None)
from app.voice.listener import VoiceListener  # noqa: E402
from app.voice.recorder import (  # noqa: E402
    Segmenter, describe_inputs, parse_device_list, pick_device, resolve_device, to_wav_bytes, tonal_share)

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


def test_intent_classification():
    help_ = ["help", "HELP ME", "I need help", "call an ambulance", "I am not okay", "I'm not ok", "no",
             "I can't get up", "it hurts", "no I am fine"]        # the last is mixed: escalating is the safe side
    ok = ["I am okay", "I'M OKAY", "yes I am fine", "im alright", "i am all right", "I’m fine"]
    none = ["okay", "yes", "fine", "", "   ", "the weather is nice", "um"]
    for s in help_:
        assert classify(s) == "help", s
    for s in ok:
        assert classify(s) == "ok", s
    for s in none:
        assert classify(s) is None, s
    # Documents a known risk: the dialog's own spoken prompt contains "I'm OK" and would count as an answer.
    # The parser cannot tell, so the defence is the browser opening the microphone only AFTER the prompt ends.
    assert classify("Are you okay? Press I'm OK, or stand up.") == "ok"


class _FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, ({} if body is None else body)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _fake_post(response, calls=None):
    """Stands in for requests.post: records the request, then returns (or raises) `response`."""
    def post(url, **kwargs):
        if calls is not None:
            calls.append((url, kwargs))
        if isinstance(response, Exception):
            raise response
        return response
    return post


def test_groq_transcribe_sends_the_expected_request_and_returns_the_text():
    calls = []
    body = {"text": " I am fine.", "segments": [{"text": " I am fine.", "no_speech_prob": 0.01, "avg_logprob": -0.2}]}
    assert groq_stt.transcribe(b"RIFFwav", "test-key", post=_fake_post(_FakeResponse(200, body), calls)) == "I am fine."
    url, kw = calls[0]
    assert url == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert kw["headers"] == {"Authorization": "Bearer test-key"}
    assert kw["data"]["model"] == "whisper-large-v3" and kw["data"]["language"] == "en"
    assert kw["files"]["file"] == ("speech.wav", b"RIFFwav", "audio/wav")
    assert "prompt" not in kw["data"]       # a prompt such as "I am okay" could be echoed back on near-silent audio
    assert groq_stt.transcribe(b"x", "k", post=_fake_post(_FakeResponse(200, {"text": " help "}))) == "help"


def test_groq_drops_segments_whisper_doubts():
    junk = {"text": " Thank you.", "no_speech_prob": 0.9, "avg_logprob": -1.4}      # typical invention on silence
    real = {"text": " I am okay", "no_speech_prob": 0.02, "avg_logprob": -0.3}
    assert groq_stt.transcribe(b"x", "k", post=_fake_post(_FakeResponse(200, {"segments": [junk, real]}))) == "I am okay"
    assert groq_stt.transcribe(b"x", "k", post=_fake_post(_FakeResponse(200, {"segments": [junk]}))) == ""


def test_groq_failures_become_stt_error_and_never_leak_the_key():
    key = "SECRET-KEY-123"
    cases = ((_FakeResponse(401), "rejected"), (_FakeResponse(403), "rejected"), (_FakeResponse(429), "rate limit"),
             (_FakeResponse(500), "HTTP 500"), (_FakeResponse(200, ValueError("not json")), "unreadable"),
             (ConnectionError("no route to host"), "network"), (TimeoutError("slow"), "network"))
    for response, expected in cases:
        try:
            groq_stt.transcribe(b"x", key, post=_fake_post(response))
        except groq_stt.SttError as e:
            assert expected in str(e) and key not in str(e), (expected, str(e))
        else:
            raise AssertionError(f"{expected}: must raise SttError")


def test_api_key_from_env_then_dotenv():
    import os
    import tempfile
    saved = os.environ.pop("GROQ_API_KEY", None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            assert groq_stt.api_key(env) is None                        # no variable, no file
            env.write_text("# comment\nGROQ_API_KEY=key_from_file\nOTHER=1\n")
            assert groq_stt.api_key(env) == "key_from_file"
            os.environ["GROQ_API_KEY"] = "key_from_env"
            assert groq_stt.api_key(env) == "key_from_env"              # the environment wins
            os.environ["GROQ_API_KEY"] = "   "
            assert groq_stt.api_key(env) == "key_from_file"             # blank counts as unset
    finally:
        os.environ.pop("GROQ_API_KEY", None)
        if saved is not None:
            os.environ["GROQ_API_KEY"] = saved


def test_intent_accepts_sentences_cut_by_endpointing():
    # The recogniser sometimes ends a sentence early, so the leading "I" can be missing: "I" / "AM FINE".
    for s in ("am fine", "am okay", "AM FINE", "am alright", "i'm ok", "I'm fine", "all good", "doing fine",
              "doing okay", "I am all good", "yes I am fine"):
        assert classify(s) == "ok", s
    # A single word, or half a sentence, must never cancel the countdown.
    for s in ("okay", "ok", "OKAY", "yes", "fine", "good", "alright", "all", "I", "AM", "I AM", "I AM LOOKING", "I AM WICKED"):
        assert classify(s) is None, s
    # Any negative or help word wins, even next to an affirmative.
    for s in ("not", "I don't feel good", "it isn't fine", "I can't get up", "no", "pain", "am not fine",
              "I am fine but it hurts", "am fine no", "I am not okay", "help I am fine"):
        assert classify(s) == "help", s


def test_voice_listener_acts_on_transcribed_text():
    acted = []
    v = VoiceListener(on_ok=lambda: acted.append("ok"), on_help=lambda: acted.append("help"))
    for text, intent in (("I am fine.", "ok"), ("Help!", "help"), ("No, I am fine.", "help"), ("Okay.", None),
                         ("Thank you.", None), ("", None), ("I am okay", "ok"), ("I'm not okay.", "help")):
        acted.clear()
        assert v.handle(text) == intent, text
        assert v.transcript == text and v.intent == intent
        assert acted == ([intent] if intent else []), (text, acted)


def test_voice_listener_joins_a_sentence_cut_by_a_pause():
    acted = []
    v = VoiceListener(on_ok=lambda: acted.append("ok"), on_help=lambda: acted.append("help"))
    assert v.handle("I am", now=0.0) is None and acted == []
    assert v.handle("fine.", now=1.5) == "ok" and acted == ["ok"]            # one sentence, cut at the pause
    acted.clear()
    assert v.handle("I am", now=10.0) is None
    assert v.handle("fine.", now=20.0) is None and acted == []               # too long ago: not the same sentence
    assert v.handle("fine.", now=20.5) is None                               # a bare word still never counts
    assert v.handle("I am okay", now=21.0) == "ok"
    assert v.handle("   ", now=30.0) is None and v.transcript == "   "       # silence changes nothing
    v2 = VoiceListener(on_ok=lambda: acted.append("ok"), on_help=lambda: acted.append("help"))
    assert v2.handle("it hurts", now=0.0) == "help" and acted[-1] == "help"  # a distress word still acts at once


_FAKE_DEVICES = [       # this machine's input devices as sounddevice lists them (default input: the silent DroidCam one)
    {"name": "Microsoft Sound Mapper - Input", "hostapi": 0, "max_input_channels": 2, "default_samplerate": 44100.0},
    {"name": "Microphone (DroidCam Audio)", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 44100.0},
    {"name": "Microphone (Realtek Audio)", "hostapi": 0, "max_input_channels": 2, "default_samplerate": 44100.0},
    {"name": "Microphone (2- KORP THALLIUM)", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 44100.0},
    {"name": "Speakers (Realtek Audio)", "hostapi": 0, "max_input_channels": 0, "default_samplerate": 44100.0},
    {"name": "Microphone (2- KORP THALLIUM)", "hostapi": 1, "max_input_channels": 1, "default_samplerate": 48000.0},
    {"name": "WASAPI Only Mic", "hostapi": 1, "max_input_channels": 1, "default_samplerate": 48000.0},
]


def _fake_sounddevice(default=1, InputStream=None):
    """A stand-in for the sounddevice module with the device list above."""
    import types

    def query_devices(device=None, kind=None):
        if isinstance(device, int):
            return dict(_FAKE_DEVICES[device], index=device)
        if device is None and kind is None:
            return [dict(d, index=i) for i, d in enumerate(_FAKE_DEVICES)]
        return dict(_FAKE_DEVICES[default], index=default)               # kind="input": the system default

    return types.SimpleNamespace(query_devices=query_devices, query_hostapis=lambda i: {"name": ["MME", "WASAPI"][i]},
                                 InputStream=InputStream)


def test_parse_device_list():
    assert parse_device_list("3") == [3] and parse_device_list(" 3 ") == [3]
    assert parse_device_list("KORP THALLIUM") == ["KORP THALLIUM"]
    assert parse_device_list('"KORP THALLIUM"') == ["KORP THALLIUM"]                  # quotes are ignored
    assert parse_device_list("KORP THALLIUM, 2 ; Realtek") == ["KORP THALLIUM", 2, "Realtek"]
    assert parse_device_list("") == [] and parse_device_list(None) == [] and parse_device_list(" , ; ") == []


def test_resolve_device_binds_the_korp_thallium_headset_however_it_is_written():
    sd = _fake_sounddevice()
    for written in (3, "KORP THALLIUM", "korp thallium", "KORPTHALLIUM", "Korp-Thallium", "2- korp",
                    "Microphone (2- KORP THALLIUM)"):
        assert resolve_device(sd, written) == 3, written      # never its copy under the other audio API (index 5)
    assert resolve_device(sd, "wasapi only") == 6             # only in another audio API: found there
    for bad in (4, 99, "Speakers", "no such headset"):        # an output-only device, a missing index, unknown names
        try:
            resolve_device(sd, bad)
        except ValueError as e:
            assert "[3] Microphone (2- KORP THALLIUM)" in str(e) and "--devices" in str(e), (bad, str(e))
        else:
            raise AssertionError(f"{bad!r} must not resolve")


def test_pick_device_falls_back_down_the_list_and_describes_the_devices():
    sd = _fake_sounddevice()
    assert pick_device(sd, []) == (None, "Microphone (DroidCam Audio)")             # nothing set: the system default
    assert pick_device(sd, ["KORP THALLIUM"]) == (3, "Microphone (2- KORP THALLIUM)")
    assert pick_device(sd, ["Missing Headset", 3, "Realtek"]) == (3, "Microphone (2- KORP THALLIUM)")   # skip a missing one
    try:
        pick_device(sd, ["Missing Headset", 42])
    except ValueError as e:
        assert "none of the configured input devices" in str(e) and "[3] Microphone (2- KORP THALLIUM)" in str(e)
    else:
        raise AssertionError("nothing found must raise, listing the devices")
    text = describe_inputs(sd)
    assert "[1] Microphone (DroidCam Audio)" in text and "<- Windows default" in text
    assert "Speakers" not in text and "WASAPI" in text                              # inputs only, with their audio API


def test_open_stream_binds_the_headset_and_tries_other_sample_rates():
    import numpy as np
    opened = []

    class PickyStream:
        def __init__(self, **kwargs):
            if kwargs["samplerate"] != 48000:
                raise RuntimeError("Invalid sample rate [PaErrorCode -9997]")
            opened.append(kwargs)

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    with recorder.open_stream(_fake_sounddevice(InputStream=PickyStream), ["KORP THALLIUM"]) as (blocks, rate, name):
        assert (rate, name) == (48000, "Microphone (2- KORP THALLIUM)")           # 44100 refused, 48000 accepted
        assert opened[-1]["device"] == 3 and opened[-1]["channels"] == 1
        opened[-1]["callback"](np.ones((4, 1), np.float32), 4, None, None)
        assert blocks.get_nowait().tolist() == [1.0] * 4

    class Refusing:
        def __init__(self, **kwargs):
            raise RuntimeError("Device unavailable")

    try:
        with recorder.open_stream(_fake_sounddevice(InputStream=Refusing), ["KORP THALLIUM"]):
            pass
    except RuntimeError as e:
        assert "Device unavailable" in str(e)                                       # the real reason, not a vague one
    else:
        raise AssertionError("a device that cannot be opened must raise")


def _run_voice_with_fake_microphone(deliver_silence):
    """Runs VoiceListener.start() against a fake sounddevice: the stream opens, but it delivers either nothing at all or
    digital silence (like a muted or virtual microphone). Returns (listener once it gave up, status right after start)."""
    import sys
    import threading
    import time
    import types
    import numpy as np
    from app.voice import listener as listener_mod

    class FakeStream:
        def __init__(self, callback=None, **kwargs):
            self.callback, self.running = callback, False

        def start(self):
            self.running = True
            if deliver_silence:
                def pump():
                    while self.running:
                        self.callback(np.zeros((320, 1), np.float32), 320, None, None)
                        time.sleep(0.02)
                threading.Thread(target=pump, daemon=True).start()

        def stop(self):
            self.running = False

        def close(self):
            self.running = False

    fake = _fake_sounddevice(InputStream=FakeStream)
    saved = (sys.modules.get("sounddevice"), groq_stt.api_key, listener_mod.WATCHDOG_S)
    sys.modules["sounddevice"] = fake
    groq_stt.api_key = lambda *a, **k: "test-key"
    listener_mod.WATCHDOG_S = 0.3
    try:
        v = VoiceListener(on_ok=lambda: None, on_help=lambda: None)
        v.start()
        first = v.status
        deadline = time.time() + 5
        while not v.warning and v.status in ("starting", "listening") and time.time() < deadline:
            time.sleep(0.05)
        return v, first
    finally:
        if saved[0] is None:
            sys.modules.pop("sounddevice", None)
        else:
            sys.modules["sounddevice"] = saved[0]
        groq_stt.api_key, listener_mod.WATCHDOG_S = saved[1], saved[2]


def test_voice_gives_up_when_the_microphone_delivers_no_audio():
    v, first = _run_voice_with_fake_microphone(deliver_silence=False)
    assert first == "starting"                                  # never "listening" before any audio has arrived
    assert v.status == "unavailable" and "no audio" in v.error, (v.status, v.error)


def test_voice_keeps_listening_and_warns_when_the_microphone_is_silent():
    v, _ = _run_voice_with_fake_microphone(deliver_silence=True)     # no STT_INPUT_DEVICE: the silent DroidCam default
    try:
        # a headset that is silent until you speak (noise gate) is NOT a failure: it warns and stays "listening"
        assert v.status == "listening" and v.error is None, (v.status, v.error)
        assert v.warning and "No sound" in v.warning and "DroidCam Audio" in v.warning, v.warning
    finally:
        v.stop()


def test_voice_is_unavailable_without_a_key_and_starts_no_thread():
    saved = groq_stt.api_key
    groq_stt.api_key = lambda *a, **k: None
    try:
        v = VoiceListener(on_ok=lambda: None, on_help=lambda: None)
        v.init()
        assert v.status == "unavailable" and "GROQ_API_KEY" in v.error
        v.start()                                              # the dialog asks for the microphone: still unavailable
        assert v.status == "unavailable" and not v.active and v._thread is None
        assert v.snapshot() == {"status": "unavailable", "transcript": "", "intent": None, "error": v.error,
                                "warning": None}
    finally:
        groq_stt.api_key = saved


def _noise(rng, seconds, rate=16000, amp=0.1):
    import numpy as np
    return rng.normal(0, amp, int(seconds * rate)).astype(np.float32)


def _feed_all(seg, audio, block=800):
    out = []
    for i in range(0, len(audio), block):
        u = seg.feed(audio[i:i + block])
        if u is not None:
            out.append(u)
    return out


def test_segmenter_returns_one_utterance_with_preroll_and_trailing_silence():
    import numpy as np
    rng = np.random.default_rng(1)
    audio = np.concatenate([np.zeros(8000, np.float32), _noise(rng, 0.6), np.zeros(16000, np.float32)])
    utterances = _feed_all(Segmenter(16000), audio)
    assert len(utterances) == 1
    assert 1.1 < len(utterances[0]) / 16000 < 1.6      # 0.4 s pre-roll + speech + 0.4 s of trailing silence


def test_segmenter_ignores_beeps_and_quiet_noise():
    import numpy as np
    rng = np.random.default_rng(2)
    t = np.arange(int(0.12 * 16000)) / 16000
    beep = (0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)     # a countdown tick: 0.12 s, far shorter than speech
    gap = np.zeros(16000, np.float32)
    assert _feed_all(Segmenter(16000), np.concatenate([gap, beep, gap, beep, gap])) == []
    assert _feed_all(Segmenter(16000), _noise(rng, 3.0, amp=0.003)) == []     # a quiet room is not speech


def test_segmenter_caps_a_long_utterance():
    import numpy as np
    utterances = _feed_all(Segmenter(16000), _noise(np.random.default_rng(3), 7.0))
    assert utterances and 4.9 < len(utterances[0]) / 16000 < 5.2


def _voiced_like(seconds, rng, rate=16000, f0=200, rms=0.03):
    """A rough stand-in for a vowel: harmonics of f0 with a speech-like tilt, plus breath noise."""
    import numpy as np
    t = np.arange(int(seconds * rate)) / rate
    x = sum(np.sin(2 * np.pi * f0 * h * t) / np.sqrt(h) for h in range(1, 16)) + 0.2 * rng.normal(0, 1, len(t))
    return (x * rms / x.std()).astype(np.float32)


def test_tonal_share_separates_beeps_from_speechlike_sound():
    import numpy as np
    rng = np.random.default_rng(5)
    beep = (0.3 * np.sin(2 * np.pi * 880 * np.arange(320) / 16000)).astype(np.float32)     # one 20 ms frame of a beep
    assert tonal_share(beep) > 0.9
    assert tonal_share(_voiced_like(0.02, rng)) < 0.6          # harmonics + breath noise: spread over many bins
    assert tonal_share(_noise(rng, 0.02)) < 0.3                # a noise burst such as the "h" or "p" of "help"


def test_segmenter_keeps_a_quick_single_word_like_help():
    import numpy as np
    rng = np.random.default_rng(6)
    word = _voiced_like(0.14, rng, rms=0.02)                   # only ~0.14 s of voiced sound, as in a sharp "help"
    audio = np.concatenate([np.zeros(8000, np.float32), word, np.zeros(19200, np.float32)])
    utterances = _feed_all(Segmenter(16000), audio)
    assert len(utterances) == 1
    assert 0.7 < len(utterances[0]) / 16000 < 1.2


def test_utterance_ends_after_the_configured_silence():
    import numpy as np
    rng = np.random.default_rng(8)
    word = _voiced_like(0.2, rng, rms=0.03)
    lead = np.zeros(8000, np.float32)

    def after(silence_s, **settings_):
        audio = np.concatenate([lead, word, np.zeros(int(silence_s * 16000), np.float32)])
        return _feed_all(Segmenter(16000, **settings_), audio)

    assert Segmenter(16000).trail_s == 0.4                      # the default: a one-word answer goes out after 0.4 s
    assert after(0.3) == [] and len(after(0.5)) == 1            # not yet at 0.3 s of silence, done by 0.5 s
    assert after(0.45, trail_s=0.5) == [] and len(after(0.6, trail_s=0.5)) == 1     # the setting is respected
    assert len(after(0.35, trail_s=0.3)) == 1


def test_segmenter_catches_a_very_short_word_but_not_a_click():
    import numpy as np
    rng = np.random.default_rng(9)
    zeros = np.zeros(8000, np.float32)
    assert Segmenter(16000).min_on_s == 0.06 and Segmenter(16000)._min_on == 3     # 3 frames of 20 ms
    word = _voiced_like(0.08, rng, rms=0.03)                    # 4 frames: a clipped, sharp "help" or "OK"
    assert len(_feed_all(Segmenter(16000), np.concatenate([zeros, word, np.zeros(16000, np.float32)]))) == 1
    click = _voiced_like(0.04, rng, rms=0.03)                   # 2 frames: a tap or a click
    seg = Segmenter(16000)
    assert _feed_all(seg, np.concatenate([zeros, click, np.zeros(16000, np.float32)])) == []
    assert seg.ignored and seg.ignored[-1]["s"] < 0.06          # remembered, so the smoke test can list it
    assert len(_feed_all(Segmenter(16000, min_on_s=0.04), np.concatenate([zeros, click, np.zeros(16000, np.float32)]))) == 1


def test_voice_settings_read_the_environment_then_dotenv():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        env = Path(tmp) / ".env"
        env.write_text("VOICE_TRAIL_S=0.3\nSTT_INPUT_DEVICE=Realtek\nVOICE_MIN_RMS=oops\n")
        assert settings.get_float("VOICE_TRAIL_S", 0.4, env) == 0.3          # from the file
        assert settings.get("STT_INPUT_DEVICE", None, env) == "Realtek"
        assert settings.get_float("VOICE_MIN_RMS", 0.008, env) == 0.008      # not a number: the default
        assert settings.get_float("VOICE_TONE_SHARE", 0.65, env) == 0.65     # absent: the default
        os.environ["VOICE_TRAIL_S"] = "0.35"
        try:
            assert settings.get_float("VOICE_TRAIL_S", 0.4, env) == 0.35     # the environment wins
        finally:
            os.environ.pop("VOICE_TRAIL_S", None)
    os.environ["STT_INPUT_DEVICE"] = "3"
    try:
        assert recorder.device_preferences() == [3]
        os.environ["STT_INPUT_DEVICE"] = "KORP THALLIUM, Realtek"
        assert recorder.device_preferences() == ["KORP THALLIUM", "Realtek"]
    finally:
        os.environ.pop("STT_INPUT_DEVICE", None)
    assert recorder.device_preferences() == []


def test_beeps_are_ignored_and_do_not_hide_the_next_word():
    import numpy as np
    rng = np.random.default_rng(7)
    t = np.arange(int(0.15 * 16000)) / 16000
    beep = (0.3 * np.sign(np.sin(2 * np.pi * 880 * t))).astype(np.float32)     # the dialog's tick is a square wave
    gap = np.zeros(8000, np.float32)
    word = _voiced_like(0.25, rng, rms=0.03)
    seg = Segmenter(16000)
    # two ticks as long as a quick "help", then the word: only the word may come out, and the ticks must not raise
    # the ambient estimate (which would hide the word)
    utterances = _feed_all(seg, np.concatenate([gap, beep, gap, beep, np.zeros(4800, np.float32), word,
                                                np.zeros(19200, np.float32)]))
    assert len(utterances) == 1, (len(utterances), seg.tone_frames, seg.threshold)
    assert seg.tone_frames >= 8


def test_wav_is_16khz_mono_and_normalised():
    import io
    import wave
    import numpy as np
    rate = 48000

    def encode(amplitude):
        x = (amplitude * np.sin(2 * np.pi * 300 * np.arange(rate) / rate)).astype(np.float32)   # 1 s tone
        with wave.open(io.BytesIO(to_wav_bytes(x, rate))) as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
            assert abs(w.getnframes() - 16000) <= 2
            return abs(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)).max() / 32767

    assert 0.6 < encode(0.15) < 0.75        # a quiet microphone is brought up to a clear level, without clipping
    assert 0.3 < encode(0.05) < 0.45        # ... but by at most 8x


def test_pipeline_from_speech_audio_to_action_with_mocked_whisper():
    import numpy as np
    rng = np.random.default_rng(4)
    audio = np.concatenate([np.zeros(8000, np.float32), _noise(rng, 1.0), np.zeros(16000, np.float32)])
    utterances = _feed_all(Segmenter(16000), audio)
    assert len(utterances) == 1
    wav = to_wav_bytes(utterances[0], 16000)
    acted = []
    v = VoiceListener(on_ok=lambda: acted.append("ok"), on_help=lambda: acted.append("help"))
    text = groq_stt.transcribe(wav, "k", post=_fake_post(_FakeResponse(200, {"text": "I am okay."})))
    assert v.handle(text) == "ok" and acted == ["ok"]
    text = groq_stt.transcribe(wav, "k", post=_fake_post(_FakeResponse(200, {"text": "Help me, I fell."})))
    assert v.handle(text) == "help" and acted == ["ok", "help"]


def test_intent_matches_help_loosely():
    for s in ("help", "Help!", "HELP ME", "helpi", "aelp", "elp", "halp", "yelp", "hep", "helps", "helping",
              "help me please", "bachao", "Bachao!", "bhachao", "bacchao", "madad", "emergency", "hell", "held",
              "um help um", "i need help now"):
        assert classify(s) == "help", s
    for s in ("hello", "hey", "he'll be fine", "hold", "yes", "okay", "the weather is nice", "um"):
        assert classify(s) is None, s                          # near the word, but not close enough to count
    assert classify("I am well") == "ok" and classify("I am fine") == "ok"      # the loose match leaves answers alone
    assert parse_intent is classify                             # the name the plans use


def test_intent_accepts_phonetic_near_misses_only_after_i_am():
    for s in ("i am fine", "i'm fine", "am fine", "i am five", "i am kind", "i am oky", "i am okay", "i'm okay",
              "am okay", "I AM OKEY", "I AM FIGHT", "I'M KIND", "AM FIVE", "i am well", "i am all good", "am alright"):
        assert classify(s) == "ok", s
    # the same words alone, or without "am", never count as an answer
    for s in ("five", "kind", "fight", "well", "oky", "okey", "ok", "okay", "yes", "fine", "good", "kind of",
              "I have five", "fine five", "i five"):
        assert classify(s) is None, s
    # every negative or distress word still wins, however affirmative the rest sounds
    for s in ("no i am five", "help", "i am fine help", "i am not fine", "i am not kind", "i can't", "it hurts", "pain",
              "i fell", "i had a fall", "i am stuck", "i feel dizzy", "i feel sick", "this is bad", "i have fallen",
              "emergency", "i am fine but i fell"):
        assert classify(s) == "help", s


def test_intent_near_misses_seen_in_live_testing():
    # a soft "help" comes out as "hell" or "held" on a laptop mic: still an emergency
    for s in ("hell", "HELL", "held", "HELD", "what the hell", "no i am ok", "hell i am ok"):
        assert classify(s) == "help", s
    assert classify("hello") is None                           # a different word: not an emergency
    for s in ("i am ok", "i'm ok", "am ok", "i am oky", "i am oke", "i am okey", "i am ache", "i'm ache", "AM ACHE"):
        assert classify(s) == "ok", s
    for s in ("ok", "okay", "oke", "ache", "i ache", "my back ache"):
        assert classify(s) is None, s                          # a bare word, or no "am" in front, never counts


def test_resolve_device_uses_default_audio_api_only():
    class FakeSD:                                       # the shape of this machine's device list
        def query_devices(self, device=None, kind=None):
            if kind == "input":
                return {"index": 1, "hostapi": 0, "name": "Microphone (DroidCam Audio)", "max_input_channels": 1}
            return [
                {"name": "Microsoft Sound Mapper - Input", "hostapi": 0, "max_input_channels": 2},
                {"name": "Microphone (DroidCam Audio)", "hostapi": 0, "max_input_channels": 1},
                {"name": "Microphone (Realtek Audio)", "hostapi": 0, "max_input_channels": 2},
                {"name": "Microphone (Realtek Audio)", "hostapi": 1, "max_input_channels": 2},   # same mic, other API
                {"name": "Speakers (Realtek Audio)", "hostapi": 0, "max_input_channels": 0},     # output only
            ]
    sd = FakeSD()
    assert resolve_device(sd, None) is None             # keep the Windows default
    assert resolve_device(sd, 3) == 3                   # a number is used as given
    assert resolve_device(sd, "realtek") == 2           # not the speakers, not the other audio API's copy
    assert resolve_device(sd, "DROIDCAM") == 1          # case-insensitive
    try:
        resolve_device(sd, "no such microphone")
    except ValueError as e:
        assert "--devices" in str(e)
    else:
        raise AssertionError("an unknown name must raise")


FLOOR = _floor_pose()       # flat on the floor, all joints tracked: stays "down" however long the test runs


def test_escalate_skips_the_countdown():
    p = Pipeline("Knee Flexion")
    t, alert, seen, fired = _hold(p, 0.0, FLOOR, 5.5)               # down for the 5 s dwell: countdown starts
    assert alert["state"] == COUNTDOWN and alert["rechecks"] == 0, alert
    p.escalation.escalate()                                         # voice "help" / Trigger Emergency button
    t, alert, seen, fired = _hold(p, t, FLOOR, 0.2)
    assert seen == [ALERTED] and fired == 1, (seen, fired)


def test_escalate_while_idle_is_ignored_and_does_not_pre_arm():
    p = Pipeline("Knee Flexion")
    p.escalation.escalate()
    t, alert, seen, fired = _hold(p, 0.0, STANDING, 0.5)
    assert seen == [IDLE] and fired == 0, seen
    t, alert, seen, fired = _hold(p, _fall(p, t), LYING, 5.5)       # a later real fall still gets its countdown
    assert alert["state"] == COUNTDOWN and fired == 0, alert


def test_recheck_when_still_down_after_acknowledge():
    p = Pipeline("Knee Flexion")
    t, alert, seen, fired = _hold(p, 0.0, FLOOR, 5.5)
    assert alert["state"] == COUNTDOWN
    p.escalation.ack()                                              # "I'm OK" (button or voice), but still on the floor
    t, alert, seen, fired = _hold(p, t, FLOOR, 119.0)               # quiet for just under two minutes
    assert seen == [ACKNOWLEDGED] and fired == 0 and alert["rechecks"] == 0, (seen, alert)
    t, alert, seen, fired = _hold(p, t, FLOOR, 2.0)                 # two minutes since the answer: asked again
    assert seen == [ACKNOWLEDGED, COUNTDOWN] and alert["rechecks"] == 1, (seen, alert)
    assert 7 < alert["seconds_left"] <= 10, alert                   # a fresh 10 s countdown
    t, alert, seen, fired = _hold(p, t, FLOOR, 10.5)                # unanswered again: the alarm
    assert seen == [COUNTDOWN, ALERTED] and fired == 1, (seen, fired)


def test_recheck_never_fires_if_the_person_got_up():
    p = Pipeline("Knee Flexion")
    t, alert, seen, fired = _hold(p, 0.0, FLOOR, 5.5)
    p.escalation.ack()
    t, alert, seen, fired = _hold(p, t, FLOOR, 30.0)
    t, alert, seen, fired = _hold(p, t, STANDING, 2.0)              # stands up inside the two minutes
    assert seen == [ACKNOWLEDGED, IDLE] and alert["rechecks"] == 0, (seen, alert)   # IDLE once upright 0.5 s
    t, alert, seen, fired = _hold(p, t, STANDING, 130.0)
    assert seen == [IDLE] and fired == 0, seen


def test_recheck_counts_each_round():
    p = Pipeline("Knee Flexion")
    t, alert, seen, fired = _hold(p, 0.0, FLOOR, 5.5)
    for expected in (1, 2):
        p.escalation.ack()
        t, alert, seen, fired = _hold(p, t, FLOOR, 121.0)
        assert alert["state"] == COUNTDOWN and alert["rechecks"] == expected, (expected, alert)


def test_test_fall_simulation_unchanged_and_never_rechecks():
    p = Pipeline("Knee Flexion")
    p.escalation.simulate()                                         # the Test Fall button
    t, alert, seen, fired = _hold(p, 0.0, STANDING, 1.0)
    assert alert["state"] == COUNTDOWN, alert                       # standing does not cancel a simulated fall
    p.escalation.ack()
    t, alert, seen, fired = _hold(p, t, STANDING, 0.5)
    assert seen == [ACKNOWLEDGED, IDLE], seen
    t, alert, seen, fired = _hold(p, t, STANDING, 130.0)
    assert seen == [IDLE] and fired == 0, seen


class _FakeVoice:
    """Stands in for VoiceListener: records what the session asks of it."""
    def __init__(self):
        self.active, self.started, self.stopped, self.quit_called = False, 0, 0, False
        self.status, self.error = "idle", None

    def start(self):
        self.started += 1
        self.active = True

    def stop(self):
        self.stopped += 1
        self.active = False

    def quit(self):
        self.quit_called = True

    def snapshot(self):
        return {"status": self.status, "transcript": "", "intent": None, "error": None, "warning": None}


def test_sessions_do_not_share_pipeline_state():
    a, b = Session("a"), Session("b")
    for i in range(1, int(5.5 * FPS) + 1):             # a lies flat on the floor through the 5 s dwell; b just stands
        a.process(FLOOR, i / FPS)
        b.process(STANDING, i / FPS)
    assert a.telemetry()["fall"]["alert"]["state"] == "COUNTDOWN", a.telemetry()["fall"]
    assert b.telemetry()["fall"]["alert"]["state"] == "IDLE" and b.telemetry()["fall"]["status"] == "NORMAL"
    a.ack()                                             # answering a's countdown does not touch b
    a.process(FLOOR, 6.0)
    b.process(STANDING, 6.0)
    assert a.telemetry()["fall"]["alert"]["state"] == "ACKNOWLEDGED"
    assert b.telemetry()["fall"]["alert"]["state"] == "IDLE"
    a.request_exercise("Knee Flexion")                  # nor does changing a's exercise
    a.apply_pending()
    assert a.pipeline.exercise == "Knee Flexion" and b.pipeline.exercise == EXERCISE_NAMES[0]
    assert a.pipeline is not b.pipeline and a.pipeline.escalation is not b.pipeline.escalation


def test_session_commands_only_set_flags_until_the_feeder_applies_them():
    s = Session("s")
    assert s.pipeline.exercise == "Elbow Curl"
    s.request_exercise("Hip Flexion")
    assert s.pipeline.exercise == "Elbow Curl" and s.pending_exercise == "Hip Flexion"   # a flag, nothing applied yet
    s.apply_pending()
    assert s.pipeline.exercise == "Hip Flexion" and s.pending_exercise is None
    try:
        s.request_exercise("Nope")
    except ValueError as e:
        assert "exercise must be one of" in str(e) and s.pending_exercise is None
    else:
        raise AssertionError("an unknown exercise must be refused")
    s.pipeline.reps.limbs["L"].reps = 3
    s.request_reset()
    assert s.pipeline.reps.snapshot()["L"]["reps"] == 3 and s.pending_reset              # still only a flag
    s.apply_pending()
    assert s.pipeline.reps.snapshot()["L"]["reps"] == 0 and not s.pending_reset
    try:
        Session("bad", "Nope")
    except ValueError:
        pass
    else:
        raise AssertionError("a session cannot start on an unknown exercise")


def test_session_voice_follows_the_fall_dialog():
    s = Session("s")
    s.voice = _FakeVoice()
    assert s.listen() == {"status": "idle", "error": None} and s.voice.started == 0      # no fall check: no microphone
    s.simulate_fall()
    s.process(STANDING, 0.1)
    assert s.telemetry()["fall"]["alert"]["state"] == "COUNTDOWN" and "voice" in s.telemetry()["fall"]
    s.listen()                                                                           # the dialog's prompt has ended
    assert s.voice.started == 1 and s.voice.active
    s.process(STANDING, 0.2)
    assert s.voice.stopped == 0                                                          # dialog still open: keep listening
    s.ack()
    s.process(STANDING, 0.3)
    assert s.telemetry()["fall"]["alert"]["state"] == "ACKNOWLEDGED"
    assert s.voice.stopped == 1 and not s.voice.active                                   # dialog gone: microphone closed
    quiet = Session("no voice")                                                          # browser-voice sessions have none
    quiet.process(STANDING, 0.1)
    assert "voice" not in quiet.telemetry()["fall"]
    assert quiet.listen()["status"] == "unavailable"


def test_session_manager_lifecycle():
    m = SessionManager(max_sessions=3)
    a, b = m.create("a"), m.create()                    # an id is generated when none is given
    assert len(m) == 2 and m.get("a") is a and b.id in m and a.id != b.id and m.get("missing") is None
    for call in (lambda: m.create("a"), lambda: m.create("x", exercise="Nope")):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("a duplicate id or an unknown exercise must be refused")
    assert "x" not in m and len(m) == 2                 # a refused session is never registered
    c = m.create("c")
    try:
        m.create("d")
    except SessionLimitError:
        pass
    else:
        raise AssertionError("the limit must be enforced")
    a.voice = _FakeVoice()
    assert m.close("a") is True and a.closed and a.voice.quit_called and "a" not in m
    assert m.close("a") is False                        # closing twice is harmless
    a.close()
    m.create("d")                                       # room again
    assert sorted(m.ids()) == sorted([b.id, "c", "d"])
    m.close_all()
    assert len(m) == 0 and b.closed and c.closed


def _scaled(res, k):
    return PoseResult({n: Keypoint(p.x * k, p.y * k, p.conf) for n, p in res.keypoints.items()},
                      BBox(res.bbox.x * k, res.bbox.y * k, res.bbox.w * k, res.bbox.h * k, res.bbox.conf), res.timestamp_ms)


def test_to_reference_scales_pose_to_the_640_wide_frame():
    res = _pose(240, 0)
    assert to_reference(res, 640) is res and to_reference(None, 1280) is None and to_reference(res, 0) is res
    big = to_reference(res, 1280)
    src, out = res.keypoints["l_hip"], big.keypoints["l_hip"]
    assert (out.x, out.y, out.conf) == (src.x / 2, src.y / 2, src.conf)
    assert (big.bbox.x, big.bbox.y, big.bbox.w, big.bbox.h) == (res.bbox.x / 2, res.bbox.y / 2, res.bbox.w / 2, res.bbox.h / 2)
    assert big.bbox.conf == res.bbox.conf and big.timestamp_ms == res.timestamp_ms
    assert res.keypoints["l_hip"].x == src.x            # the input is not modified
    for width in (320, 1280):                           # a phone's frame size must not change the verdict
        plain, phone = Session("plain"), Session("phone")
        for i in range(1, int(5.5 * FPS) + 1):
            plain.process(FLOOR, i / FPS)
            phone.process(_scaled(FLOOR, width / 640), i / FPS, frame_width=width)
        assert phone.telemetry()["fall"] == plain.telemetry()["fall"], width
        assert phone.telemetry()["fall"]["alert"]["state"] == "COUNTDOWN"


def test_desktop_routes_still_work_through_the_session():
    from fastapi.testclient import TestClient
    import app.main as main
    client = TestClient(main.app)                       # no `with`: the lifespan (camera, engine, microphone) never starts
    s = main.local_session
    assert main.sessions.get("local") is s and main.worker.session is s
    assert client.get("/api/exercises").json() == EXERCISE_NAMES
    info = client.get("/api/exercise_info").json()
    assert set(info) == set(EXERCISE_NAMES) and set(info["Elbow Curl"]) == {"rest", "peak", "sides"}
    t = client.get("/api/telemetry").json()
    for key in ("camera_index", "camera_backend", "camera_error", "engine", "device", "fps", "latency_ms", "timings",
                "privacy", "pose_visible", "error", "exercise"):
        assert key in t, key                            # every field the frontend reads is still there
    assert client.get("/api/health").json()["status"] == "ok"
    # exercise: refused with the same HTTP 400 and message as before, applied by the feeding thread
    assert client.post("/api/config", json={"exercise": "Elbow Curl"}).status_code == 200
    assert s.pending_exercise == "Elbow Curl"
    bad = client.post("/api/config", json={"exercise": "Nope"})
    assert bad.status_code == 400 and "exercise must be one of" in bad.json()["detail"]
    assert client.post("/api/config", json={"engine": "nope"}).status_code == 400
    assert client.post("/api/config", json={"privacy": True}).json()["privacy"] is False    # not confirmed until a blanked frame
    client.post("/api/config", json={"privacy": False})
    assert client.post("/api/reset").json() == {"ok": True} and s.pending_reset
    s.apply_pending()                                   # what the camera thread does between frames
    assert s.pipeline.exercise == "Elbow Curl" and not s.pending_reset
    # the fall flow, through the routes
    assert client.post("/api/fall/test").json() == {"ok": True}
    s.process(_pose(240, 0), 1.0, infer_ms=12.3)
    t = client.get("/api/telemetry").json()
    assert t["fall"]["alert"]["state"] == "COUNTDOWN" and "voice" in t["fall"]
    assert t["latency_ms"] == 12.3 and t["pose_visible"] is True
    client.post("/api/fall/escalate")
    s.process(_pose(240, 0), 1.1)
    assert client.get("/api/telemetry").json()["fall"]["alert"]["state"] == "ALERTED"
    client.post("/api/fall/ack")
    s.process(_pose(240, 0), 1.2)
    assert client.get("/api/telemetry").json()["fall"]["alert"]["state"] == "IDLE"
    r = client.post("/api/fall/listen").json()         # no fall check open: the microphone is not started
    assert r["ok"] is True and set(r) == {"ok", "status", "error"}


import json  # noqa: E402
import struct  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402


def test_reference_scale_uses_the_long_side_so_portrait_and_landscape_agree():
    res = _pose(240, 0)
    landscape, portrait = to_reference(res, 1280, 960), to_reference(res, 960, 1280)
    hip = res.keypoints["l_hip"].x
    assert landscape.keypoints["l_hip"].x == portrait.keypoints["l_hip"].x == hip / 2
    assert to_reference(res, 640, 480) is res and to_reference(res, 480, 640) is res    # already at the reference scale
    a, b = Session("a"), Session("b")
    for i in range(1, int(5.5 * FPS) + 1):
        a.process(FLOOR, i / FPS, frame_width=640, frame_height=480)
        b.process(_scaled(FLOOR, 1.0), i / FPS, frame_width=480, frame_height=640)      # a portrait frame: same scale
    assert a.telemetry()["fall"] == b.telemetry()["fall"]


def test_session_baseline_reset_is_a_flag_until_the_feeder_applies_it():
    s = Session("s")
    s.process(_pose(240, 0), 0.1)
    assert len(s.pipeline.fall._hist) == 1
    s.request_baseline_reset()
    assert s.pending_baseline_reset and len(s.pipeline.fall._hist) == 1               # only a flag so far
    s.apply_pending()
    assert not s.pending_baseline_reset and len(s.pipeline.fall._hist) == 0


# ---- Phase 2: multi-session inference ---------------------------------------------------------------------------

FRAME_FLOOR, FRAME_STANDING, FRAME_NOBODY, FRAME_BAD = 0, 100, 200, 250    # pixel values the test engine reads as scenes


def _frame(value, h=480, w=640):
    import numpy as np
    return np.full((h, w, 3), value, np.uint8)


def _wait_until(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class _SceneEngine(PoseEngine):
    """A pose engine for tests. The frame's pixel value picks the scene (FRAME_FLOOR / FRAME_STANDING / FRAME_NOBODY;
    FRAME_BAD raises) and the pose comes back in the frame's own pixel space, scaled by its long side, like a real one."""
    name = "scene"

    def __init__(self, stateless=True, gate=None):
        self.stateless = stateless
        self.gate = gate                    # an Event: until it is set, infer_batch blocks (the engine is "busy")
        self.entered = threading.Event()
        self.batches, self.seen, self.closed = [], [], False

    def load(self):
        pass

    def close(self):
        self.closed = True

    def infer(self, frame, timestamp_ms):
        value = int(frame[0, 0, 0])
        self.seen.append(value)
        if value > 225:
            raise RuntimeError("bad frame")
        if value > 150:
            return None
        return _scaled(FLOOR if value < 50 else _pose(240, 0), max(frame.shape[:2]) / 640)

    def infer_batch(self, frames, timestamps_ms):
        self.batches.append(len(frames))
        if self.gate is not None:
            self.entered.set()
            self.gate.wait(5.0)
        if any(int(f[0, 0, 0]) > 225 for f in frames):
            raise RuntimeError("bad frame in the batch")
        return super().infer_batch(frames, timestamps_ms)


class _EngineFactory:
    """create_engine for InferenceService: hands out test engines and remembers them."""
    def __init__(self, **engine_kwargs):
        self.engine_kwargs, self.made, self.names = engine_kwargs, [], []

    def __call__(self, name):
        self.names.append(name)
        if name == "broken":
            raise RuntimeError("cannot load")
        engine = _SceneEngine(**self.engine_kwargs)
        self.made.append(engine)
        return engine


def _started_service(factory, engine_name="main", fallback=None, **kwargs):
    service = InferenceService(factory, engine_name=engine_name, fallback_engine=fallback, **kwargs)
    service.start()
    assert service.wait_ready(5), "the inference service never became ready"
    return service


def test_inference_service_routes_each_frame_to_its_own_session():
    factory = _EngineFactory()
    service = _started_service(factory)
    try:
        a, b = Session("a"), Session("b")
        for i in range(1, int(5.5 * FPS) + 1):
            ra = service.infer(a, _frame(FRAME_FLOOR), i / FPS)         # a lies flat on the floor
            rb = service.infer(b, _frame(FRAME_STANDING), i / FPS)      # b stands
            assert ra.error is None and rb.error is None, (ra, rb)
        assert a.telemetry()["fall"]["alert"]["state"] == "COUNTDOWN"
        assert b.telemetry()["fall"]["alert"]["state"] == "IDLE" and b.telemetry()["fall"]["status"] == "NORMAL"
        assert ra.pose is not None and rb.pose is not None             # the pose comes back too, for drawing
        assert factory.names == ["main"] and len(factory.made) == 1    # one shared engine served both patients
    finally:
        service.stop()


def test_waiting_sessions_are_batched_into_one_engine_call():
    gate = threading.Event()
    factory = _EngineFactory(gate=gate)
    service = _started_service(factory)
    try:
        engine = factory.made[0]
        patients = [Session(f"s{i}") for i in range(4)]
        first = service.submit(patients[0], _frame(FRAME_STANDING), 0.1)
        assert engine.entered.wait(5)                                   # the engine is busy with s0 ...
        rest = [service.submit(s, _frame(FRAME_STANDING), 0.1) for s in patients[1:]]    # ... so these three wait
        gate.set()
        results = [f.result(5) for f in [first] + rest]
        assert engine.batches == [1, 3], engine.batches                 # and then go through together
        assert [r.batch_size for r in results] == [1, 3, 3, 3]
        assert all(r.error is None and not r.dropped for r in results)
        assert all(s.pose_visible for s in patients)                    # every session was fed its own pose
        assert service.stats["batches"] == 2 and service.stats["frames"] == 4
    finally:
        service.stop()


def test_a_batch_is_capped_and_the_rest_wait_their_turn():
    gate = threading.Event()
    factory = _EngineFactory(gate=gate)
    service = _started_service(factory, max_batch=2)
    try:
        engine = factory.made[0]
        patients = [Session(f"s{i}") for i in range(5)]
        first = service.submit(patients[0], _frame(FRAME_STANDING), 0.1)
        assert engine.entered.wait(5)
        rest = [service.submit(s, _frame(FRAME_STANDING), 0.1) for s in patients[1:]]
        gate.set()
        for f in [first] + rest:
            assert f.result(5).error is None
        assert engine.batches == [1, 2, 2], engine.batches              # nobody is starved: all five were served
    finally:
        service.stop()


def test_only_the_newest_waiting_frame_of_a_session_is_inferred():
    gate = threading.Event()
    factory = _EngineFactory(gate=gate)
    service = _started_service(factory)
    try:
        engine = factory.made[0]
        busy = service.submit(Session("busy"), _frame(FRAME_STANDING), 0.1)
        assert engine.entered.wait(5)
        s1 = Session("s1")
        old = service.submit(s1, _frame(FRAME_FLOOR), 0.1)             # waits behind the engine ...
        new = service.submit(s1, _frame(FRAME_STANDING), 0.2)          # ... and is replaced by a newer frame
        gate.set()
        assert old.result(5).dropped is True
        result = new.result(5)
        assert not result.dropped and result.error is None
        busy.result(5)
        assert engine.seen == [FRAME_STANDING, FRAME_STANDING]          # the floor frame never reached the engine
        assert engine.batches == [1, 1] and service.stats["dropped"] == 1
        assert s1.pose_visible and s1.telemetry()["fall"]["status"] == "NORMAL"
    finally:
        service.stop()


def test_a_failing_frame_is_contained_to_its_own_session():
    gate = threading.Event()
    factory = _EngineFactory(gate=gate)
    service = _started_service(factory)
    try:
        engine = factory.made[0]
        busy = service.submit(Session("busy"), _frame(FRAME_STANDING), 0.1)
        assert engine.entered.wait(5)
        a, b = Session("a"), Session("b")
        fa = service.submit(a, _frame(FRAME_BAD), 0.1)
        fb = service.submit(b, _frame(FRAME_STANDING), 0.1)
        gate.set()
        busy.result(5)
        ra, rb = fa.result(5), fb.result(5)
        assert engine.batches == [1, 2]                                 # one batch holding both; it failed as a whole
        assert ra.error and "inference failed" in ra.error and ra.pose is None
        assert rb.error is None and rb.pose is not None and rb.batch_size == 2     # b was retried alone and succeeded
        assert not a.pose_visible and "fall" not in a.telemetry()       # a was NOT fed a made-up "nobody in view"
        assert b.telemetry()["fall"]["status"] == "NORMAL"
    finally:
        service.stop()


def test_a_stateful_engine_gets_one_instance_per_session():
    factory = _EngineFactory(stateless=False)
    service = _started_service(factory)
    try:
        a, b = Session("a"), Session("b")
        service.infer(a, _frame(FRAME_STANDING), 0.1)
        service.infer(b, _frame(FRAME_STANDING), 0.1)
        service.infer(a, _frame(FRAME_STANDING), 0.2)
        assert len(factory.made) == 2                                   # the load-time instance became a's; b got its own
        assert sorted(len(e.seen) for e in factory.made) == [1, 2]      # no frame of one patient reached the other's engine
        owner_of_a = next(e for e in factory.made if len(e.seen) == 2)
        service.forget("a")
        assert _wait_until(lambda: owner_of_a.closed)                   # a's engine is released with the session
        assert not any(e.closed for e in factory.made if e is not owner_of_a)
    finally:
        service.stop()


def test_engine_switch_happens_between_batches_and_a_failed_switch_changes_nothing():
    factory = _EngineFactory()
    service = _started_service(factory)
    try:
        assert service.engine_name == "main"
        service.request_engine("other")
        assert _wait_until(lambda: service.engine_name == "other")
        assert factory.names == ["main", "other"] and factory.made[0].closed and not factory.made[1].closed
        service.request_engine("broken")
        assert _wait_until(lambda: factory.names[-1] == "broken")
        time.sleep(0.2)
        assert service.engine_name == "other" and not factory.made[1].closed     # the failed switch changed nothing
        assert service.infer(Session("a"), _frame(FRAME_STANDING), 0.1).error is None      # and it still serves
    finally:
        service.stop()


def test_the_service_falls_back_when_the_default_engine_will_not_load():
    factory = _EngineFactory()
    service = _started_service(factory, engine_name="broken", fallback="mediapipe")
    try:
        assert service.engine_name == "mediapipe" and factory.names == ["broken", "mediapipe"]
        assert service.infer(Session("a"), _frame(FRAME_STANDING), 0.1).error is None
    finally:
        service.stop()
    nothing = _started_service(_EngineFactory(), engine_name="broken", fallback="broken")
    try:
        s = Session("s")
        r = nothing.infer(s, _frame(FRAME_STANDING), 0.1)
        assert r.error and "engine unavailable" in r.error and r.pose is None
        assert not s.pose_visible and "fall" not in s.telemetry()       # with no engine a session is fed nothing
    finally:
        nothing.stop()


def test_every_resolution_and_orientation_is_judged_at_the_same_reference_scale():
    factory = _EngineFactory()
    service = _started_service(factory)
    shapes = [(480, 640), (960, 1280), (240, 320), (640, 480), (1280, 960)]     # (h, w): landscape 1x 2x 0.5x, portrait 1x 2x
    try:
        patients = [Session(f"s{i}") for i in range(len(shapes))]
        frames = [_frame(FRAME_FLOOR, h, w) for h, w in shapes]
        for i in range(1, int(5.5 * FPS) + 1):
            for s, frame in zip(patients, frames):
                assert service.infer(s, frame, i / FPS).error is None
        reference = patients[0].telemetry()["fall"]
        assert reference["alert"]["state"] == "COUNTDOWN"
        for s, shape in zip(patients, shapes):
            assert s.telemetry()["fall"] == reference, shape             # the same scene gives the same verdict everywhere
    finally:
        service.stop()


def test_a_timed_out_frame_does_not_feed_the_session_a_made_up_empty_pose():
    gate = threading.Event()
    factory = _EngineFactory(gate=gate)
    service = _started_service(factory)
    try:
        engine = factory.made[0]
        busy = service.submit(Session("busy"), _frame(FRAME_STANDING), 0.1)
        assert engine.entered.wait(5)
        a = Session("a")
        assert service.infer(a, _frame(FRAME_STANDING), 0.1, timeout=0.05) is None
        assert not a.pose_visible and "fall" not in a.telemetry()       # nothing was fed
        gate.set()
        busy.result(5)
        assert _wait_until(lambda: a.pose_visible)                      # the frame was only late, and is processed
    finally:
        service.stop()


def test_the_service_applies_pending_commands_even_when_no_frames_arrive():
    service = _started_service(_EngineFactory())
    try:
        s = Session("s")
        service.infer(s, _frame(FRAME_STANDING), 0.1)                   # registers the session with the service
        s.request_exercise("Hip Flexion")
        s.request_baseline_reset()
        assert _wait_until(lambda: s.pipeline.exercise == "Hip Flexion" and not s.pending_baseline_reset)
        service.forget("s")                                             # no longer swept ...
        time.sleep(0.3)
        s.request_exercise("Elbow Curl")
        time.sleep(0.3)
        assert s.pending_exercise == "Elbow Curl"                       # ... so later commands stay pending
    finally:
        service.stop()


def test_the_desktop_worker_feeds_the_local_session_through_the_service():
    from fastapi.testclient import TestClient
    import app.main as main
    assert main.worker.service is main.service and main.worker.session is main.local_session
    assert main.service.engine_name == "yolo"                           # the default is unchanged
    client = TestClient(main.app)
    assert client.post("/api/config", json={"engine": "mediapipe"}).status_code == 200
    try:
        assert main.service._pending_engine == "mediapipe"              # requested; the service thread applies it
        assert client.get("/api/telemetry").json()["engine"] == "yolo"  # and telemetry stays truthful until it has
    finally:
        main.service._pending_engine = None
    try:                                                                 # (the router itself is tested with private apps)
        with client.websocket_connect("/ws/session"):
            pass
    except Exception as e:  # noqa: BLE001
        assert type(e).__name__ in ("WebSocketDisconnect", "WebSocketDenialResponse"), repr(e)
    else:
        assert main._remote_enabled(), "remote sessions must be off unless REMOTE_SESSIONS=1"


def _remote_app(service, sessions, enabled=True, max_remote=4):
    from fastapi import FastAPI
    from app.remote import create_router
    app = FastAPI()
    app.include_router(create_router(sessions, service, enabled=lambda: enabled, max_remote=lambda: max_remote))
    return app


def _jpeg(value, h, w):
    import cv2
    import numpy as np
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def _wire(seq, capture_ms, jpeg):
    return struct.pack(">Id", seq, capture_ms) + jpeg


def test_a_remote_session_streams_frames_and_gets_telemetry_back():
    from fastapi.testclient import TestClient
    factory = _EngineFactory()
    service = _started_service(factory)
    sessions_ = SessionManager()
    try:
        client = TestClient(_remote_app(service, sessions_))
        with client.websocket_connect("/ws/session") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello" and hello["session"].startswith("remote-")
            assert hello["reference_long_side"] == 640 and hello["exercises"] == EXERCISE_NAMES
            sid = hello["session"]
            assert sessions_.get(sid) is not None
            for seq in range(1, 4):                                     # the floor scene, from a small 64 x 48 frame
                ws.send_bytes(_wire(seq, seq * 100.0, _jpeg(FRAME_FLOOR, 48, 64)))
                reply = ws.receive_json()
                assert reply["type"] == "t" and reply["ack"] == seq and "error" not in reply, reply
            assert reply["frame"] == {"w": 64, "h": 48} and reply["pose_visible"] is True
            assert len(reply["pose"]["kp"]["l_hip"]) == 3 and len(reply["pose"]["bbox"]) == 4
            assert reply["fall"]["status"] == "SUSPECTED"               # a 64 px frame was judged at the reference scale
            ws.send_json({"type": "cmd", "name": "exercise", "value": "Knee Flexion"})
            assert ws.receive_json() == {"type": "cmd_result", "name": "exercise", "ok": True}
            ws.send_json({"type": "cmd", "name": "exercise", "value": "Nope"})
            bad = ws.receive_json()
            assert bad["ok"] is False and "exercise must be one of" in bad["error"]
            ws.send_json({"type": "cmd", "name": "launch"})
            assert ws.receive_json()["ok"] is False
            ws.send_json({"type": "cmd", "name": "test_fall"})
            assert ws.receive_json() == {"type": "cmd_result", "name": "test_fall", "ok": True}
            ws.send_bytes(_wire(0, 0.0, b"this is not a jpeg"))
            assert ws.receive_json()["error"] == "not a JPEG image"
            ws.send_bytes(b"abc")
            assert ws.receive_json()["error"] == "frame too short"
            ws.send_bytes(_wire(9, 900.0, _jpeg(FRAME_STANDING, 48, 64)))
            last = ws.receive_json()
            assert last["ack"] == 9 and last["exercise"] == "Knee Flexion"   # the command was applied before this frame
        assert _wait_until(lambda: sessions_.get(sid) is None)          # disconnecting closes the session
    finally:
        service.stop()


def test_remote_sessions_are_refused_when_disabled_or_full():
    from fastapi.testclient import TestClient
    service = _started_service(_EngineFactory())
    try:
        def refused(client):
            try:
                with client.websocket_connect("/ws/session"):
                    pass
            except Exception as e:  # noqa: BLE001
                assert type(e).__name__ in ("WebSocketDisconnect", "WebSocketDenialResponse"), repr(e)
                return True
            return False
        assert refused(TestClient(_remote_app(service, SessionManager(), enabled=False)))
        full = TestClient(_remote_app(service, SessionManager(), max_remote=1))
        with full.websocket_connect("/ws/session") as first:
            assert first.receive_json()["type"] == "hello"
            assert refused(full)                                        # the second one is turned away
    finally:
        service.stop()


# ---- Phase 4: access control for tunnels ------------------------------------------------------------------------

def _refused_with(client, path="/ws/session", **kw):
    """The close code (or None when the handshake itself was refused) of a socket the server turns away."""
    try:
        with client.websocket_connect(path, **kw) as ws:
            ws.receive_json()
    except Exception as e:  # noqa: BLE001
        assert type(e).__name__ in ("WebSocketDisconnect", "WebSocketDenialResponse"), repr(e)
        return getattr(e, "code", None) or 0
    return None


def test_token_origin_and_failure_limiter_basics():
    from app.security import FailureLimiter, is_tunneled, origin_allowed, token_ok
    assert token_ok("s3cret", "s3cret") and not token_ok("s3cret!", "s3cret") and not token_ok("", "s3cret")
    assert origin_allowed(None, ["x"])                                        # no Origin: not a browser
    assert origin_allowed("https://a.trycloudflare.com", ["a.trycloudflare.com"])
    assert origin_allowed("https://a.trycloudflare.com/", ["", "A.trycloudflare.com"])
    assert not origin_allowed("https://evil.example", ["a.trycloudflare.com"])
    assert not origin_allowed("null", ["a.trycloudflare.com"])
    assert origin_allowed("https://evil.example", ["a"], ["https://evil.example/"]) and origin_allowed("https://x.y", [], ["*"])
    assert is_tunneled([(b"host", b"x"), (b"X-Forwarded-For", b"1.2.3.4")]) and not is_tunneled([(b"host", b"x")])
    for local in (b"localhost:5173", b"127.0.0.1:8000", b"192.168.1.20:5173", b"[::1]:8000", b"my-laptop", b"pc.local", b"testserver"):
        assert not is_tunneled([(b"host", local)]), local
    for public in (b"abc.trycloudflare.com", b"x.ngrok-free.app:443", b"8.8.8.8:8000"):
        assert is_tunneled([(b"host", public)]), public                     # no forwarding headers needed
    now = [0.0]
    lim = FailureLimiter(max_fails=2, window_s=60, clock=lambda: now[0])
    lim.record(); assert not lim.blocked()
    lim.record(); assert lim.blocked()
    now[0] = 61.0
    assert not lim.blocked()                                                   # the window passed


def test_remote_gateway_requires_the_access_token_when_one_is_set():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.remote import create_router
    from app.security import FailureLimiter
    service = _started_service(_EngineFactory())
    sessions_ = SessionManager()
    try:
        app = FastAPI()
        app.include_router(create_router(sessions_, service, token=lambda: "s3cret", auth_timeout=0.3,
                                         limiter=FailureLimiter(max_fails=2)))
        client = TestClient(app)
        with client.websocket_connect("/ws/session") as ws:                    # the right token
            ws.send_json({"type": "auth", "token": "s3cret"})
            assert ws.receive_json()["type"] == "hello" and len(sessions_) == 1
            ws.send_json({"type": "auth", "token": "again"})                  # a stray auth message is ignored, not an error
            ws.send_json({"type": "cmd", "name": "reset"})
            assert ws.receive_json() == {"type": "cmd_result", "name": "reset", "ok": True}
        assert _wait_until(lambda: len(sessions_) == 0)
        assert _refused_with(client) == 4401                                   # silent: no token within auth_timeout
        with client.websocket_connect("/ws/session") as ws:                    # wrong token, twice: counts as guessing
            ws.send_json({"type": "auth", "token": "nope"})
            try:
                ws.receive_json()
            except Exception as e:  # noqa: BLE001
                assert getattr(e, "code", None) == 4401
        with client.websocket_connect("/ws/session") as ws:
            ws.send_json({"type": "auth", "token": "nope"})
            try:
                ws.receive_json()
            except Exception as e:  # noqa: BLE001
                assert getattr(e, "code", None) == 4401
        with client.websocket_connect("/ws/session") as ws:                    # now locked out, even with the right token
            ws.send_json({"type": "auth", "token": "s3cret"})
            try:
                ws.receive_json()
            except Exception as e:  # noqa: BLE001
                assert getattr(e, "code", None) == 1013
            else:
                raise AssertionError("a locked-out client must be refused")
        assert len(sessions_) == 0                                             # no session was ever created for a refusal
    finally:
        service.stop()


def test_remote_gateway_checks_the_browser_origin():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.remote import create_router
    service = _started_service(_EngineFactory())
    try:
        def client(allowed=()):
            app = FastAPI()
            app.include_router(create_router(SessionManager(), service, allowed_origins=lambda: list(allowed)))
            return TestClient(app)
        assert _refused_with(client(), headers={"origin": "https://evil.example"}) is not None   # another site
        with client().websocket_connect("/ws/session", headers={"origin": "http://testserver"}) as ws:     # this site
            assert ws.receive_json()["type"] == "hello"
        with client().websocket_connect("/ws/session", headers={"origin": "https://t.trycloudflare.com",    # via a proxy
                                                                "x-forwarded-host": "t.trycloudflare.com"}) as ws:
            assert ws.receive_json()["type"] == "hello"
        with client(["https://other.example"]).websocket_connect("/ws/session", headers={"origin": "https://other.example"}) as ws:
            assert ws.receive_json()["type"] == "hello"                       # listed in REMOTE_ALLOWED_ORIGINS
    finally:
        service.stop()


def test_tunneled_requests_cannot_reach_the_desktop_routes():
    from fastapi import FastAPI, WebSocket
    from fastapi.testclient import TestClient
    from app.security import TunnelGuard
    app = FastAPI()
    app.add_middleware(TunnelGuard)

    @app.get("/api/video_feed")
    def feed():
        return {"camera": True}

    @app.get("/api/exercises")
    def exercises():
        return ["a"]

    @app.websocket("/ws/telemetry")
    async def telemetry(ws: WebSocket):
        await ws.accept()
        await ws.send_json({"ok": True})

    @app.get("/index.html")
    def page():
        return {"page": True}

    client, via = TestClient(app), {"x-forwarded-for": "203.0.113.9"}
    assert client.get("/api/video_feed").status_code == 200                 # directly (localhost / LAN): unchanged
    assert client.get("/api/video_feed", headers=via).status_code == 403    # through a tunnel: refused
    assert client.get("/api/exercises", headers=via).status_code == 200     # the harmless reads stay open
    assert client.get("/index.html", headers=via).status_code == 200        # and so do the app's own files
    assert client.get("/api/video_feed", headers={"cf-connecting-ip": "203.0.113.9"}).status_code == 403
    assert _refused_with(client, "/ws/telemetry", headers=via) is not None
    with client.websocket_connect("/ws/telemetry") as ws:
        assert ws.receive_json() == {"ok": True}


def test_health_tells_a_tunnel_visitor_only_what_the_connect_screen_needs():
    from fastapi.testclient import TestClient
    import app.main as main
    client = TestClient(main.app)
    direct = client.get("/api/health").json()
    assert "camera_index" in direct                                          # the desktop page still gets everything
    assert direct["remote_sessions"] is main._remote_enabled() and direct["remote_auth"] is bool(main._remote_token())
    tunneled = client.get("/api/health", headers={"x-forwarded-for": "203.0.113.9"}).json()
    assert set(tunneled) == {"status", "remote_sessions", "remote_auth"}
    assert client.post("/api/fall/test", headers={"x-forwarded-for": "203.0.113.9"}).status_code == 403
    assert client.get("/api/video_feed", headers={"x-forwarded-for": "203.0.113.9"}).status_code == 403


# ---- browser voice: session commands and the audio message ------------------------------------------------------

def test_session_commands_are_understood_loosely_but_never_by_accident():
    from app.voice.intent import classify_command as c
    for text in ("Start my session.", "start session", "Begin my session", "start the sesion please", "resume session",
                 "Let's go!", "lets go", "okay, let us go", "LET’S GO"):
        assert c(text) == "start_session", text
    for text in ("Stop session", "stop the session please", "End my session.", "stop sessions", "finish session"):
        assert c(text) == "stop_session", text
    for text in ("Show preview", "display preview", "Display the preview.", "open preview", "show pre view", "start preview"):
        assert c(text) == "show_preview", text
    for text in ("Hide preview.", "hide the preview", "close preview", "stop preview"):
        assert c(text) == "hide_preview", text
    for text in ("", "start", "stop", "preview", "session", "I am okay", "show me the weather", "don't stop the session",
                 "stop and start the session", "show and hide preview", "start the preview session", "help",
                 "let it go", "let's not go", "let me know when we can go outside and have a long walk"):
        assert c(text) is None, text


def test_transcribe_names_the_audio_format_it_is_given():
    from app.voice import groq_stt
    seen = {}

    class R:
        status_code = 200
        def json(self): return {"text": "start my session"}

    def post(url, **kw):
        seen.update(kw)
        return R()
    assert groq_stt.transcribe(b"x", "k", post=post, filename="speech.webm", content_type="audio/webm") == "start my session"
    assert seen["files"]["file"] == ("speech.webm", b"x", "audio/webm")
    groq_stt.transcribe(b"x", "k", post=post)                                  # the microphone path is unchanged
    assert seen["files"]["file"][0] == "speech.wav"


def _audio(mime="audio/webm;codecs=opus", payload=b"\x1aE\xdf\xa3" * 300):
    import base64
    return json.dumps({"type": "audio", "mime": mime, "data": base64.b64encode(payload).decode()}, separators=(",", ":"))


def test_remote_voice_runs_session_commands_and_the_fall_answers():
    from fastapi.testclient import TestClient
    from app.voice import groq_stt
    service = _started_service(_EngineFactory())
    sessions_ = SessionManager()
    said, sent = ["start my session"], {}
    real_t, real_k = groq_stt.transcribe, groq_stt.api_key
    groq_stt.api_key = lambda env_file=None: "k"

    def fake(audio, key, post=None, filename="speech.wav", content_type="audio/wav"):
        sent.update(audio=audio, filename=filename, content_type=content_type)
        return said[0]
    groq_stt.transcribe = fake
    try:
        client = TestClient(_remote_app(service, sessions_))
        with client.websocket_connect("/ws/session") as ws:
            ws.receive_json()                                                  # hello
            ws.send_text(_audio("audio/webm;codecs=opus"))
            assert ws.receive_json() == {"type": "voice", "ok": True, "text": "start my session", "intent": "start_session"}
            assert sent["filename"] == "speech.webm" and sent["content_type"] == "audio/webm" and len(sent["audio"]) == 1200
            for words, intent in (("Hide preview", "hide_preview"), ("show the preview", "show_preview"),
                                  ("stop session", "stop_session"), ("what a nice day", None)):
                said[0] = words
                ws.send_text(_audio("audio/mp4"))
                reply = ws.receive_json()
                assert reply["intent"] == intent and sent["filename"] == "speech.m4a", reply
            said[0] = "I am fine"                                              # no fall check open: only commands count
            ws.send_text(_audio())
            assert ws.receive_json()["intent"] is None
            ws.send_json({"type": "cmd", "name": "test_fall"})                 # now a fall check opens
            assert ws.receive_json()["ok"] is True
            for seq in range(1, 60):
                ws.send_bytes(_wire(seq, seq * 100.0, _jpeg(FRAME_STANDING, 48, 64)))
                reply = ws.receive_json()
                if reply["fall"]["alert"]["state"] == "COUNTDOWN":
                    break
            else:
                raise AssertionError("the simulated fall never opened the countdown")
            said[0] = "start my session"                                       # commands are off while the dialog is open
            ws.send_text(_audio())
            assert ws.receive_json()["intent"] is None
            said[0] = "I am fine"
            ws.send_text(_audio())
            assert ws.receive_json()["intent"] == "ok"
            said[0] = "help"
            ws.send_text(_audio())
            assert ws.receive_json()["intent"] == "help"
    finally:
        groq_stt.transcribe, groq_stt.api_key = real_t, real_k
        service.stop()


def test_remote_voice_rejects_bad_audio_and_missing_keys_and_flooding():
    from fastapi.testclient import TestClient
    import app.remote as remote
    import app.voicecmd as voicecmd
    from app.voice import groq_stt
    service = _started_service(_EngineFactory())
    real_t, real_k, real_cap = groq_stt.transcribe, groq_stt.api_key, voicecmd.VOICE_PER_MIN
    calls = []
    groq_stt.transcribe = lambda *a, **k: calls.append(1) or "stop session"
    try:
        client = TestClient(_remote_app(service, SessionManager()))
        with client.websocket_connect("/ws/session") as ws:
            ws.receive_json()
            groq_stt.api_key = lambda env_file=None: None
            ws.send_text(_audio())
            assert ws.receive_json()["status"] == "unavailable"                # no GROQ_API_KEY on the server
            groq_stt.api_key = lambda env_file=None: "k"
            ws.send_text(_audio("video/x-matroska"))
            assert "unsupported audio type" in ws.receive_json()["error"]
            ws.send_text('{"type":"audio","mime":"audio/webm","data":"!!not base64!!"}')
            assert ws.receive_json()["error"] == "bad audio message"
            ws.send_text(_audio(payload=b"x" * (remote.MAX_AUDIO_BYTES + 1)))
            assert ws.receive_json()["error"] == "audio too large"
            assert calls == []                                                 # none of those reached Groq
            voicecmd.VOICE_PER_MIN = 2
            for _ in range(2):
                ws.send_text(_audio())
                assert ws.receive_json()["ok"] is True
            ws.send_text(_audio())
            assert "rate limit" in ws.receive_json()["error"] and len(calls) == 2
            groq_stt.transcribe = lambda *a, **k: (_ for _ in ()).throw(groq_stt.SttError("Groq rate limit reached"))
            voicecmd.VOICE_PER_MIN = 99
            ws.send_text(_audio())
            assert ws.receive_json()["error"] == "Groq rate limit reached"
            ws.send_json({"type": "cmd", "name": "reset"})                     # the socket is still healthy afterwards
            assert ws.receive_json()["ok"] is True
    finally:
        groq_stt.transcribe, groq_stt.api_key, voicecmd.VOICE_PER_MIN = real_t, real_k, real_cap
        service.stop()


def test_standby_pauses_rep_counting_but_never_the_fall_detector():
    assert Session("d").telemetry()["counting"] is True                      # the desktop session counts from the start, as before
    s = Session("s", standby=True)
    calls = []
    real = s.pipeline.reps.update
    s.pipeline.reps.update = lambda angles, t: (calls.append(t), real(angles, t))[1]
    for i in range(10):
        s.process(_pose(240, 0), i / FPS)
    assert calls == [] and s.telemetry()["counting"] is False               # standing in frame, nothing scored
    for i in range(1, int(5.5 * FPS) + 1):                                   # a fall while still in standby ...
        s.process(FLOOR, 1.0 + i / FPS)
    assert s.telemetry()["fall"]["alert"]["state"] == "COUNTDOWN"            # ... is still detected
    s.pending_reset = False
    s.request_start()
    assert s.pipeline.counting is False and s.pending_counting is True       # only a flag until the feeder applies it
    s.apply_pending()
    assert s.pipeline.counting is True and s.pending_counting is None and not s.pending_reset
    s.process(_pose(240, 0), 20.0)
    assert calls                                                              # counting now
    s.request_stop()
    s.apply_pending()
    assert s.pipeline.counting is False and s.telemetry()["counting"] is False


def test_a_phone_session_starts_in_standby_and_the_hud_commands_start_and_pause_it():
    from fastapi.testclient import TestClient
    service = _started_service(_EngineFactory())
    try:
        client = TestClient(_remote_app(service, SessionManager()))
        with client.websocket_connect("/ws/session") as ws:
            ws.receive_json()
            seq = [0]

            def counting():
                seq[0] += 1
                ws.send_bytes(_wire(seq[0], seq[0] * 100.0, _jpeg(FRAME_STANDING, 48, 64)))
                return ws.receive_json()["counting"]
            assert counting() is False                                       # standby until told otherwise
            ws.send_json({"type": "cmd", "name": "start_session"})
            assert ws.receive_json() == {"type": "cmd_result", "name": "start_session", "ok": True}
            assert _wait_until(lambda: counting() is True)
            ws.send_json({"type": "cmd", "name": "stop_session"})
            assert ws.receive_json()["ok"] is True
            assert _wait_until(lambda: counting() is False)
    finally:
        service.stop()


def test_the_desktop_session_buttons_start_and_pause_counting():
    from fastapi.testclient import TestClient
    import app.main as main
    client = TestClient(main.app)
    assert client.post("/api/session/start").json() == {"ok": True}
    assert main.local_session.pending_counting is True and main.local_session.pending_reset is True
    assert client.post("/api/session/stop").json() == {"ok": True}
    assert main.local_session.pending_counting is False
    main.local_session.pending_counting, main.local_session.pending_reset = None, False
    assert client.post("/api/session/start", headers={"x-forwarded-for": "203.0.113.9"}).status_code == 403   # desktop routes
    assert main.local_session.pending_counting is None


def _kp(pose):
    return {n: [round(k.x, 1), round(k.y, 1), round(k.conf, 2)] for n, k in pose.keypoints.items()}


def _kp_text(pose, seq=1, t=100.0, w=640, h=480, kp=None):
    return json.dumps({"type": "kp", "seq": seq, "t": t, "w": w, "h": h, "kp": _kp(pose) if kp is None else kp},
                      separators=(",", ":"))


def test_keypoint_messages_from_a_phone_are_validated():
    from app.remote import KP_PREFIX, parse_keypoints
    text = _kp_text(_pose(240, 0), seq=7, t=1234.5)
    assert text.startswith(KP_PREFIX)                                          # how the gateway tells them from commands
    seq, t, w, h, pose = parse_keypoints(text)
    assert (seq, t, w, h) == (7, 1234.5, 640, 480)
    assert set(pose.keypoints) == set(_pose(240, 0).keypoints) and pose.bbox.w > 0 and pose.keypoints["l_hip"].x == 305.0
    assert parse_keypoints(_kp_text(None, kp={}))[4] is None                   # nobody in view
    two = {"l_hip": [300, 200, 0.9], "r_hip": [330, 200, 0.9]}
    assert parse_keypoints(_kp_text(None, kp=two))[4] is None                  # too few joints to outline a person
    loud = _kp(_pose(240, 0))
    loud["l_hip"][2] = 7
    assert parse_keypoints(_kp_text(None, kp=loud))[4].keypoints["l_hip"].conf == 1.0       # confidence is clamped
    bad = ['{"type":"kp"}', "not json", _kp_text(None, kp={"tail": [1, 2, 0.9]}), _kp_text(None, kp={"l_hip": [1, 2]}),
           _kp_text(None, kp={"l_hip": ["a", 2, 0.5]}), _kp_text(None, w=0, kp={}), _kp_text(None, w=99999, kp={}),
           '{"type":"kp","seq":1,"t":1,"w":640,"h":480,"kp":{"l_hip":[1e999,2,0.5]}}',      # not a finite number
           '{"type":"kp","seq":1,"t":1,"w":640,"h":480,"kp":[1,2]}']
    for text in bad:
        try:
            parse_keypoints(text)
        except ValueError:
            continue
        raise AssertionError(f"accepted: {text[:60]}")


def test_a_pose_computed_on_the_phone_feeds_the_session_even_with_no_engine_loaded():
    service = _started_service(_EngineFactory(), engine_name="broken", fallback="broken")      # nothing loads
    try:
        s = Session("s")
        for i in range(1, int(5.5 * FPS) + 1):                                 # lying on the floor, as the phone's model saw it
            r = service.submit_pose(s, FLOOR, i / FPS, 640, 480).result(5)
            assert r.error is None and not r.dropped and r.infer_ms == 0.0     # no server time spent
        assert s.telemetry()["fall"]["alert"]["state"] == "COUNTDOWN"          # the fall pipeline ran on it
        r = service.submit_pose(Session("n"), None, 0.1, 640, 480).result(5)   # "nobody in view" is a result too
        assert r.pose is None and r.error is None
    finally:
        service.stop()


def test_poses_and_frames_share_a_cycle_but_only_frames_reach_the_engine():
    gate = threading.Event()
    factory = _EngineFactory(gate=gate)
    service = _started_service(factory)
    try:
        engine = factory.made[0]
        busy = service.submit(Session("busy"), _frame(FRAME_STANDING), 0.1)
        assert engine.entered.wait(5)
        s1, s2 = Session("frames"), Session("poses")
        f1 = service.submit(s1, _frame(FRAME_STANDING), 0.2)
        f2 = service.submit_pose(s2, _pose(240, 0), 0.2, 640, 480)
        gate.set()
        busy.result(5)
        r1, r2 = f1.result(5), f2.result(5)
        assert engine.batches == [1, 1], engine.batches                        # the pose never went to the engine
        assert r1.error is None and r2.error is None and r2.batch_size == 2 and r2.infer_ms == 0.0
        assert s1.pose_visible and s2.pose_visible                             # both sessions were fed
    finally:
        service.stop()


def test_a_phone_that_runs_the_model_itself_sends_only_keypoints():
    from fastapi.testclient import TestClient
    service = _started_service(_EngineFactory())
    try:
        client = TestClient(_remote_app(service, SessionManager()))
        with client.websocket_connect("/ws/session") as ws:
            ws.receive_json()
            n = [0]

            def send_pose(pose=None, kp=None):
                n[0] += 1
                ws.send_text(_kp_text(pose or _pose(240, 0), seq=n[0], t=n[0] * 50.0, kp=kp))

            def answer():
                time.sleep(0.12)                                               # answers are limited to ~10 a second
                send_pose()
                return ws.receive_json()
            send_pose()
            first = ws.receive_json()
            assert first["type"] == "t" and "pose" not in first                # the phone already has its own pose
            assert first["frame"] == {"w": 640, "h": 480} and first["infer_ms"] == 0.0 and first["pose_visible"] is True
            assert first["counting"] is False                                   # a phone session starts in standby
            ws.send_json({"type": "cmd", "name": "start_session"})
            assert ws.receive_json() == {"type": "cmd_result", "name": "start_session", "ok": True}
            assert _wait_until(lambda: answer()["counting"] is True)
            time.sleep(0.12)
            send_pose(kp={"tail": [1, 2, 0.9]})                                # a broken message is refused, not fatal
            assert ws.receive_json()["error"].startswith("unknown joint")
            assert answer()["counting"] is True
    finally:
        service.stop()


def test_the_browser_model_route_serves_only_the_two_pose_models():
    from fastapi.testclient import TestClient
    import app.main as main
    client = TestClient(main.app)
    for name in ("yolo11n-pose.pt", "nope.task", "..%2Fapp%2Fmain.py", "pose_landmarker_heavy.task"):
        assert client.get(f"/models/{name}").status_code == 404, name
    path = main.MODELS_DIR / "pose_landmarker_lite.task"
    r = client.get("/models/pose_landmarker_lite.task")
    if path.is_file():
        assert r.status_code == 200 and "max-age" in r.headers["cache-control"] and len(r.content) > 1000
    else:
        assert r.status_code == 404
    via = client.get("/models/pose_landmarker_lite.task", headers={"x-forwarded-for": "203.0.113.9"})
    assert via.status_code == r.status_code                                    # a public file, not a desktop route
    assert client.get("/models/yolo11n-pose-320.onnx").status_code == 404      # the removed client-side YOLO model is not served


def test_yolo_uses_fp16_only_on_a_gpu_and_the_engine_settings_come_from_the_environment():
    from app.engines.yolo_engine import YoloEngine
    assert YoloEngine(device="cuda:0").half is True                          # FP16 by default on the GPU
    assert YoloEngine(device="cuda:0", half=False).half is False
    assert YoloEngine(device="cpu").half is False                            # the CPU has no half precision: never asked for
    assert YoloEngine(device="cpu")._precision_args() == {}
    fp16 = YoloEngine(device="cuda:0")._precision_args()                     # whichever name the installed Ultralytics understands:
    assert fp16 in ({"quantize": 16}, {"half": True})                        # `half` alone would warn on every single frame there
    assert YoloEngine(device="cuda:0", half=False)._precision_args() == {}
    import app.main as main
    saved = {k: os.environ.get(k) for k in ("YOLO_IMGSZ", "YOLO_HALF")}
    try:
        os.environ["YOLO_IMGSZ"], os.environ["YOLO_HALF"] = "640", "1"       # explicit, so a value in backend/.env cannot interfere
        defaults = main._engine_kwargs("yolo")
        assert defaults["device"] == "cuda:0" and defaults["half"] is True and defaults["imgsz"] == 640
        os.environ["YOLO_IMGSZ"], os.environ["YOLO_HALF"] = "480", "off"
        tuned = main._engine_kwargs("yolo")
        assert tuned["imgsz"] == 480 and tuned["half"] is False
        os.environ["YOLO_IMGSZ"] = "500"                                     # not a multiple of 32: ignored
        assert main._engine_kwargs("yolo")["imgsz"] == 640
        os.environ["YOLO_IMGSZ"] = "64"                                      # absurdly small: ignored
        assert main._engine_kwargs("yolo")["imgsz"] == 640
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    assert main._engine_kwargs("mediapipe") == {"variant": "full"}
    assert main._engine_kwargs("mediapipe_lite") == {"variant": "lite"}      # selectable on the badge, never the default


def test_the_lite_mediapipe_model_can_be_chosen_over_the_api_and_the_socket():
    from fastapi.testclient import TestClient
    import app.main as main
    assert list(main.ENGINES) == ["mediapipe", "mediapipe_lite", "yolo"]
    client = TestClient(main.app)
    assert client.post("/api/config", json={"engine": "mediapipe_lite"}).status_code == 200
    try:
        assert main.service._pending_engine == "mediapipe_lite"
    finally:
        main.service._pending_engine = None


def test_a_phone_sees_and_switches_the_pose_model_over_its_socket():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.remote import create_router
    factory = _EngineFactory()
    service = _started_service(factory)                                         # starts on "main"
    try:
        def client(engines):
            app = FastAPI()
            app.include_router(create_router(SessionManager(), service, engine_names=lambda: engines))
            return TestClient(app)
        with client(["main", "other"]).websocket_connect("/ws/session") as ws:
            hello = ws.receive_json()
            assert hello["engine"] == "main" and hello["engines"] == ["main", "other"]
            ws.send_bytes(_wire(1, 100.0, _jpeg(FRAME_STANDING, 48, 64)))
            assert ws.receive_json()["engine"] == "main"                         # every reply names the model in use
            ws.send_json({"type": "cmd", "name": "engine", "value": "nope"})
            bad = ws.receive_json()
            assert bad["ok"] is False and "engine must be one of" in bad["error"]
            ws.send_json({"type": "cmd", "name": "engine", "value": "other"})
            assert ws.receive_json() == {"type": "cmd_result", "name": "engine", "ok": True}
            assert _wait_until(lambda: service.engine_name == "other")
            ws.send_bytes(_wire(2, 200.0, _jpeg(FRAME_STANDING, 48, 64)))
            reply = ws.receive_json()
            assert reply["engine"] == "other" and reply["infer_ms"] >= 0         # and the phone can show the inference time
        with client([]).websocket_connect("/ws/session") as ws:                  # switching not offered: refused
            ws.receive_json()
            ws.send_json({"type": "cmd", "name": "engine", "value": "main"})
            assert ws.receive_json()["ok"] is False
    finally:
        service.stop()


def test_the_desktop_page_gets_the_same_voice_answers_over_http():
    from fastapi.testclient import TestClient
    import app.main as main
    import app.voicecmd as voicecmd
    from app.voice import groq_stt
    said, seen = ["show preview"], {}
    real_t, real_k = groq_stt.transcribe, groq_stt.api_key
    groq_stt.api_key = lambda env_file=None: "k"

    def fake(audio, key, post=None, filename="speech.wav", content_type="audio/wav"):
        seen.update(filename=filename, content_type=content_type, size=len(audio))
        return said[0]
    groq_stt.transcribe = fake
    main._local_voice_calls.clear()
    try:
        client = TestClient(main.app)
        audio = b"\x1aE\xdf\xa3" * 300
        r = client.post("/api/voice", content=audio, headers={"content-type": "audio/webm;codecs=opus"})
        assert r.status_code == 200 and r.json() == {"type": "voice", "ok": True, "text": "show preview", "intent": "show_preview"}
        assert seen == {"filename": "speech.webm", "content_type": "audio/webm", "size": 1200}
        for words, intent in (("close preview", "hide_preview"), ("Let's go", "start_session"), ("stop session", "stop_session"),
                              ("good morning", None)):
            said[0] = words
            if intent == "start_session":
                main.local_session.pending_reset = False
            reply = client.post("/api/voice", content=audio, headers={"content-type": "audio/mp4"}).json()
            assert reply["intent"] == intent, reply
            if intent == "start_session":                                  # the counters reset, exactly as on the phone
                assert main.local_session.pending_reset is True
                main.local_session.pending_reset = False
        assert client.post("/api/voice", content=audio, headers={"content-type": "video/mp4"}).json()["ok"] is False
        r = client.post("/api/voice", content=b"x" * (voicecmd.MAX_AUDIO_BYTES + 1), headers={"content-type": "audio/webm"})
        assert r.status_code == 413
        r = client.post("/api/voice", content=audio, headers={"content-type": "audio/webm", "x-forwarded-for": "203.0.113.9"})
        assert r.status_code == 403                                        # a desktop route: not through a tunnel
    finally:
        groq_stt.transcribe, groq_stt.api_key = real_t, real_k
        main._local_voice_calls.clear()


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
