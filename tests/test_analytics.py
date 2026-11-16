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
