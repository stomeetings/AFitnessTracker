"""Golden parity traces for the on-device TypeScript engine (frontend/src/engine/).

Builds synthetic keypoint traces (a standing person curling, a seated person with a calibrated baseline, a fall with the alert
countdown, the Test Fall button, a lost pose, walking, an exercise switch), replays each one through the REAL server path
(parse_keypoints -> Session.apply_pending -> Session.process, exactly what /ws/session does for a phone) and saves the inputs and the
Python results frame by frame to tests/golden/*.json. frontend/src/engine/golden.test.ts runs the TypeScript engine on the same
frames and must reproduce reps, rep states, fall status and alert transitions (within one frame).

    backend/.venv/Scripts/python scripts/make_golden.py

Run it again whenever backend/app/pipeline changes, commit the result, and make the TypeScript match. Deterministic: no clock, a
fixed random seed. A recorded phone trace can be added the same way: a JSON file in tests/golden/ with the same "frames" list (and
no "expected": this script fills it in), listed in EXTRA below.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.remote import parse_keypoints  # noqa: E402
from app.sessions import Session  # noqa: E402

OUT = ROOT / "tests" / "golden"
EXTRA = ROOT / "tests" / "golden_input"          # optional recorded traces: *.json with "name", "exercise", "frames"
JOINTS = ["nose", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
          "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
CONF = 0.9


# ---- synthetic bodies, in a 640 x 480 picture (x right, y DOWN, like an image) ---------------------------------------------------

def _end(origin, length, phi, side):
    """The point `length` from `origin` whose angle to straight UP is phi degrees (phi 180 = straight down), to the left (-1) or right (+1)."""
    r = math.radians(phi)
    return (origin[0] + side * math.sin(r) * length, origin[1] - math.cos(r) * length)


def standing(cx=320.0, elbow=(165.0, 165.0), knee=(175.0, 175.0), shoulder=(15.0, 15.0), sep=0.0):
    """A standing person: torso 100 px, thigh and shin 90, upper arm 60, forearm 55. `elbow`/`knee` are the angles at those joints
    (180 = straight), `shoulder` the angle between the arm and the body (0 = hanging), `sep` pulls the ankles apart (walking)."""
    p = {"nose": (cx, 130.0), "l_shoulder": (cx - 30, 160.0), "r_shoulder": (cx + 30, 160.0),
         "l_hip": (cx - 20, 260.0), "r_hip": (cx + 20, 260.0)}
    for i, side in enumerate((-1, 1)):
        n = "lr"[i]
        sh = p[f"{n}_shoulder"]
        th = math.radians(shoulder[i])
        p[f"{n}_elbow"] = (sh[0] + side * math.sin(th) * 60, sh[1] + math.cos(th) * 60)      # the arm swings out from hanging
        # forearm: elbow angle (shoulder-elbow-wrist) measured at the elbow, in the arm's own direction
        ux, uy = sh[0] - p[f"{n}_elbow"][0], sh[1] - p[f"{n}_elbow"][1]
        base = math.atan2(uy, ux)                                                           # direction elbow -> shoulder
        a = base + side * math.radians(elbow[i]) * -1
        p[f"{n}_wrist"] = (p[f"{n}_elbow"][0] + math.cos(a) * 55, p[f"{n}_elbow"][1] + math.sin(a) * 55)
        hip = p[f"{n}_hip"]
        p[f"{n}_knee"] = (hip[0], hip[1] + 90)
        p[f"{n}_ankle"] = _end(p[f"{n}_knee"], 90, knee[i], side)
        p[f"{n}_ankle"] = (p[f"{n}_ankle"][0] + side * sep, p[f"{n}_ankle"][1])
    return p


def seated(cx=320.0, knee=(95.0, 95.0)):
    """Sitting on a bed edge: thighs forward (horizontal), shins hanging, `knee` = the angle at the knees."""
    p = {"nose": (cx, 170.0), "l_shoulder": (cx - 30, 200.0), "r_shoulder": (cx + 30, 200.0),
         "l_hip": (cx - 20, 300.0), "r_hip": (cx + 20, 300.0),
         "l_elbow": (cx - 30, 260.0), "r_elbow": (cx + 30, 260.0), "l_wrist": (cx - 30, 315.0), "r_wrist": (cx + 30, 315.0)}
    for i in range(2):
        n = "lr"[i]
        hip = p[f"{n}_hip"]
        p[f"{n}_knee"] = (hip[0] + 90, hip[1])
        r = math.radians(knee[i])
        p[f"{n}_ankle"] = (p[f"{n}_knee"][0] - math.cos(r) * 90, p[f"{n}_knee"][1] + math.sin(r) * 90)
    return p


def lying():
    """Flat on the floor, head to the left."""
    return {"nose": (160.0, 410.0), "l_shoulder": (200.0, 395.0), "r_shoulder": (200.0, 425.0),
            "l_elbow": (250.0, 392.0), "r_elbow": (250.0, 428.0), "l_wrist": (300.0, 392.0), "r_wrist": (300.0, 428.0),
            "l_hip": (320.0, 400.0), "r_hip": (320.0, 420.0), "l_knee": (410.0, 402.0), "r_knee": (410.0, 418.0),
            "l_ankle": (500.0, 402.0), "r_ankle": (500.0, 418.0)}


def lerp(a, b, u):
    return {k: (a[k][0] + (b[k][0] - a[k][0]) * u, a[k][1] + (b[k][1] - a[k][1]) * u) for k in a}


def frame(i, fps, pose, rng, scale=1.0, size=(640, 480), world=False, conf=None, cmd=()):
    """One kp message's content: joints rounded like the phone does (0.1 px), a little deterministic jitter, optional world joints."""
    kp = {}
    for j, (x, y) in pose.items():
        kp[j] = [round((x + rng.uniform(-0.6, 0.6)) * scale, 1), round((y + rng.uniform(-0.6, 0.6)) * scale, 1), (conf or {}).get(j, CONF)]
    out = {"t": round(i * 1000.0 / fps, 3), "w": int(size[0] * scale), "h": int(size[1] * scale), "kp": kp}
    if world and kp:
        out["world"] = {j: [round((x - 320) * 0.002, 3), round((y - 260) * 0.002, 3), round(0.05 * math.sin(0.3 * i + n), 3)]
                        for n, (j, (x, y)) in enumerate(pose.items())}
    if cmd:
        out["cmd"] = list(cmd)
    return out


def trace(name, exercise, fps, seconds, pose_at, cmds, **kw):
    """`pose_at(t)` -> joints (or None = nobody); `cmds` = {frame index: [commands]}."""
    rng = random.Random(7)
    frames = []
    for i in range(int(seconds * fps)):
        t = i / fps
        pose = pose_at(t)
        if pose is None:
            frames.append({"t": round(i * 1000.0 / fps, 3), "w": int(640 * kw.get("scale", 1.0)), "h": int(480 * kw.get("scale", 1.0)),
                           "kp": {}, **({"cmd": cmds[i]} if i in cmds else {})})
        else:
            frames.append(frame(i, fps, pose, rng, cmd=cmds.get(i, ()), **kw))
    return {"name": name, "exercise": exercise, "fps": fps, "frames": frames}


def at(fps, seconds):
    """Frame index of a time in seconds (for the command tables)."""
    return int(round(seconds * fps))


def cycle(t, t0, period, lo, hi):
    """A smooth repetition from `lo` (rest) to `hi` (peak) and back, `period` seconds long, starting at t0; `lo` before t0."""
    if t < t0:
        return lo
    return lo + (hi - lo) * (0.5 - 0.5 * math.cos(2 * math.pi * (t - t0) / period))


def build():
    out = []
    # 1. standing elbow curls, both arms, 3D joints present, a 320 x 240 picture (the engine rescales to the 640 reference)
    out.append(trace("elbow_curl_3d", "Elbow Curl", 30, 14,
                     lambda t: standing(elbow=(cycle(t, 2.0, 2.0, 160, 55) if t < 12 else 160, cycle(t, 2.0, 2.0, 160, 62) if t < 12 else 160)),
                     {at(30, 2.0): ["start"]}, scale=0.5, world=True))
    # 2. seated knee flexion: the rest angle (95) is measured in standby; a pause and a resume in the middle
    out.append(trace("knee_seated_calibrated", "Knee Flexion", 15, 24,
                     lambda t: seated(knee=(cycle(t, 3.0, 3.0, 95, 30) if t < 15 or t > 18 else 95,
                                            cycle(t, 3.0, 3.0, 95, 30) if t < 15 or t > 18 else 95)),
                     {at(15, 3.0): ["start"], at(15, 15.5): ["pause"], at(15, 18.0): ["resume"]}))
    # 3. a fall: standing, down in 0.4 s, lying through the dwell and the countdown, "I'm OK", then getting up
    def fall_pose(t):
        if t < 3.0:
            return standing()
        if t < 3.4:
            return lerp(standing(), lying(), (t - 3.0) / 0.4)
        if t < 20.0:
            return lying()
        if t < 21.0:
            return lerp(lying(), standing(), t - 20.0)
        return standing()
    out.append(trace("fall_and_alert", "Elbow Curl", 30, 25, fall_pose, {at(30, 19.5): ["ack"]}))      # confirmed ~8.4 s, alert ~18.4 s, then "I'm OK" while still down
    # 4. Test Fall + "I'm OK", a lost pose, Test Fall + emergency, then "I'm OK" while alerted
    out.append(trace("test_fall_and_lost_pose", "Elbow Curl", 30, 16, lambda t: None if 6.0 <= t < 7.0 else standing(),
                     {at(30, 1.0): ["simulate"], at(30, 4.0): ["ack"], at(30, 8.0): ["simulate"], at(30, 9.0): ["escalate"], at(30, 11.0): ["ack"]}))
    # 5. walking: ankles swing apart and together, the hips travel; counting is on but nothing is being exercised
    out.append(trace("walking_steps", "Knee Flexion", 30, 10,
                     lambda t: standing(cx=100 + 45 * t),                                # the hips travel 45 px/s
                     {at(30, 0.5): ["start"]}))
    walking = out[-1]
    for i, fr in enumerate(walking["frames"]):                                           # swing the ankles in the picture
        s = 55 * math.sin(2 * math.pi * 1.5 * i / 30)
        fr["kp"]["l_ankle"][0] = round(fr["kp"]["l_ankle"][0] - s, 1)
        fr["kp"]["r_ankle"][0] = round(fr["kp"]["r_ankle"][0] + s, 1)
    # 6. shoulder raises, then another exercise (a squat measured by the mean knee angle) and a reset in between
    def mixed(t):
        if t < 12:
            th = cycle(t, 1.0, 2.0, 12, 100) if t < 9 else 12
            return standing(shoulder=(th, th))
        k = cycle(t, 13.0, 2.0, 175, 100) if t < 19 else 175
        return standing(knee=(k, k))
    out.append(trace("exercise_switch", "Shoulder Raise", 30, 21, mixed,
                     {at(30, 0.5): ["start"], at(30, 9.0): ["reset"], at(30, 12.0): ["exercise:Mini Squat"], at(30, 12.5): ["start"]}))
    for p in sorted(EXTRA.glob("*.json")) if EXTRA.is_dir() else ():
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


# ---- the server path -------------------------------------------------------------------------------------------------------------

def apply(s: Session, cmd: str) -> None:
    if cmd == "start":
        s.request_start()
    elif cmd == "stop":
        s.request_stop()
    elif cmd == "pause":
        s.request_pause()
    elif cmd == "resume":
        s.request_resume()
    elif cmd == "ack":
        s.ack()
    elif cmd == "escalate":
        s.escalate()
    elif cmd == "simulate":
        s.simulate_fall()
    elif cmd == "reset":
        s.request_reset()
    elif cmd.startswith("exercise:"):
        s.request_exercise(cmd.split(":", 1)[1])
    else:
        raise ValueError(f"unknown command {cmd!r}")


def record(tel: dict) -> dict:
    fall = tel["fall"]
    return {
        "counting": tel["counting"], "visible": tel["pose_visible"], "posture": tel["posture"], "exercise": tel["exercise"],
        "reps": {side: [l["reps"], l["state"], l["calibrated"], l["rest"], l["peak"]] for side, l in tel["reps"].items()},
        "angles": tel["angles"],
        "fall": [fall["status"], fall["on_floor"], fall["impact"]],
        "alert": [fall["alert"]["state"], fall["alert"]["rechecks"]],
        "steps": tel["mobility"]["steps"], "mobility": tel["mobility"]["state"],
    }


DEFAULT_SWITCHES = {"ANGLES_3D": "1", "CALIBRATE": "1", "SMOOTH_TAU_S": "0.08"}     # a fresh server's defaults, whatever backend/.env says


def replay(tr: dict) -> list[dict]:
    saved = {k: os.environ.get(k) for k in DEFAULT_SWITCHES}
    os.environ.update(DEFAULT_SWITCHES)                           # read when the Session (its Pipeline) is created
    try:
        s = Session("golden", tr["exercise"], standby=True)       # a phone session starts in standby
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    expected = []
    for i, fr in enumerate(tr["frames"]):
        for cmd in fr.get("cmd", ()):
            apply(s, cmd)
        msg = {"type": "kp", "seq": i, "t": fr["t"], "w": fr["w"], "h": fr["h"], "kp": fr["kp"]}
        if fr.get("world"):
            msg["world"] = fr["world"]
        _, t_ms, w, h, pose = parse_keypoints(json.dumps(msg))   # exactly what /ws/session does with the phone's message
        s.apply_pending()
        s.process(pose, t_ms / 1000.0, 0.0, w, h)
        expected.append(record(s.telemetry()))
    return expected


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for tr in build():
        tr["expected"] = replay(tr)
        tr["options"] = {"angles3d": True, "calibrate": True, "tauS": 0.08}
        path = OUT / f"{tr['name']}.json"
        path.write_text(json.dumps(tr, separators=(",", ":")), encoding="utf-8")
        last = tr["expected"][-1]
        print(f"{path.relative_to(ROOT)}: {len(tr['frames'])} frames, reps {last['reps']}, fall {last['fall'][0]}, alert {last['alert'][0]}, steps {last['steps']}")


if __name__ == "__main__":
    main()
