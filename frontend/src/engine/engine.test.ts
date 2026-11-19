import { describe, expect, it } from "vitest";
import { analyticsLocal, defaultEngineMode, type Keypoints } from "../remote/logic";
import { Escalation } from "./escalation";
import { EXERCISES, EXERCISE_NAMES } from "./exercises";
import { AngleEma, ANGLE_KEYS, PoseFrame, jointAnglesInto, N_ANGLES } from "./kinematics";
import { LocalEngine } from "./localEngine";
import { LimbFSM, calibratedLimits } from "./repCounter";

/** Unit tests that need no golden files: the pieces' behaviour as the Python tests describe it. */

const standing = (elbow = 165): Keypoints => {
  const kp: Keypoints = {
    nose: [320, 130, 0.9], l_shoulder: [290, 160, 0.9], r_shoulder: [350, 160, 0.9], l_hip: [300, 260, 0.9], r_hip: [340, 260, 0.9],
    l_knee: [300, 350, 0.9], r_knee: [340, 350, 0.9], l_ankle: [300, 440, 0.9], r_ankle: [340, 440, 0.9],
    l_elbow: [290, 220, 0.9], r_elbow: [350, 220, 0.9],
  };
  const r = (elbow * Math.PI) / 180;                                 // wrist: `elbow` degrees from the line to the shoulder (up)
  kp.l_wrist = [290 - Math.sin(r) * 55, 220 - Math.cos(r) * 55, 0.9];
  kp.r_wrist = [350 + Math.sin(r) * 55, 220 - Math.cos(r) * 55, 0.9];
  return kp;
};

describe("engine switch", () => {
  it("defaults to local in the bundled APK only, and needs the pose on this device too", () => {
    expect(defaultEngineMode(true)).toBe("local");
    expect(defaultEngineMode(false)).toBe("server");
    expect(analyticsLocal("local", "device")).toBe(true);
    expect(analyticsLocal("local", "server")).toBe(false);          // frames go to the server: it must do the analytics
    expect(analyticsLocal("server", "device")).toBe(false);
  });
});

describe("exercise table", () => {
  it("is the one shared JSON, in the table's order", () => {
    expect(EXERCISE_NAMES[0]).toBe("Elbow Curl");
    expect(EXERCISES["Seated Knee Extension"]).toEqual({ joints: { L: "l_knee", R: "r_knee" }, rest: 95, peak: 165 });
    expect(EXERCISE_NAMES.length).toBe(10);
  });
});

describe("joint angles", () => {
  it("measures a right angle at the elbow in 2D and prefers 3D when every joint has it", () => {
    const f = new PoseFrame(), vals = new Float64Array(N_ANGLES), has = new Uint8Array(N_ANGLES);
    f.load(standing(90), undefined, 1);
    jointAnglesInto(f, true, vals, has);
    const e = ANGLE_KEYS.indexOf("l_elbow");
    expect(has[e]).toBe(1);
    expect(vals[e]).toBeCloseTo(90, 5);
    const world = { l_shoulder: [0, 0, 0], l_elbow: [0, 0.3, 0], l_wrist: [0.3, 0.3, 0] } as Record<string, [number, number, number]>;
    f.load(standing(150), world, 1);
    jointAnglesInto(f, true, vals, has);
    expect(vals[e]).toBeCloseTo(90, 5);                                // the world joints say 90, the image says 150
    jointAnglesInto(f, false, vals, has);
    expect(vals[e]).toBeCloseTo(150, 5);                               // ANGLES_3D=0 ignores them
  });
  it("leaves out an angle when a joint is below confidence 0.5 or missing, and the mean uses what exists", () => {
    const kp = standing(90);
    kp.r_wrist[2] = 0.3;
    delete kp.l_ankle;
    const f = new PoseFrame(), vals = new Float64Array(N_ANGLES), has = new Uint8Array(N_ANGLES);
    f.load(kp, undefined, 1);
    jointAnglesInto(f, true, vals, has);
    expect(has[ANGLE_KEYS.indexOf("r_elbow")]).toBe(0);
    expect(has[ANGLE_KEYS.indexOf("l_knee")]).toBe(0);
    expect(has[ANGLE_KEYS.indexOf("elbow_mean")]).toBe(1);
    expect(vals[ANGLE_KEYS.indexOf("elbow_mean")]).toBeCloseTo(vals[ANGLE_KEYS.indexOf("l_elbow")], 9);
  });
});

describe("smoothing", () => {
  it("follows a step 63 % after tau seconds at any frame rate (time based)", () => {
    for (const fps of [10, 30]) {
      const ema = new AngleEma(0.35, 0.08), v = new Float64Array(N_ANGLES), h = new Uint8Array(N_ANGLES);
      h[0] = 1; v[0] = 0; ema.update(v, h, 0);
      let out = 0;
      for (let i = 1; i <= Math.round(0.08 * fps); i++) { v[0] = 100; ema.update(v, h, i / fps); out = v[0]; }
      expect(out).toBeGreaterThan(55);
      expect(out).toBeLessThan(80);
    }
  });
});

describe("rep counting and calibration", () => {
  it("counts one rep through IDLE -> FLEXING -> PEAK -> RETURNING -> IDLE, with the range of motion", () => {
    const l = new LimbFSM(160, 50);
    const angles = [160, 140, 110, 70, 52, 60, 100, 150, 160];
    const done = angles.map((a, i) => l.update(a, i * 0.3));
    expect(done.filter(Boolean).length).toBe(1);
    expect(l.reps).toBe(1);
    expect(l.lastRom).toBeCloseTo(108, 5);
    expect(l.state).toBe("IDLE");
  });
  it("ignores a rep that is too fast or never leaves the start", () => {
    const l = new LimbFSM(160, 50);
    [160, 100, 52, 60, 150, 160].forEach((a, i) => l.update(a, i * 0.02));       // 0.1 s: below the 0.4 s minimum
    expect(l.reps).toBe(0);
  });
  it("calibrates from the person's own rest, keeps the table when there is no room", () => {
    expect(calibratedLimits(170, 100, 95)).toEqual([95, 25]);
    expect(calibratedLimits(100, 165, 150)).toEqual([150, 178]);
    expect(calibratedLimits(100, 165, 170)).toBeNull();
  });

  it("measures the rest angle during standby and starts counting against it; a plain standing posture keeps the table", () => {
    const seated = (knee: number): Keypoints => {
      const r = (knee * Math.PI) / 180;
      const side = (x: number): Keypoints => ({
        [`hip${x}`]: [x, 300, 0.9], [`knee${x}`]: [x + 90, 300, 0.9], [`ankle${x}`]: [x + 90 - Math.cos(r) * 90, 300 + Math.sin(r) * 90, 0.9],
      });
      const l = side(300), r2 = side(340);
      return {
        l_shoulder: [290, 200, 0.9], r_shoulder: [350, 200, 0.9], nose: [320, 170, 0.9],
        l_hip: l.hip300, l_knee: l.knee300, l_ankle: l.ankle300, r_hip: r2.hip340, r_knee: r2.knee340, r_ankle: r2.ankle340,
      };
    };
    const eng = new LocalEngine("Knee Flexion");
    for (let i = 0; i < 60; i++) eng.process(seated(95), undefined, 640, 480, i / 30);
    eng.start();
    const limb = eng.pipeline.reps.limbs[0];
    expect(limb.calibrated).toBe(true);
    expect(limb.rest).toBeCloseTo(95, 0);
    const stand = new LocalEngine("Knee Flexion");
    for (let i = 0; i < 60; i++) stand.process(standing(), undefined, 640, 480, i / 30);
    stand.start();
    expect(stand.pipeline.reps.limbs[0].calibrated).toBe(false);
    expect(stand.pipeline.reps.limbs[0].rest).toBe(170);
  });
});

describe("escalation", () => {
  it("a simulated fall counts down 10 s to ALERTED; an answer ends it", () => {
    const e = new Escalation();
    e.simulate();
    e.update("NORMAL", false, 0);
    expect(e.state).toBe("COUNTDOWN");
    e.update("NORMAL", false, 9.9);
    expect(e.state).toBe("COUNTDOWN");
    e.update("NORMAL", false, 10.0);
    expect(e.state).toBe("ALERTED");
    expect(e.fired).toBe(true);
    e.ack();
    e.update("NORMAL", false, 11);
    expect(e.state).toBe("IDLE");
  });
  it("an acknowledgement cannot pre-answer a later fall (a flag lives one frame)", () => {
    const e = new Escalation();
    e.ack();
    e.update("NORMAL", false, 0);
    e.update("CONFIRMED", true, 1);
    expect(e.state).toBe("COUNTDOWN");
  });
});

describe("the engine as a session", () => {
  it("starts in standby, counts only after start, pauses and resumes without a reset, stops back to standby", () => {
    const eng = new LocalEngine("Elbow Curl");
    expect(eng.telemetry().counting).toBe(false);
    eng.start();
    expect(eng.telemetry().counting).toBe(true);
    const v = eng.version;
    expect(eng.pause()).toBe(true);
    expect(eng.paused).toBe(true);
    expect(eng.telemetry().counting).toBe(false);
    expect(eng.version).toBeGreaterThan(v);
    eng.resume();
    expect(eng.paused).toBe(false);
    expect(eng.telemetry().counting).toBe(true);
    eng.stop();
    expect(eng.pause()).toBe(false);                                    // nothing running to pause
  });
  it("ignores an unknown exercise and reports the telemetry the UI reads", () => {
    const eng = new LocalEngine("Not an exercise");
    expect(eng.telemetry().exercise).toBe("Elbow Curl");
    eng.setExercise("Mini Squat");
    const t = eng.telemetry();
    expect(t.exercise).toBe("Mini Squat");
    expect(Object.keys(t.reps)).toEqual(["B"]);
    expect(t.fall.alert.state).toBe("IDLE");
  });
  it("is fast: well under a millisecond per frame (the budget is 0.1 ms; the bound is loose so a busy machine does not flake)", () => {
    const eng = new LocalEngine("Elbow Curl");
    eng.start();
    const frames = [standing(160), standing(90)];
    for (let i = 0; i < 300; i++) eng.process(frames[i % 2], undefined, 640, 480, i / 30);     // warm up
    const n = 3000, t0 = performance.now();
    for (let i = 0; i < n; i++) eng.process(frames[i % 2], undefined, 640, 480, 10 + i / 30);
    const per = (performance.now() - t0) / n;
    console.info(`engine: ${(per * 1000).toFixed(1)} us per frame`);
    expect(per).toBeLessThan(0.5);
  });
});
