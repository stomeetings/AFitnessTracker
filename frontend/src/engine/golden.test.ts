import { describe, expect, it } from "vitest";
import type { Keypoints, World } from "../remote/logic";
import { LocalEngine } from "./localEngine";

/** Parity with the server. tests/golden/*.json is written by scripts/make_golden.py: synthetic keypoint traces replayed through the
 *  REAL server path (parse_keypoints -> Session -> Pipeline) together with what Python answered on every frame. The TypeScript
 *  engine gets the same frames and the same commands and must answer the same: reps per limb, the rep state machine, fall status,
 *  alert state, steps (state changes at the same frame, or one frame apart), and the angles to rounding. */
type Frame = { t: number; w: number; h: number; kp: Keypoints; world?: World; cmd?: string[] };
type Expected = {
  counting: boolean; visible: boolean; posture: string; exercise: string;
  reps: Record<string, [number, string, boolean, number, number]>;      // reps, state, calibrated, rest, peak
  angles: Record<string, number>; fall: [string, boolean, boolean]; alert: [string, number]; steps: number; mobility: string;
};
type Trace = { name: string; exercise: string; options: { angles3d: boolean; calibrate: boolean; tauS: number }; frames: Frame[]; expected: Expected[] };

const files = import.meta.glob("../../../tests/golden/*.json", { eager: true, import: "default" }) as Record<string, Trace>;
const traces = Object.values(files);

function run(tr: Trace) {
  const eng = new LocalEngine(tr.exercise, tr.options);
  return tr.frames.map((fr) => {
    for (const c of fr.cmd ?? []) {
      if (c === "start") eng.start(); else if (c === "stop") eng.stop(); else if (c === "pause") eng.pause(); else if (c === "resume") eng.resume();
      else if (c === "ack") eng.ack(); else if (c === "escalate") eng.escalate(); else if (c === "simulate") eng.simulateFall();
      else if (c === "reset") eng.reset(); else if (c.startsWith("exercise:")) eng.setExercise(c.slice(9));
      else throw new Error(`unknown command ${c}`);
    }
    eng.process(Object.keys(fr.kp).length ? fr.kp : null, fr.world, fr.w, fr.h, fr.t / 1000);
    return eng.telemetry();
  });
}

/** The value changes of a per-frame signal: [frame, value]. */
const changes = (seq: unknown[]) => seq.reduce<Array<[number, unknown]>>((out, v, i) => (i === 0 || v !== seq[i - 1] ? (out.push([i, v]), out) : out), []);

/** Same sequence of values, each change at the same frame or one frame apart. */
function sameChanges(name: string, got: unknown[], want: unknown[]) {
  const g = changes(got), w = changes(want);
  expect(g.map((c) => c[1]), `${name}: values`).toEqual(w.map((c) => c[1]));
  g.forEach((c, i) => expect(Math.abs(c[0] - w[i][0]), `${name}: change ${i} (${String(c[1])}) at frame ${c[0]} vs ${w[i][0]}`).toBeLessThanOrEqual(1));
}

describe("golden parity with the Python pipeline", () => {
  it("has golden traces (run: backend/.venv/Scripts/python scripts/make_golden.py)", () => {
    expect(traces.length).toBeGreaterThanOrEqual(6);
  });

  for (const tr of traces) {
    it(`${tr.name}: ${tr.frames.length} frames`, () => {
      const got = run(tr), want = tr.expected;
      expect(got.length).toBe(want.length);
      const sides = [...new Set(want.flatMap((w) => Object.keys(w.reps)))];      // an exercise switch changes the sides (L, R -> B)
      for (const side of sides) {
        sameChanges(`${side} reps`, got.map((g) => g.reps[side]?.reps), want.map((w) => w.reps[side]?.[0]));
        sameChanges(`${side} state`, got.map((g) => g.reps[side]?.state), want.map((w) => w.reps[side]?.[1]));
        sameChanges(`${side} calibrated`, got.map((g) => g.reps[side]?.calibrated), want.map((w) => w.reps[side]?.[2]));
      }
      sameChanges("fall status", got.map((g) => g.fall.status), want.map((w) => w.fall[0]));
      sameChanges("on floor", got.map((g) => g.fall.on_floor), want.map((w) => w.fall[1]));
      sameChanges("alert state", got.map((g) => g.fall.alert.state), want.map((w) => w.alert[0]));
      sameChanges("rechecks", got.map((g) => g.fall.alert.rechecks), want.map((w) => w.alert[1]));
      sameChanges("counting", got.map((g) => g.counting), want.map((w) => w.counting));
      sameChanges("exercise", got.map((g) => g.exercise), want.map((w) => w.exercise));
      sameChanges("steps", got.map((g) => g.mobility.steps), want.map((w) => w.steps));
      sameChanges("pose visible", got.map((g) => g.pose_visible), want.map((w) => w.visible));

      // the angles: the same joints present on every frame, equal to the rounding (both sides round to 0.1)
      let worst = 0;
      got.forEach((g, i) => {
        expect(Object.keys(g.angles).sort(), `angles present at frame ${i}`).toEqual(Object.keys(want[i].angles).sort());
        for (const k in g.angles) worst = Math.max(worst, Math.abs(g.angles[k] - want[i].angles[k]));
      });
      expect(worst, "largest angle difference (degrees)").toBeLessThanOrEqual(0.2);
      // the rest/peak in use, wherever the same limbs exist (calibration measured the same baseline)
      const last = got.length - 1;
      for (const side of Object.keys(want[last].reps)) {
        expect(Math.abs(got[last].reps[side].rest - want[last].reps[side][3])).toBeLessThanOrEqual(0.2);
        expect(Math.abs(got[last].reps[side].peak - want[last].reps[side][4])).toBeLessThanOrEqual(0.2);
      }
    });
  }

  // The traces must actually exercise what they claim to, or a matching "nothing happened" would pass.
  const named = (n: string) => traces.find((t) => t.name === n);
  it("the elbow curl trace counts reps on both arms", () => {
    const e = named("elbow_curl_3d")?.expected.at(-1);
    expect(e?.reps.L[0]).toBeGreaterThanOrEqual(4);
    expect(e?.reps.R[0]).toBeGreaterThanOrEqual(4);
  });
  it("the seated trace calibrates its rest angle, before and after the pause", () => {
    const t = named("knee_seated_calibrated");
    expect(t?.expected.some((e) => e.reps.L[2] && e.reps.R[2])).toBe(true);
    expect(t?.expected.at(-1)?.reps.L[0]).toBeGreaterThanOrEqual(3);
  });
  it("the fall trace reaches CONFIRMED and ALERTED and ends recovered", () => {
    const e = named("fall_and_alert")?.expected ?? [];
    expect(e.some((x) => x.fall[0] === "CONFIRMED")).toBe(true);
    expect(e.some((x) => x.alert[0] === "ALERTED")).toBe(true);
    expect(e.at(-1)?.fall[0]).toBe("NORMAL");
  });
  it("the test-fall trace goes through COUNTDOWN and ALERTED", () => {
    const e = named("test_fall_and_lost_pose")?.expected ?? [];
    expect(e.some((x) => x.alert[0] === "COUNTDOWN") && e.some((x) => x.alert[0] === "ALERTED")).toBe(true);
  });
  it("the walking trace counts steps", () => {
    expect(named("walking_steps")?.expected.at(-1)?.steps).toBeGreaterThanOrEqual(3);
  });
});
