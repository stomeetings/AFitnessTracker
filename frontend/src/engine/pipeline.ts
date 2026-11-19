/** One shared analytics core: pose in, reps / falls / escalation out, time based. Port of backend/app/pipeline/core.py. The caller
 *  (LocalEngine) passes the pose exactly as the phone sends it to the server (joints in pixels of the w x h frame, optional world
 *  joints); the server's own steps are reproduced here: a pose needs 4 confident joints to count as a person (bbox_from_keypoints),
 *  and pixels are rescaled to the 640 px reference (sessions.to_reference). Allocation-free per frame. */
import type { Keypoints, World } from "../remote/logic";
import { EXERCISES } from "./exercises";
import { Escalation } from "./escalation";
import { FallDetector } from "./fallDetector";
import { AngleEma, KNEE_MEAN, MIN_CONF, N_ANGLES, PoseFrame, jointAnglesInto } from "./kinematics";
import { Mobility } from "./mobility";
import { RepCounter, calibratedLimits } from "./repCounter";

const SITTING_KNEE_DEG = 130.0;
export const SMOOTH_TAU_S = 0.08;       // what the old per-frame alpha of 0.35 amounted to at 30 fps
const REFERENCE_LONG_SIDE = 640;

const CAL_WINDOW_S = 1.5;               // the median of the last this-many seconds of standby
const CAL_MIN_SAMPLES = 8;
const CAL_MIN_SPAN_S = 0.8;
const CAL_MAX_STALE_S = 0.5;            // the newest sample must be this recent when counting starts
const CAL_SIDE_DIFF_DEG = 30.0;         // two-sided exercises: the sides' measured rests may differ by this much
const CAL_MIN_SHIFT_DEG = 25.0;         // a measured rest this close to the table's is the table's posture
const CAL_STABLE_DEG = 20.0;            // the angle must have stayed within this band
const CAL_CAP = 256;                    // samples per limb (1.5 s at up to ~170 fps)

export type PipelineOptions = {
  angles3d?: boolean;                   // ANGLES_3D: false measures angles in the image only
  calibrate?: boolean;                  // CALIBRATE: false keeps the table's rest/peak
  tauS?: number;                        // SMOOTH_TAU_S: 0 = the old fixed per-frame alpha (0.35)
};

export class Pipeline {
  readonly frame = new PoseFrame();
  readonly vals = new Float64Array(N_ANGLES);      // this frame's smoothed angles ...
  readonly has = new Uint8Array(N_ANGLES);         // ... and which of them exist
  readonly mobility = new Mobility();
  readonly fall = new FallDetector();
  readonly escalation = new Escalation();          // not cleared by reset(): resetting reps must not hide an alarm
  reps: RepCounter;
  /** false = standby: reps are NOT counted (someone still stepping into the frame must not score). Fall detection, escalation
   *  and the step counter run either way. */
  counting = true;
  poseVisible = false;
  posture = "unknown";
  private readonly ema: AngleEma;
  private readonly angles3d: boolean;
  private readonly calibrating: boolean;
  private lastT = 0;
  // standby samples (t, angle) of each limb's joint, newest last
  private readonly calT = [new Float64Array(CAL_CAP), new Float64Array(CAL_CAP)];
  private readonly calA = [new Float64Array(CAL_CAP), new Float64Array(CAL_CAP)];
  private readonly calHead = [0, 0];
  private readonly calCount = [0, 0];

  constructor(exercise: string, opts: PipelineOptions = {}) {
    this.angles3d = opts.angles3d ?? true;
    this.calibrating = opts.calibrate ?? true;
    const tau = opts.tauS ?? SMOOTH_TAU_S;
    this.ema = new AngleEma(0.35, tau || null);
    this.reps = new RepCounter(exercise);
  }

  get exercise(): string { return this.reps.exercise; }

  setExercise(exercise: string): void {
    this.reps = new RepCounter(exercise);
    this.ema.reset();
    this.calCount[0] = this.calCount[1] = 0;       // those angles belonged to the previous exercise's joints
  }

  reset(): void { this.reps.reset(); this.mobility.reset(); this.ema.reset(); }

  private sampleStandby(t: number): void {
    const r = this.reps;
    for (let i = 0; i < r.limbs.length; i++) {
      const key = r.keys[i];
      if (!this.has[key]) continue;
      if (this.calCount[i] === CAL_CAP) { this.calHead[i] = (this.calHead[i] + 1) % CAL_CAP; this.calCount[i]--; }
      const w = (this.calHead[i] + this.calCount[i]) % CAL_CAP;
      this.calT[i][w] = t; this.calA[i][w] = this.vals[key]; this.calCount[i]++;
      while (this.calCount[i] > 0 && this.calT[i][this.calHead[i]] < t - CAL_WINDOW_S) { this.calHead[i] = (this.calHead[i] + 1) % CAL_CAP; this.calCount[i]--; }
    }
  }

  /** The person's own starting angle for limb `i`, from the last second and a half of standby; null when that cannot be trusted
   *  (too few samples, they stopped, or the angle was moving about). Runs once per Start, so it may allocate. */
  private measuredRest(i: number): number | null {
    const ts = this.calT[i], as = this.calA[i], head = this.calHead[i], n = this.calCount[i];
    const picked: number[] = [];
    let first = NaN, last = NaN;
    for (let k = 0; k < n; k++) {
      const idx = (head + k) % CAL_CAP;
      if (ts[idx] < this.lastT - CAL_WINDOW_S) continue;
      if (picked.length === 0) first = ts[idx];
      last = ts[idx];
      picked.push(as[idx]);
    }
    if (picked.length < CAL_MIN_SAMPLES) return null;
    if (last - first < CAL_MIN_SPAN_S || this.lastT - last > CAL_MAX_STALE_S) return null;
    if (Math.max(...picked) - Math.min(...picked) > CAL_STABLE_DEG) return null;
    picked.sort((a, b) => a - b);
    const m = picked.length >> 1;
    return picked.length % 2 ? picked[m] : (picked[m - 1] + picked[m]) / 2;
  }

  /** Call when counting is about to start (standby -> counting). Each limb's `rest` becomes the median angle held in standby and
   *  its `peak` follows from the table's range and direction, so someone starting seated or lying gets a baseline that fits them.
   *  Anything doubtful keeps the table's standing values. True if any limb was calibrated. */
  calibrate(): boolean {
    const cfg = EXERCISES[this.exercise];
    const limbs = this.reps.limbs, n = limbs.length;
    const found: Array<[number, number] | null> = [];
    for (let i = 0; i < n; i++) {
      const measured = this.calibrating ? this.measuredRest(i) : null;
      found.push(measured === null ? null : calibratedLimits(cfg.rest, cfg.peak, measured));
    }
    const some = found.filter((f): f is [number, number] => f !== null);
    // Both sides start from the same posture: if one could not be measured or the two disagree, every limb keeps the table values
    // (calibrating just one would make one arm count against the person's baseline and the other against the standing table).
    if (n > 1) {
      const rests = some.map((f) => f[0]);
      if (some.length < n || Math.max(...rests) - Math.min(...rests) > CAL_SIDE_DIFF_DEG) found.fill(null);
    }
    // A posture close to the table's (standing) needs no calibration: shifting the peak by sensor noise only makes it harder to reach.
    if (found.every((f) => f === null || Math.abs(f[0] - cfg.rest) < CAL_MIN_SHIFT_DEG)) found.fill(null);
    let any = false;
    for (let i = 0; i < n; i++) {
      const f = found[i];
      if (f) { limbs[i].setLimits(f[0], f[1], true); any = true; } else limbs[i].useTable();
    }
    this.calCount[0] = this.calCount[1] = 0;
    return any;
  }

  /** One frame. `kp` = joints in pixels of the w x h frame (null = nobody), `world` = the optional metric 3D joints, `t` seconds. */
  process(kp: Keypoints | null, world: World | undefined, w: number, h: number, t: number): void {
    this.lastT = t;
    // bbox_from_keypoints: the confident joints, padded 5 %, clamped to the frame; fewer than 4 = no person
    let present = false, bw = 0, bh = 0, k = 1;
    if (kp) {
      let n = 0, x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const name in kp) {
        const p = kp[name];
        if (p[2] < MIN_CONF) continue;
        n++;
        if (p[0] < x0) x0 = p[0];
        if (p[0] > x1) x1 = p[0];
        if (p[1] < y0) y0 = p[1];
        if (p[1] > y1) y1 = p[1];
      }
      if (n >= 4) {
        present = true;
        const px = (x1 - x0) * 0.05, py = (y1 - y0) * 0.05;
        const bx0 = Math.max(0, x0 - px), by0 = Math.max(0, y0 - py), bx1 = Math.min(w, x1 + px), by1 = Math.min(h, y1 + py);
        const long = Math.max(w, h);
        if (long > 0 && long !== REFERENCE_LONG_SIDE) k = REFERENCE_LONG_SIDE / long;     // to_reference
        bw = (bx1 - bx0) * k; bh = (by1 - by0) * k;
      }
    }
    this.poseVisible = present;
    if (present) {
      this.frame.load(kp!, world, k);
      jointAnglesInto(this.frame, this.angles3d, this.vals, this.has);
      this.ema.update(this.vals, this.has, t);
      if (this.counting) this.reps.update(this.vals, this.has, t);
      else this.sampleStandby(t);                  // not counting yet: learn where this person starts from
      this.mobility.update(this.frame, t, this.reps.active());
    } else this.has.fill(0);
    this.fall.update(present ? this.frame : null, bw, bh, t);
    this.escalation.update(this.fall.status, this.fall.impact, t);

    if (!present) this.posture = "unknown";
    else if (this.fall.onFloor || (this.fall.tilt === this.fall.tilt && Math.round(this.fall.tilt * 10) / 10 > 55)) this.posture = "lying";   // the server compares the tilt rounded to 0.1
    else if ((this.has[KNEE_MEAN] ? this.vals[KNEE_MEAN] : 180) < SITTING_KNEE_DEG) this.posture = "sitting";
    else this.posture = "standing";
  }

  /** Rounded angles of the last frame, for the HUD. Allocates: call at UI rate, not per frame. */
  anglesSnapshot(names: readonly string[]): Record<string, number> {
    const out: Record<string, number> = {};
    for (let k = 0; k < N_ANGLES; k++) if (this.has[k]) out[names[k]] = Math.round(this.vals[k] * 10) / 10;
    return out;
  }
}
