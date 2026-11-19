/** Per-limb rep state machine with hysteresis and range-of-motion tracking. Port of backend/app/pipeline/rep_counter.py.
 *  Progress p = (angle - rest) / (peak - rest): 0 at rest, 1 at full peak, for flexion and raise exercises alike.
 *  IDLE -> FLEXING -> PEAK -> RETURNING -> (rep complete) -> IDLE */
import { EXERCISES } from "./exercises";
import { keyIndex } from "./kinematics";

export const IDLE = "IDLE", FLEXING = "FLEXING", PEAK = "PEAK", RETURNING = "RETURNING";
const START = 0.20, ABORT = 0.10;      // leave IDLE above START; FLEXING falls back to IDLE below ABORT
const PEAK_IN = 0.80, PEAK_OUT = 0.60;
const DONE = 0.15;                     // RETURNING completes the rep below DONE
const MIN_REP_S = 0.4, MAX_REP_S = 10.0;

const CAL_ANGLE_LO = 15.0, CAL_ANGLE_HI = 178.0;   // a calibrated peak stays inside what a joint can do
const CAL_MIN_RANGE = 25.0;                         // less room than this between rest and peak is no exercise

/** (rest, peak) for a person who starts at `measuredRest`: the table's range and direction from their own rest; null when that
 *  leaves too little room (the table values then stay in use). */
export function calibratedLimits(restTable: number, peakTable: number, measuredRest: number): [number, number] | null {
  const direction = peakTable > restTable ? 1 : -1;
  let peak = measuredRest + direction * Math.abs(peakTable - restTable);
  peak = Math.min(Math.max(peak, CAL_ANGLE_LO), CAL_ANGLE_HI);
  if (Math.abs(peak - measuredRest) < CAL_MIN_RANGE) return null;
  return [measuredRest, peak];
}

const r1 = (x: number) => Math.round(x * 10) / 10;

export class LimbFSM {
  readonly tableRest: number; readonly tablePeak: number;
  rest: number; peak: number;
  calibrated = false;
  state = IDLE;
  reps = 0;
  lastRom = 0; bestRom = 0;
  angle = NaN;                          // NaN = none yet
  private tStart = 0;
  private amin = 0; private amax = 0;
  private idleMin = 0; private idleMax = 0; private hasIdle = false;   // angle extremes seen while IDLE (the ROM includes the rest position)

  constructor(rest: number, peak: number) {
    this.tableRest = rest; this.tablePeak = peak; this.rest = rest; this.peak = peak;
  }
  setLimits(rest: number, peak: number, calibrated = true): void { this.rest = rest; this.peak = peak; this.calibrated = calibrated; }
  useTable(): void { this.rest = this.tableRest; this.peak = this.tablePeak; this.calibrated = false; }
  reset(): void {
    this.state = IDLE; this.reps = 0; this.lastRom = 0; this.bestRom = 0; this.angle = NaN;
    this.tStart = 0; this.amin = this.amax = 0; this.hasIdle = false;
  }

  /** True on the frame a rep completes. */
  update(angle: number, t: number): boolean {
    this.angle = angle;
    const p = (angle - this.rest) / (this.peak - this.rest);
    if (this.state === IDLE) {
      if (!this.hasIdle) { this.idleMin = this.idleMax = angle; this.hasIdle = true; }
      if (angle < this.idleMin) this.idleMin = angle;
      if (angle > this.idleMax) this.idleMax = angle;
      if (p > START) {
        this.state = FLEXING; this.tStart = t;
        this.amin = this.idleMin; this.amax = this.idleMax;
        this.hasIdle = false;
      }
      return false;
    }
    if (angle < this.amin) this.amin = angle;
    if (angle > this.amax) this.amax = angle;
    if (this.state === FLEXING) {
      if (p < ABORT) this.state = IDLE;
      else if (p > PEAK_IN) this.state = PEAK;
    } else if (this.state === PEAK) {
      if (p < PEAK_OUT) this.state = RETURNING;
    } else if (this.state === RETURNING) {
      if (p > PEAK_IN) this.state = PEAK;
      else if (p < DONE) {
        this.state = IDLE;
        const dt = t - this.tStart;
        if (dt >= MIN_REP_S && dt <= MAX_REP_S) {
          this.reps++;
          this.lastRom = this.amax - this.amin;
          if (this.lastRom > this.bestRom) this.bestRom = this.lastRom;
          return true;
        }
      }
    }
    return false;
  }

  snapshot() {
    return {
      reps: this.reps, state: this.state, angle: this.angle === this.angle ? r1(this.angle) : null,
      last_rom: r1(this.lastRom), best_rom: r1(this.bestRom), rest: r1(this.rest), peak: r1(this.peak), calibrated: this.calibrated,
    };
  }
}

export class RepCounter {
  readonly exercise: string;
  readonly sides: string[];             // "L", "R" or "B"
  readonly keys: number[];              // each side's angle slot (kinematics.ANGLE_KEYS)
  readonly limbs: LimbFSM[];
  constructor(exercise: string) {
    const cfg = EXERCISES[exercise];
    this.exercise = exercise;
    this.sides = Object.keys(cfg.joints);
    this.keys = this.sides.map((s) => keyIndex(cfg.joints[s]));
    this.limbs = this.sides.map(() => new LimbFSM(cfg.rest, cfg.peak));
  }
  update(vals: Float64Array, has: Uint8Array, t: number): void {
    for (let i = 0; i < this.limbs.length; i++) if (has[this.keys[i]]) this.limbs[i].update(vals[this.keys[i]], t);
  }
  active(): boolean {
    for (let i = 0; i < this.limbs.length; i++) if (this.limbs[i].state !== IDLE) return true;
    return false;
  }
  total(): number {
    let n = 0;
    for (let i = 0; i < this.limbs.length; i++) n += this.limbs[i].reps;
    return n;
  }
  reset(): void { for (let i = 0; i < this.limbs.length; i++) this.limbs[i].reset(); }
  snapshot(): Record<string, ReturnType<LimbFSM["snapshot"]>> {
    const out: Record<string, ReturnType<LimbFSM["snapshot"]>> = {};
    for (let i = 0; i < this.sides.length; i++) out[this.sides[i]] = this.limbs[i].snapshot();
    return out;
  }
}
