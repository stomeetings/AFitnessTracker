/** Walking / step-cadence estimate from ankle separation oscillation and hip displacement (2D image space, all lengths in torso
 *  lengths, so distance-independent). Port of backend/app/pipeline/mobility.py. Ring buffers of numbers, no per-frame allocation. */
import { L_ANKLE, L_HIP, L_SHOULDER, R_ANKLE, R_HIP, R_SHOULDER, type PoseFrame } from "./kinematics";

const SEP_HI = 0.45, SEP_LO = 0.25;     // a step = ankle separation rises above HI then falls below LO
const MIN_STEP_S = 0.3;
const CADENCE_WINDOW_S = 10.0;
const DISP_WINDOW_S = 2.0;
const WALK_DISP = 0.5;
const RECENT_STEP_S = 2.0;
const HIP_CAP = 512, STEP_CAP = 128;     // 2 s of hips at up to 256 fps, 10 s of steps at up to 12 steps a second

export class Mobility {
  steps = 0;
  state = "Stationary";
  private readonly hipT = new Float64Array(HIP_CAP);
  private readonly hipX = new Float64Array(HIP_CAP);
  private readonly hipY = new Float64Array(HIP_CAP);
  private hHead = 0; private hCount = 0;
  private readonly stepT = new Float64Array(STEP_CAP);
  private sHead = 0; private sCount = 0;
  private armed = false;
  private lastStepT = -1e9;

  reset(): void {
    this.steps = 0; this.state = "Stationary";
    this.hHead = this.hCount = this.sHead = this.sCount = 0;
    this.armed = false; this.lastStepT = -1e9;
  }

  update(f: PoseFrame, t: number, exercising: boolean): void {
    if (!(f.ok(L_ANKLE) && f.ok(R_ANKLE) && f.ok(L_HIP) && f.ok(R_HIP) && f.ok(L_SHOULDER) && f.ok(R_SHOULDER))) return;
    const hx = (f.x(L_HIP) + f.x(R_HIP)) / 2, hy = (f.y(L_HIP) + f.y(R_HIP)) / 2;
    const sx = (f.x(L_SHOULDER) + f.x(R_SHOULDER)) / 2, sy = (f.y(L_SHOULDER) + f.y(R_SHOULDER)) / 2;
    const torso = Math.max(Math.hypot(hx - sx, hy - sy), 1e-6);
    const sep = Math.hypot(f.x(L_ANKLE) - f.x(R_ANKLE), f.y(L_ANKLE) - f.y(R_ANKLE)) / torso;

    if (sep > SEP_HI) this.armed = true;
    else if (this.armed && sep < SEP_LO && t - this.lastStepT > MIN_STEP_S) {
      this.steps++;
      if (this.sCount === STEP_CAP) { this.sHead = (this.sHead + 1) % STEP_CAP; this.sCount--; }
      this.stepT[(this.sHead + this.sCount) % STEP_CAP] = t; this.sCount++;
      this.lastStepT = t;
      this.armed = false;
    }
    while (this.sCount > 0 && t - this.stepT[this.sHead] > CADENCE_WINDOW_S) { this.sHead = (this.sHead + 1) % STEP_CAP; this.sCount--; }

    if (this.hCount === HIP_CAP) { this.hHead = (this.hHead + 1) % HIP_CAP; this.hCount--; }
    const w = (this.hHead + this.hCount) % HIP_CAP;
    this.hipT[w] = t; this.hipX[w] = hx; this.hipY[w] = hy; this.hCount++;
    while (this.hCount > 0 && t - this.hipT[this.hHead] > DISP_WINDOW_S) { this.hHead = (this.hHead + 1) % HIP_CAP; this.hCount--; }
    const disp = Math.hypot(this.hipX[this.hHead] - hx, this.hipY[this.hHead] - hy) / torso;

    if (disp > WALK_DISP && t - this.lastStepT < RECENT_STEP_S) this.state = "Walking";
    else if (exercising) this.state = "Active Exercise";
    else this.state = "Stationary";
  }

  snapshot() {
    return { state: this.state, steps: this.steps, cadence_spm: Math.round(this.sCount * 60 / CADENCE_WINDOW_S) };
  }
}
