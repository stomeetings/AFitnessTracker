/** Joint angles from a pose (3D world coordinates when every joint of an angle has them, else 2D) and the angle smoothing.
 *  Port of backend/app/pipeline/kinematics.py. Hot path rules (see .claude/skills/cv-fitness-pipeline): everything lives in
 *  preallocated typed arrays, joints are addressed by index, nothing is created per frame. */
import type { Keypoints, World } from "../remote/logic";

export const MIN_CONF = 0.5;
export const N_JOINTS = 13;
const STRIDE = 7;                     // per joint: x, y, conf, wx, wy, wz, hasWorld

/** The 13 joints the pipeline reads, in index order. */
export const JOINT_NAMES = ["nose", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
  "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"] as const;
export const NOSE = 0, L_SHOULDER = 1, R_SHOULDER = 2, L_ELBOW = 3, R_ELBOW = 4, L_WRIST = 5, R_WRIST = 6,
  L_HIP = 7, R_HIP = 8, L_KNEE = 9, R_KNEE = 10, L_ANKLE = 11, R_ANKLE = 12;

/** One frame's pose as flat numbers, in the REFERENCE scale (see `load`). Reused for every frame. */
export class PoseFrame {
  readonly v = new Float64Array(N_JOINTS * STRIDE);

  /** Copies the joints of `kp` (and `world`) in, multiplying image pixels by `k` (the scale to the 640 px reference, see
   *  sessions.to_reference). A joint that is absent gets confidence -1. Confidence is clamped to 0..1 like parse_keypoints. */
  load(kp: Keypoints, world: World | undefined, k: number): void {
    const v = this.v;
    for (let j = 0; j < N_JOINTS; j++) {
      const name = JOINT_NAMES[j], p = kp[name], o = j * STRIDE;
      if (!p) { v[o + 2] = -1; v[o + 6] = 0; continue; }
      v[o] = p[0] * k; v[o + 1] = p[1] * k;
      v[o + 2] = p[2] < 0 ? 0 : p[2] > 1 ? 1 : p[2];
      const w = world ? world[name] : undefined;
      if (w) { v[o + 3] = w[0]; v[o + 4] = w[1]; v[o + 5] = w[2]; v[o + 6] = 1; } else v[o + 6] = 0;
    }
  }
  /** Joint `j` is present and confident (res.get(name) is not None). */
  ok(j: number, minConf = MIN_CONF): boolean { return this.v[j * STRIDE + 2] >= minConf; }
  x(j: number): number { return this.v[j * STRIDE]; }
  y(j: number): number { return this.v[j * STRIDE + 1]; }
}

// Angle keys: index = position in the output arrays.
export const ANGLE_KEYS = ["l_elbow", "l_shoulder", "l_knee", "l_hip", "l_hip_abd", "r_elbow", "r_shoulder", "r_knee", "r_hip", "r_hip_abd",
  "knee_mean", "hip_mean", "elbow_mean"] as const;
export const N_ANGLES = ANGLE_KEYS.length;
export const keyIndex = (name: string): number => (ANGLE_KEYS as readonly string[]).indexOf(name);
export const KNEE_MEAN = 10;

// [output index, a, b, c]: the angle is measured at b. Side l first, then r (the Python loop order does not matter here).
const DEFS: ReadonlyArray<readonly [number, number, number, number]> = [
  [0, L_SHOULDER, L_ELBOW, L_WRIST], [1, L_HIP, L_SHOULDER, L_ELBOW], [2, L_HIP, L_KNEE, L_ANKLE], [3, L_SHOULDER, L_HIP, L_KNEE],
  [4, R_HIP, L_HIP, L_KNEE],                                  // l_hip_abd: pelvis line (to the other hip) and the thigh
  [5, R_SHOULDER, R_ELBOW, R_WRIST], [6, R_HIP, R_SHOULDER, R_ELBOW], [7, R_HIP, R_KNEE, R_ANKLE], [8, R_SHOULDER, R_HIP, R_KNEE],
  [9, L_HIP, R_HIP, R_KNEE],
];
const MEANS: ReadonlyArray<readonly [number, number, number]> = [[10, 2, 7], [11, 3, 8], [12, 0, 5]];   // knee, hip, elbow: [out, l, r]

const DEG = 180 / Math.PI;
const clip1 = (x: number) => (x < -1 ? -1 : x > 1 ? 1 : x);

/** angle_deg for 2D points; NaN when a vector has no length (Python returns None). */
function angle2(ax: number, ay: number, bx: number, by: number, cx: number, cy: number): number {
  const v1x = ax - bx, v1y = ay - by, v2x = cx - bx, v2y = cy - by;
  const n = Math.sqrt(v1x * v1x + v1y * v1y) * Math.sqrt(v2x * v2x + v2y * v2y);
  if (n < 1e-9) return NaN;
  return Math.acos(clip1((v1x * v2x + v1y * v2y) / n)) * DEG;
}
function angle3(v: Float64Array, a: number, b: number, c: number): number {
  const ia = a * STRIDE + 3, ib = b * STRIDE + 3, ic = c * STRIDE + 3;
  const v1x = v[ia] - v[ib], v1y = v[ia + 1] - v[ib + 1], v1z = v[ia + 2] - v[ib + 2];
  const v2x = v[ic] - v[ib], v2y = v[ic + 1] - v[ib + 1], v2z = v[ic + 2] - v[ib + 2];
  const n = Math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z) * Math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z);
  if (n < 1e-9) return NaN;
  return Math.acos(clip1((v1x * v2x + v1y * v2y + v1z * v2z) / n)) * DEG;
}

/** joint_angles: fills `out` with the angles present in this frame and marks them in `has` (1 = present). Missing joints, or a
 *  joint below confidence 0.5, omit the angle. 3D is used when `use3d` and all three joints carry a world position. */
export function jointAnglesInto(f: PoseFrame, use3d: boolean, out: Float64Array, has: Uint8Array): void {
  const v = f.v;
  has.fill(0);
  for (let d = 0; d < DEFS.length; d++) {
    const def = DEFS[d], a = def[1], b = def[2], c = def[3];
    if (!(f.ok(a) && f.ok(b) && f.ok(c))) continue;
    const ang = use3d && v[a * STRIDE + 6] && v[b * STRIDE + 6] && v[c * STRIDE + 6]
      ? angle3(v, a, b, c)
      : angle2(v[a * STRIDE], v[a * STRIDE + 1], v[b * STRIDE], v[b * STRIDE + 1], v[c * STRIDE], v[c * STRIDE + 1]);
    if (ang === ang) { out[def[0]] = ang; has[def[0]] = 1; }         // NaN check
  }
  for (let m = 0; m < MEANS.length; m++) {
    const mean = MEANS[m], l = mean[1], r = mean[2];
    if (has[l] && has[r]) { out[mean[0]] = (out[l] + out[r]) / 2; has[mean[0]] = 1; }
    else if (has[l]) { out[mean[0]] = out[l]; has[mean[0]] = 1; }
    else if (has[r]) { out[mean[0]] = out[r]; has[mean[0]] = 1; }
  }
}

/** Per-key exponential moving average over the angle slots. A key absent in a frame is neither updated nor reported (the rep
 *  FSMs freeze); its smoothed value is kept for when it returns. Two modes, as in Python: fixed `alpha`, or time based
 *  (`tau` seconds: alpha = 1 - exp(-dt / tau), the same reaction over TIME at 10 and 30 fps). */
export class AngleEma {
  private readonly s = new Float64Array(N_ANGLES);
  private readonly seen = new Uint8Array(N_ANGLES);
  private t = 0;
  private hasT = false;
  constructor(private readonly alpha = 0.35, private readonly tau: number | null = null) {}

  /** Smooths `vals` (where `has`) in place. */
  update(vals: Float64Array, has: Uint8Array, t: number): void {
    let a = this.alpha;
    if (this.tau) {
      if (this.hasT) a = 1 - Math.exp(-Math.max(t - this.t, 0) / this.tau);     // dt 0 (a repeated timestamp) holds the value
      this.t = t; this.hasT = true;
    }
    for (let k = 0; k < N_ANGLES; k++) {
      if (!has[k]) continue;
      this.s[k] = this.seen[k] ? this.s[k] + a * (vals[k] - this.s[k]) : vals[k];
      this.seen[k] = 1;
      vals[k] = this.s[k];
    }
  }
  reset(): void { this.seen.fill(0); this.hasT = false; }
}
