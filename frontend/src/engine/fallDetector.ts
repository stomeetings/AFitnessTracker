/** 3-stage fall detection: NORMAL -> SUSPECTED -> CONFIRMED (-> NORMAL on recovery). Port of backend/app/pipeline/fall_detector.py;
 *  read that file's docstring for the reasoning behind every constant. Pixels are in the 640 px reference scale (PoseFrame.load).
 *  The 30 s hip history is a ring buffer of numbers: no allocation per frame. */
import {
  L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, R_ANKLE, R_HIP, R_SHOULDER, type PoseFrame,
} from "./kinematics";

export const NORMAL = "NORMAL", SUSPECTED = "SUSPECTED", CONFIRMED = "CONFIRMED";

const TILT_SUSPECT = 55.0;
const ASPECT_WIDE = 1.2;
const DROP_V = 2.5;
const HIP_DROP = 1.2;
const HIP_NEAR_ANKLE = 0.4;
const FLAT_EXTENT = 0.8;
const ASPECT_LYING = 1.6;
const COMPRESSED = 0.55;
const LEGS_SHORT = 0.27;
const ANKLE_BIN_PX = 40;
const INVERTED = 0.3;
const DWELL_S = 5.0;
const RECOVER_S = 0.5;
const VEL_WINDOW_S = 0.3;
const BASELINE_S = 30.0;
const LOST_AFTER_DOWN_S = 6.0;
const CAP = 4096;                       // history slots: 30 s at up to ~136 fps (older entries are dropped when full)

const r1 = (x: number) => Math.round(x * 10) / 10;
const r2 = (x: number) => Math.round(x * 100) / 100;

export class FallDetector {
  status = NORMAL;
  tilt = NaN;                           // NaN = None
  vel = 0;
  aspect = NaN;
  hipDrop = NaN;
  onFloor = false;
  impact = false;
  // history ring: (t, hip y, torso length)
  private readonly ht = new Float64Array(CAP);
  private readonly hy = new Float64Array(CAP);
  private readonly hr = new Float64Array(CAP);
  private head = 0; private count = 0;
  private stand = new Map<number, number>();            // ankle-y bin -> tallest shoulder-to-ankle height seen there
  private lastDropT = -1e9; private lastDownT = -1e9;
  private since = 0;
  private clearSince = NaN;
  private lastT = 0;

  constructor() { this.reset(); }

  reset(): void {
    this.status = NORMAL; this.tilt = NaN; this.vel = 0; this.aspect = NaN; this.hipDrop = NaN;
    this.onFloor = false; this.impact = false;
    this.head = 0; this.count = 0;
    this.stand.clear();
    this.lastDropT = this.lastDownT = -1e9;
    this.since = 0; this.clearSince = NaN;
  }

  private push(t: number, hipY: number, torso: number): void {
    if (this.count === CAP) { this.head = (this.head + 1) % CAP; this.count--; }
    const i = (this.head + this.count) % CAP;
    this.ht[i] = t; this.hy[i] = hipY; this.hr[i] = torso;
    this.count++;
  }

  /** Updates tilt/vel/aspect/hipDrop; true when this frame looks like a person down on the floor. `bw`/`bh` is the pose's box. */
  private measure(f: PoseFrame, bw: number, bh: number, t: number): boolean {
    const hasLs = f.ok(L_SHOULDER), hasRs = f.ok(R_SHOULDER), hasLh = f.ok(L_HIP), hasRh = f.ok(R_HIP);
    this.aspect = bw / Math.max(bh, 1e-6);
    if (!(hasLs && hasRs && hasLh && hasRh)) {
      // Torso not tracked (common when lying): judge by the box shape alone, unless it is a head-and-shoulders close-up.
      this.tilt = NaN; this.hipDrop = NaN; this.vel = 0;
      const closeup = hasLs && hasRs && Math.abs(f.x(L_SHOULDER) - f.x(R_SHOULDER)) > 0.5 * bw;
      return this.aspect > ASPECT_LYING && !closeup;
    }
    const shx = (f.x(L_SHOULDER) + f.x(R_SHOULDER)) / 2, shy = (f.y(L_SHOULDER) + f.y(R_SHOULDER)) / 2;
    const hipX = (f.x(L_HIP) + f.x(R_HIP)) / 2, hipY = (f.y(L_HIP) + f.y(R_HIP)) / 2;
    const dx = shx - hipX, dy = shy - hipY;
    const torso = Math.max(Math.hypot(dx, dy), 1e-6);
    this.tilt = Math.atan2(Math.abs(dx), Math.abs(dy)) * 180 / Math.PI;

    this.push(t, hipY, torso);
    while (this.count > 1 && t - this.ht[this.head] > BASELINE_S) { this.head = (this.head + 1) % CAP; this.count--; }
    let torsoRef = -Infinity, baseHip = Infinity;
    for (let n = 0; n < this.count; n++) {
      const i = (this.head + n) % CAP;
      if (this.hr[i] > torsoRef) torsoRef = this.hr[i];
      if (this.hy[i] < baseHip) baseHip = this.hy[i];
    }
    this.hipDrop = (hipY - baseHip) / torsoRef;

    let t0 = t, y0 = hipY;                                   // the oldest entry still inside the velocity window
    for (let n = 0; n < this.count; n++) {
      const i = (this.head + n) % CAP;
      if (t - this.ht[i] <= VEL_WINDOW_S) { t0 = this.ht[i]; y0 = this.hy[i]; break; }
    }
    this.vel = t > t0 ? ((hipY - y0) / torsoRef) / (t - t0) : 0;     // positive = moving down the image
    if (this.vel > DROP_V && this.hipDrop > HIP_DROP / 2) this.lastDropT = t;

    const hasLa = f.ok(L_ANKLE), hasRa = f.ok(R_ANKLE);
    const nearAnkle = hasLa && hasRa && ((f.y(L_ANKLE) + f.y(R_ANKLE)) / 2 - hipY) < HIP_NEAR_ANKLE * torsoRef;
    // flat: the whole body at one height; needs no standing baseline and no visible ankles
    let minY = Math.min(f.y(L_SHOULDER), f.y(R_SHOULDER), f.y(L_HIP), f.y(R_HIP));
    let maxY = Math.max(f.y(L_SHOULDER), f.y(R_SHOULDER), f.y(L_HIP), f.y(R_HIP));
    let legs = false;
    for (let j = L_KNEE; j <= R_ANKLE; j++) {                // knees and ankles: joints 9..12
      if (!f.ok(j)) continue;
      legs = true;
      const y = f.y(j);
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const flat = legs && (maxY - minY) < FLAT_EXTENT * torsoRef;
    const low = this.hipDrop > HIP_DROP || nearAnkle || flat;
    const horizontal = this.tilt > TILT_SUSPECT || this.aspect > ASPECT_WIDE;

    // lying along the camera's line of sight: the skeleton stays upright in the image, only shorter
    let compressed = false, inverted = false;
    if (hasLa && hasRa) {
      const ankleY = (f.y(L_ANKLE) + f.y(R_ANKLE)) / 2;
      const body = ankleY - shy;                             // shoulder-to-ankle height; negative = upside down
      const b = Math.floor(ankleY / ANKLE_BIN_PX);
      const expected = Math.max(this.stand.get(b - 1) ?? 0, this.stand.get(b) ?? 0, this.stand.get(b + 1) ?? 0);
      this.stand.set(b, Math.max(this.stand.get(b) ?? 0, body));
      compressed = expected > 0 && body < COMPRESSED * expected && (ankleY - hipY) < LEGS_SHORT * expected;
      inverted = body < -INVERTED * torsoRef;
    }
    return (horizontal && low) || compressed || inverted;
  }

  /** `f` is null when nobody is in view (`bw`/`bh` are then ignored). */
  update(f: PoseFrame | null, bw: number, bh: number, t: number): void {
    this.lastT = t;
    let fallen: boolean, dropping: boolean;
    if (f) {
      fallen = this.measure(f, bw, bh, t);
      dropping = this.lastDropT === t;
      if (fallen) this.lastDownT = t;
    } else {                                                 // pose lost: down only if it vanished soon after a drop or on the floor
      fallen = (t - Math.max(this.lastDropT, this.lastDownT)) < LOST_AFTER_DOWN_S;
      dropping = false;
      this.vel = 0;
    }
    this.onFloor = fallen;
    const prev = this.status;
    if (this.status === NORMAL) {
      if (dropping || fallen) this.status = SUSPECTED;
    } else if (fallen) {
      this.clearSince = NaN;
      if (this.status === SUSPECTED && t - this.since >= DWELL_S) this.status = CONFIRMED;
    } else {                                                 // upright again, debounced: one stray frame must not clear a real fall
      if (this.clearSince !== this.clearSince) this.clearSince = t;
      if (t - this.clearSince >= RECOVER_S) this.status = NORMAL;
    }
    if (this.status !== prev) { this.since = t; this.clearSince = NaN; }
    if (this.status === NORMAL) this.impact = false;
    else if (dropping || (prev === NORMAL && (t - this.lastDropT) < LOST_AFTER_DOWN_S)) this.impact = true;
  }

  snapshot() {
    return {
      status: this.status, impact: this.impact, on_floor: this.onFloor,
      tilt_deg: this.tilt === this.tilt ? r1(this.tilt) : null, drop_velocity: r2(this.vel),
      hip_drop: this.hipDrop === this.hipDrop ? r2(this.hipDrop) : null,
      aspect_ratio: this.aspect === this.aspect ? r2(this.aspect) : null,
      seconds_in_state: r1(this.lastT - this.since),
    };
  }
}
