/** Fall escalation: detector CONFIRMED -> "Are you okay?" countdown -> ALERTED unless acknowledged or recovered. Port of
 *  backend/app/pipeline/escalation.py (see its docstring for the state diagram).
 *  IDLE -> COUNTDOWN -> ALERTED; COUNTDOWN -> ACKNOWLEDGED ("I'm OK"); ACKNOWLEDGED -> COUNTDOWN (still down RECHECK_S later). */
import { CONFIRMED, NORMAL } from "./fallDetector";

export const IDLE = "IDLE", COUNTDOWN = "COUNTDOWN", ALERTED = "ALERTED", ACKNOWLEDGED = "ACKNOWLEDGED";
export const COUNTDOWN_S = 10.0;        // FallCheck.tsx draws its ring to this
const RECHECK_S = 120.0;

export class Escalation {
  state = IDLE;
  impact = false;
  fired = false;                        // true for the single update() in which ALERTED is entered
  rechecks = 0;
  secondsLeft = 0;
  private deadline = 0; private ackedAt = 0;
  private ackFlag = false; private escFlag = false;
  private test = false;                // the Test Fall button: a simulated episode is running

  ack(): void { this.ackFlag = true; }
  /** Skips the rest of the countdown (voice "help", Trigger Emergency). Only acts during COUNTDOWN. */
  escalate(): void { this.escFlag = true; }
  simulate(): void { this.test = true; }

  update(status: string, fallImpact: boolean, t: number): void {
    const ack = this.ackFlag; this.ackFlag = false;      // consumed every frame: a stale ack cannot pre-answer a later fall
    const esc = this.escFlag; this.escFlag = false;
    this.fired = false;
    if (this.state === IDLE) {
      if (status === CONFIRMED || this.test) {
        this.state = COUNTDOWN; this.impact = this.test || fallImpact;
        this.deadline = t + COUNTDOWN_S;
      }
    } else if (this.state === COUNTDOWN) {
      if (ack) { this.state = ACKNOWLEDGED; this.ackedAt = t; }
      else if (esc) { this.state = ALERTED; this.fired = true; }
      else if (status === NORMAL && !this.test) this.state = IDLE;       // a simulated fall has no detector state to recover from
      else if (t >= this.deadline) { this.state = ALERTED; this.fired = true; }
    } else if (this.state === ALERTED) {
      if (ack || (status === NORMAL && !this.test)) {
        if (status === NORMAL) this.state = IDLE;
        else { this.state = ACKNOWLEDGED; this.ackedAt = t; }
      }
    } else if (status === NORMAL) this.state = IDLE;                     // ACKNOWLEDGED: quiet until the episode is over
    else if (t - this.ackedAt >= RECHECK_S) { this.state = COUNTDOWN; this.deadline = t + COUNTDOWN_S; this.rechecks++; }
    if (this.state !== COUNTDOWN && this.state !== ALERTED) this.test = false;
    if (this.state === IDLE) this.rechecks = 0;
    this.secondsLeft = this.state === COUNTDOWN ? Math.max(0, this.deadline - t) : 0;
  }

  snapshot() {
    return { state: this.state, seconds_left: Math.round(this.secondsLeft * 10) / 10, impact: this.impact, rechecks: this.rechecks };
  }
}
