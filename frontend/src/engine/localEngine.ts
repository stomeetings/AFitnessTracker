/** The on-device counterpart of backend/app/sessions.Session for a phone: one patient's pipeline plus the commands that change it
 *  (start, stop, pause, resume, exercise, reset, "I'm OK", emergency, test fall). Commands apply at once: they come from the same
 *  thread that feeds the frames, so no pending-flag hand-over is needed. It produces the same `Telemetry` the server sends, and
 *  tells an optional listener about the moments worth keeping (see EngineEvent) without doing any work per frame. */
import type { Keypoints, World } from "../remote/logic";
import { EXERCISES, EXERCISE_NAMES } from "./exercises";
import { ANGLE_KEYS } from "./kinematics";
import { Pipeline, type PipelineOptions } from "./pipeline";

/** What the session recorder keeps (store/recorder.ts). Created only when something happens (a rep, a fall change, a command),
 *  never per frame. */
export type EngineEventType = "start" | "stop" | "pause" | "resume" | "exercise" | "rep" | "fall" | "alert";
export type EngineEvent = { type: EngineEventType; payload: Record<string, unknown> };

export class LocalEngine {
  readonly pipeline: Pipeline;
  /** Paused by voice ("pause" / "hold please"): "let's go" or "resume" then continue without a reset. */
  paused = false;
  /** Bumped when something the screen must show at once changed (fall status, alert state, counting, exercise, rep total):
   *  the caller compares it with the last one it saw instead of building telemetry per frame. */
  version = 0;
  latencyMs = 0;
  /** Told about start / stop / pause / resume / exercise changes, every completed rep, fall status and alert state changes. A
   *  listener that throws is ignored: recording must never disturb the workout. */
  listener: ((e: EngineEvent) => void) | null = null;
  private frameW = 0; private frameH = 0;
  private sFall = ""; private sAlert = ""; private sCounting = false; private sExercise = ""; private sTotal = -1;
  private ready = false;                               // the constructor's first look is not an event

  constructor(exercise: string = EXERCISE_NAMES[0], opts: PipelineOptions = {}) {
    this.pipeline = new Pipeline(EXERCISES[exercise] ? exercise : EXERCISE_NAMES[0], opts);
    this.pipeline.counting = false;                    // a phone starts in standby: stepping into the frame must not score
    this.changed();
    this.ready = true;
  }

  /** One frame (the hot path): see Pipeline.process. Allocates nothing. */
  process(kp: Keypoints | null, world: World | undefined, w: number, h: number, t: number): void {
    this.frameW = w; this.frameH = h;
    this.pipeline.process(kp, world, w, h, t);
    this.changed();
  }

  private emit(type: EngineEventType, payload: Record<string, unknown>): void {
    if (!this.listener) return;
    try { this.listener({ type, payload }); } catch { /* recording must never disturb the workout */ }
  }

  private changed(): void {
    const p = this.pipeline, total = p.reps.total();
    if (p.fall.status === this.sFall && p.escalation.state === this.sAlert && p.counting === this.sCounting
        && p.exercise === this.sExercise && total === this.sTotal) return;                     // the usual frame: two compares, done
    const fall = p.fall.status !== this.sFall, alert = p.escalation.state !== this.sAlert;
    const exercise = p.exercise !== this.sExercise, rep = total > this.sTotal && this.sTotal >= 0;
    this.sFall = p.fall.status; this.sAlert = p.escalation.state; this.sCounting = p.counting; this.sExercise = p.exercise; this.sTotal = total;
    this.version++;
    if (!this.ready || !this.listener) return;
    if (exercise) this.emit("exercise", { exercise: p.exercise });
    if (rep) this.emit("rep", { total, limbs: p.reps.snapshot() });
    if (fall) this.emit("fall", { status: p.fall.status, impact: p.fall.impact, tilt_deg: p.fall.snapshot().tilt_deg });
    if (alert) this.emit("alert", { state: p.escalation.state, impact: p.escalation.impact, rechecks: p.escalation.rechecks });
  }

  // ---- commands (the same meaning as Session.request_*) ------------------------------------------------------------------

  /** "Let's go" / Start: counters back to 0, the rest angles measured from the standby posture, rep counting on. */
  start(): void {
    this.paused = false;
    this.pipeline.reset();
    const calibrated = !this.pipeline.counting ? this.pipeline.calibrate() : false;
    this.pipeline.counting = true;
    this.emit("start", { exercise: this.pipeline.exercise, calibrated });
    this.changed();
  }
  /** The HUD's Pause button / "stop session": back to standby, the counts stay on screen until the next start. */
  stop(): void {
    this.paused = false; this.pipeline.counting = false;
    this.emit("stop", { exercise: this.pipeline.exercise });
    this.changed();
  }
  /** Voice "pause": only a running session can be paused; the counts stay. False when there was nothing to pause. */
  pause(): boolean {
    if (!this.pipeline.counting) return false;
    this.pipeline.counting = false; this.paused = true;
    this.emit("pause", {});
    this.changed();
    return true;
  }
  /** Counting on again WITHOUT the reset start() does (like the server, it re-measures the rest angles from the posture held). */
  resume(): void {
    if (!this.pipeline.counting) this.pipeline.calibrate();
    this.pipeline.counting = true; this.paused = false;
    this.emit("resume", {});
    this.changed();
  }
  setExercise(name: string): void {
    if (EXERCISES[name] && name !== this.pipeline.exercise) this.pipeline.setExercise(name);
    this.changed();
  }
  reset(): void { this.pipeline.reset(); this.changed(); }
  ack(): void { this.pipeline.escalation.ack(); }
  escalate(): void { this.pipeline.escalation.escalate(); }
  simulateFall(): void { this.pipeline.escalation.simulate(); }

  /** What the UI shows (the same fields as Session.telemetry() plus the frame size). Allocates: call at UI rate, not per frame. */
  telemetry() {
    const p = this.pipeline;
    return {
      type: "t",
      latency_ms: Math.round(this.latencyMs * 10) / 10, pose_visible: p.poseVisible, exercise: p.exercise, counting: p.counting,
      angles: p.anglesSnapshot(ANGLE_KEYS), reps: p.reps.snapshot(), mobility: p.mobility.snapshot(),
      fall: { ...p.fall.snapshot(), alert: p.escalation.snapshot() },
      posture: p.posture, frame: { w: this.frameW, h: this.frameH }, error: null,
    };
  }
}
