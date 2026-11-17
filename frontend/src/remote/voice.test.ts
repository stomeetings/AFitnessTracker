import { describe, expect, it } from "vitest";
import { AUDIO_MIME_PREFERENCE, Vad, holdQuiet, intentEffects, isQuiet, micErrorText, pickAudioMime, rms } from "./voice";

describe("intentEffects (the one mapping both modes use)", () => {
  it("show / hide preview open / close the Exercise Guide", () => {
    expect(intentEffects("show_preview", false)).toEqual({ showGuide: true });
    expect(intentEffects("hide_preview", false)).toEqual({ showGuide: false });
  });
  it("start session closes the guide so the screen goes cleanly into tracking", () => {
    expect(intentEffects("start_session", false)).toEqual({ showGuide: false });
  });
  it("stop session closes the guide, and a phone also stops its camera; the server's camera keeps running", () => {
    expect(intentEffects("stop_session", true)).toEqual({ showGuide: false, stopDeviceSession: true });
    expect(intentEffects("stop_session", false)).toEqual({ showGuide: false, stopDeviceSession: false });
  });
  it("gives the same guide change in both modes for every command", () => {
    for (const intent of ["show_preview", "hide_preview", "start_session", "stop_session"]) {
      expect(intentEffects(intent, true).showGuide).toBe(intentEffects(intent, false).showGuide);
    }
  });
  it("changes nothing for the fall answers, for talk that was not a command, or for nothing", () => {
    for (const intent of ["ok", "help", "something else", null]) expect(intentEffects(intent, true)).toEqual({});
  });
});

describe("micErrorText", () => {
  it("explains a refused or missing microphone", () => {
    expect(micErrorText(new DOMException("denied", "NotAllowedError"))).toMatch(/blocked/i);
    expect(micErrorText(new DOMException("none", "NotFoundError"))).toMatch(/no microphone/i);
    expect(micErrorText(new Error("x"))).toMatch(/unavailable/i);
  });
});

describe("pickAudioMime", () => {
  it("takes the first supported type in the preference order", () => {
    expect(pickAudioMime(() => true)).toBe("audio/webm;codecs=opus");
    expect(pickAudioMime((t) => t === "audio/webm")).toBe("audio/webm");
  });
  it("picks mp4 on iOS Safari, which has no webm recorder", () => {
    expect(pickAudioMime((t) => t === "audio/mp4")).toBe("audio/mp4");
  });
  it("falls back to aac, then to the browser's own default (empty)", () => {
    expect(pickAudioMime((t) => t === "audio/aac")).toBe("audio/aac");
    expect(pickAudioMime(() => false)).toBe("");
    expect(AUDIO_MIME_PREFERENCE).toHaveLength(4);
  });
});

describe("rms", () => {
  it("is the root mean square", () => {
    expect(rms([])).toBe(0);
    expect(rms([0.5, -0.5, 0.5, -0.5])).toBeCloseTo(0.5);
  });
});

/** Feeds a constant level for `ms` in 50 ms steps starting at `t`; returns the events and the new time. */
function run(vad: Vad, level: number, ms: number, t: number) {
  const events: string[] = [];
  for (let i = 0; i < ms / 50; i++) {
    t += 50;
    const e = vad.feed(level, t);
    if (e) events.push(e);
  }
  return { events, t };
}

describe("Vad", () => {
  it("ends an utterance after the trailing silence", () => {
    const vad = new Vad();
    let r = run(vad, 0.001, 500, 0);
    expect(r.events).toEqual([]);
    r = run(vad, 0.1, 600, r.t);
    expect(r.events).toEqual(["start"]);
    r = run(vad, 0.001, 1000, r.t);
    expect(r.events).toEqual(["end"]);
    expect(vad.speaking).toBe(false);
  });
  it("cancels a click that is too short to be speech", () => {
    const vad = new Vad();
    const a = run(vad, 0.1, 50, 0);
    const b = run(vad, 0.001, 1000, a.t);
    expect([...a.events, ...b.events]).toEqual(["start", "cancel"]);
  });
  it("cuts a sound that never stops", () => {
    const vad = new Vad();
    const r = run(vad, 0.1, 9000, 0);
    expect(r.events).toEqual(["start", "force", "start"]);
  });
  it("keeps a sentence together across a short pause", () => {
    const vad = new Vad();
    const a = run(vad, 0.1, 400, 0);
    const b = run(vad, 0.001, 400, a.t);    // a pause shorter than the trailing silence
    const c = run(vad, 0.1, 400, b.t);
    const d = run(vad, 0.001, 1000, c.t);
    expect([...a.events, ...b.events, ...c.events, ...d.events]).toEqual(["start", "end"]);
  });
  it("follows the room: steady noise raises the threshold, speech still stands out", () => {
    const vad = new Vad();
    run(vad, 0.01, 3000, 0);
    expect(vad.threshold).toBeGreaterThan(0.02);
    const r = run(vad, 0.015, 500, 3000);    // louder than quiet, but not above the room's noise
    expect(r.events).toEqual([]);
    expect(run(vad, 0.3, 300, r.t).events).toEqual(["start"]);
  });
});

describe("holdQuiet", () => {
  it("is quiet for the stated time and then not", async () => {
    expect(isQuiet()).toBe(false);
    holdQuiet(30);
    expect(isQuiet()).toBe(true);
    await new Promise((r) => setTimeout(r, 50));
    expect(isQuiet()).toBe(false);
  });
});
