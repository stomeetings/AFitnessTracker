import { describe, expect, it } from "vitest";
import { ACK_TIMEOUT_MS, MAX_QUALITY, MIN_FRAME_GAP_MS, MIN_QUALITY, SKELETON, canSendFrame, captureSize, drawPose, engineLabel, extractToken, fitContain, isLocalHost, nextEngine, nextQuality, packFrame, reconnectDelay, type Pose } from "./logic";

const bytes = (b: ArrayBuffer) => Array.from(new Uint8Array(b));

describe("packFrame", () => {
  it("is [u32 seq][f64 ms] big-endian then the JPEG, as struct '>Id' reads it", () => {
    // Python: struct.pack(">Id", 1, 1.5).hex() == "000000013ff8000000000000"
    const out = packFrame(1, 1.5, new Uint8Array([0xff, 0xd8, 0xff]).buffer);
    expect(bytes(out)).toEqual([0, 0, 0, 1, 0x3f, 0xf8, 0, 0, 0, 0, 0, 0, 0xff, 0xd8, 0xff]);
  });
  it("keeps large frame numbers unsigned", () => {
    expect(bytes(packFrame(0xffffffff, 0, new ArrayBuffer(0))).slice(0, 4)).toEqual([255, 255, 255, 255]);
  });
});

describe("captureSize", () => {
  it("limits the long side and keeps the aspect, portrait or landscape", () => {
    expect(captureSize(1280, 720, 480)).toEqual({ w: 480, h: 270 });
    expect(captureSize(720, 1280, 480)).toEqual({ w: 270, h: 480 });
  });
  it("never scales up", () => {
    expect(captureSize(320, 240, 480)).toEqual({ w: 320, h: 240 });
  });
});

describe("nextQuality", () => {
  it("steps down on a slow round trip, down to the floor", () => {
    let q = MAX_QUALITY;
    for (let i = 0; i < 10; i++) q = nextQuality(q, 400);
    expect(q).toEqual(MIN_QUALITY);
  });
  it("steps back up on a fast one, up to the ceiling", () => {
    let q = MIN_QUALITY;
    for (let i = 0; i < 20; i++) q = nextQuality(q, 50);
    expect(q).toEqual(MAX_QUALITY);
  });
  it("holds in between", () => {
    expect(nextQuality({ longSide: 400, quality: 0.5 }, 180)).toEqual({ longSide: 400, quality: 0.5 });
  });
});

describe("canSendFrame (the send throttle)", () => {
  it("never starts frames faster than 20 per second, even when the last one was acknowledged", () => {
    expect(MIN_FRAME_GAP_MS).toBe(50);
    expect(canSendFrame(1030, 1000, false)).toBe(false);         // 30 ms after the last frame: skip
    expect(canSendFrame(1050, 1000, false)).toBe(true);
  });
  it("waits for the server's answer before sending the next frame", () => {
    expect(canSendFrame(1500, 1000, true)).toBe(false);
    expect(canSendFrame(1000 + ACK_TIMEOUT_MS, 1000, true)).toBe(false);
  });
  it("sends again when an answer is overdue (assumed lost), so the stream cannot stall", () => {
    expect(canSendFrame(1000 + ACK_TIMEOUT_MS + 1, 1000, true)).toBe(true);
  });
  it("allows the very first frame", () => {
    expect(canSendFrame(10_000, 0, false)).toBe(true);
  });
});

describe("engineLabel for the Lite model", () => {
  it("names it", () => expect(engineLabel("mediapipe_lite")).toBe("MediaPipe Lite"));
});

describe("reconnectDelay", () => {
  it("doubles and caps at 10 s", () => {
    expect([0, 1, 2, 3].map(reconnectDelay)).toEqual([500, 1000, 2000, 4000]);
    expect(reconnectDelay(20)).toBe(10_000);
  });
});

describe("extractToken", () => {
  it("takes the token out of the address and keeps other parameters", () => {
    expect(extractToken("?token=abc")).toEqual({ token: "abc", rest: "" });
    expect(extractToken("?a=1&token=x%20y&b=2")).toEqual({ token: "x y", rest: "?a=1&b=2" });
  });
  it("leaves an address without one alone", () => {
    expect(extractToken("?a=1")).toEqual({ token: null, rest: "?a=1" });
    expect(extractToken("")).toEqual({ token: null, rest: "" });
  });
  it("ignores an empty token", () => {
    expect(extractToken("?token=").token).toBeNull();
  });
});

describe("engineLabel / nextEngine", () => {
  it("names the models for the badge", () => {
    expect(engineLabel("yolo")).toBe("YOLO");
    expect(engineLabel("mediapipe")).toBe("MediaPipe");
    expect(engineLabel("other")).toBe("other");
  });
  it("picks the model a tap switches to, wrapping round", () => {
    expect(nextEngine("mediapipe", ["mediapipe", "yolo"])).toBe("yolo");
    expect(nextEngine("yolo", ["mediapipe", "yolo"])).toBe("mediapipe");
    expect(nextEngine(null, ["mediapipe", "yolo"])).toBe("mediapipe");
    expect(nextEngine("a", ["a", "b", "c"])).toBe("b");
  });
  it("offers nothing when there is nothing to switch to", () => {
    expect(nextEngine("yolo", [])).toBeNull();
    expect(nextEngine("yolo", ["yolo"])).toBeNull();
  });
});

describe("isLocalHost", () => {
  it("is true only for the machine itself", () => {
    expect(isLocalHost("localhost")).toBe(true);
    expect(isLocalHost("192.168.1.5")).toBe(false);
    expect(isLocalHost("abc.trycloudflare.com")).toBe(false);
  });
});

describe("fitContain", () => {
  it("letterboxes a portrait frame in a landscape box", () => {
    const f = fitContain(480, 640, 800, 400);
    expect(f.scale).toBeCloseTo(0.625);
    expect(f.w).toBeCloseTo(300);
    expect(f.x).toBeCloseTo(250);
    expect(f.y).toBeCloseTo(0);
  });
});

function recorder() {
  const calls: string[] = [];
  const ctx = {
    strokeStyle: "", fillStyle: "", lineWidth: 1,
    clearRect: () => calls.push("clear"),
    beginPath: () => {},
    moveTo: (x: number, y: number) => calls.push(`move ${x},${y}`),
    lineTo: () => calls.push("line"),
    stroke: () => {},
    arc: (x: number, y: number) => calls.push(`dot ${x},${y}`),
    fill: () => {},
    strokeRect: (x: number, y: number, w: number, h: number) => calls.push(`box ${x},${y},${w},${h}`),
  };
  return { ctx, calls };
}

describe("drawPose", () => {
  const pose: Pose = { kp: { l_shoulder: [100, 100, 0.9], r_shoulder: [200, 100, 0.9], l_hip: [100, 300, 0.2] }, bbox: [50, 50, 200, 300] };

  it("maps frame pixels onto the letterboxed preview", () => {
    const { ctx, calls } = recorder();
    drawPose(ctx, pose, { w: 640, h: 480 }, { w: 1280, h: 960 });      // exactly 2x
    expect(calls).toContain("box 100,100,400,600");
    expect(calls).toContain("dot 200,200");
  });
  it("skips low-confidence joints and any bone that touches one", () => {
    const { ctx, calls } = recorder();
    drawPose(ctx, pose, { w: 640, h: 480 }, { w: 640, h: 480 });
    expect(calls.filter((c) => c === "line")).toHaveLength(1);           // only shoulder-shoulder: l_hip is below 0.5
    expect(calls.filter((c) => c.startsWith("dot"))).toHaveLength(2);
  });
  it("only clears when there is no pose", () => {
    const { ctx, calls } = recorder();
    drawPose(ctx, null, { w: 640, h: 480 }, { w: 640, h: 480 });
    expect(calls).toEqual(["clear"]);
  });
  it("turns red for a confirmed fall", () => {
    const { ctx } = recorder();
    drawPose(ctx, pose, { w: 640, h: 480 }, { w: 640, h: 480 }, "confirmed");
    expect(ctx.strokeStyle).toBe("#ef4444");
  });
});

it("uses the same bones as the backend", () => {
  expect(SKELETON).toHaveLength(16);
});
