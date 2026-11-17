import { describe, expect, it } from "vitest";
import { LANDMARK_JOINTS, MAX_LOCAL_FPS, ON_DEVICE, bboxFromKeypoints, canRunInference, defaultInference, isMobileDevice, keypointMessage, landmarksToKeypoints } from "./logic";
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

describe("on-device pose helpers", () => {
  const lm = (x: number, y: number, visibility?: number) => ({ x, y, visibility });
  const person = () => {                                              // 33 landmarks; only the mapped ones matter
    const all = Array.from({ length: 33 }, () => lm(0, 0, 0));
    all[11] = lm(0.4, 0.3, 0.98); all[12] = lm(0.6, 0.3, 0.97); all[23] = lm(0.42, 0.55, 0.96); all[24] = lm(0.58, 0.55, 0.9);
    all[25] = lm(0.42, 0.75, 0.2); all[0] = lm(0.5, 0.15);             // a knee that is hard to see; a nose with no visibility
    return all;
  };

  it("maps the 33 landmarks onto the 17 joint names, in pixels of the frame", () => {
    const kp = landmarksToKeypoints(person(), 1280, 720);
    expect(Object.keys(kp)).toHaveLength(17);
    expect(kp.l_shoulder).toEqual([512, 216, 0.98]);                   // 0.4 * 1280, 0.3 * 720
    expect(kp.r_hip).toEqual([742.4, 396, 0.9]);
    expect(kp.l_knee[2]).toBe(0.2);
    expect(kp.nose[2]).toBe(0);                                         // no visibility = no confidence
    expect(kp.l_shoulder).not.toBe(kp.r_shoulder);
    expect(Object.values(LANDMARK_JOINTS)).toContain("r_foot_index");
  });
  it("copes with fewer landmarks than expected", () => {
    expect(Object.keys(landmarksToKeypoints([lm(0.5, 0.5, 1)], 100, 100))).toEqual(["nose"]);
    expect(landmarksToKeypoints([], 100, 100)).toEqual({});
  });

  it("outlines the confident joints, and nothing when there are too few", () => {
    const box = bboxFromKeypoints(landmarksToKeypoints(person(), 1000, 1000));
    expect(box).not.toBeNull();
    const [x, y, w, h] = box!;
    expect(x).toBeLessThan(400); expect(x + w).toBeGreaterThan(600);   // spans the shoulders and hips (the weak knee is left out)
    expect(y).toBeLessThan(300); expect(y + h).toBeGreaterThan(550);
    expect(bboxFromKeypoints({ a: [1, 2, 0.9], b: [3, 4, 0.9] })).toBeNull();
  });

  it("writes the message the server reads: type first, rounded time, the joints", () => {
    const text = keypointMessage(5, 1234.56, 640, 480, { l_hip: [1, 2, 0.5] });
    expect(text.startsWith('{"type":"kp"')).toBe(true);                // the server's prefix check
    expect(JSON.parse(text)).toEqual({ type: "kp", seq: 5, t: 1234.6, w: 640, h: 480, kp: { l_hip: [1, 2, 0.5] } });
    expect(text.length).toBeLessThan(4096);                            // the server refuses longer text messages
    expect(keypointMessage(1, 0, 1280, 720, landmarksToKeypoints(person(), 1280, 720)).length).toBeLessThan(1000);
  });

  it("caps the on-device loop at 30 per second", () => {
    expect(canRunInference(33.4, 0)).toBe(true);                       // a 30 fps camera's next frame
    expect(canRunInference(16.7, 0)).toBe(false);                      // a 60 Hz display tick in between
    expect(canRunInference(1000, 969)).toBe(true);
    expect(MAX_LOCAL_FPS).toBe(30);
  });

  it("recognises phones and tablets, and defaults them to on-device pose", () => {
    const iphone = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1";
    const android = "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 Chrome/124.0 Mobile Safari/537.36";
    const ipadOs = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.0 Safari/605.1.15";
    const desktop = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36";
    expect(isMobileDevice(iphone, 5)).toBe(true);
    expect(isMobileDevice(android, 5)).toBe(true);
    expect(isMobileDevice(ipadOs, 5)).toBe(true);                      // iPadOS pretends to be a Mac, but it has a touch screen
    expect(isMobileDevice(ipadOs, 0)).toBe(false);                     // a real Mac
    expect(isMobileDevice(desktop, 0)).toBe(false);
    expect(defaultInference(true)).toBe("device");
    expect(defaultInference(false)).toBe("server");
  });

  it("lists the on-device model beside the server's in the badge's cycle", () => {
    expect(engineLabel(ON_DEVICE)).toBe("On-device MediaPipe");
    expect(nextEngine(ON_DEVICE, [ON_DEVICE, "mediapipe", "yolo"])).toBe("mediapipe");
    expect(nextEngine("yolo", [ON_DEVICE, "mediapipe", "yolo"])).toBe(ON_DEVICE);
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
