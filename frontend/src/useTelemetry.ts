import { useEffect, useState } from "react";

export type Limb = { reps: number; state: string; angle: number | null; last_rom: number; best_rom: number };
export type FallAlert = { state: "IDLE" | "COUNTDOWN" | "ALERTED" | "ACKNOWLEDGED"; seconds_left: number; impact: boolean };
export type Telemetry = {
  camera_index: number;
  camera_backend: string | null;
  camera_error: string | null;
  engine: "mediapipe" | "yolo";
  device: string;
  fps: number;
  latency_ms: number;
  pose_visible: boolean;
  error: string | null;
  exercise: string;
  angles?: Record<string, number>;
  reps?: Record<string, Limb>;
  mobility?: { state: string; steps: number; cadence_spm: number };
  fall?: { status: "NORMAL" | "SUSPECTED" | "CONFIRMED"; tilt_deg: number | null; drop_velocity: number; aspect_ratio: number | null; seconds_in_state: number | null; alert?: FallAlert };
  posture?: string;
};

/** Subscribes to /ws/telemetry; reconnects after 1 s on close. */
export function useTelemetry(): { t: Telemetry | null; online: boolean } {
  const [t, setT] = useState<Telemetry | null>(null);
  const [online, setOnline] = useState(false);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let timer: number | undefined;
    let closed = false;
    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws/telemetry`);
      ws.onopen = () => setOnline(true);
      ws.onmessage = (e) => setT(JSON.parse(e.data));
      ws.onclose = () => {
        setOnline(false);
        if (!closed) timer = window.setTimeout(connect, 1000);
      };
      ws.onerror = () => ws?.close();
    };
    connect();
    return () => {
      closed = true;
      window.clearTimeout(timer);
      ws?.close();
    };
  }, []);

  return { t, online: online && !t?.error };
}
