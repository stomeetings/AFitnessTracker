import { useCallback, useRef, useState } from "react";
import type { VoiceState } from "../useTelemetry";
import { apiUrl } from "../net";
import { MIC_CONSTRAINTS, isQuiet, micErrorText, startVoiceCapture, type Capture } from "./voice";

/** The microphone side of voice commands, the same for both modes. The caller says how an utterance travels
 *  (`upload`: the session socket on a phone, POST /api/voice on the desktop page); everything else is shared:
 *  capture, status, the last sentence heard. */

export type MicStatus = "off" | "listening" | "processing" | "blocked" | "unavailable";
export type VoiceReply = { ok: boolean; text?: string; intent?: string | null; error?: string; status?: string };

/** Desktop page: one utterance to the backend. Resolves with the backend's answer (errors included). */
export async function uploadToServer(blob: Blob): Promise<VoiceReply> {
  try {
    const r = await fetch(apiUrl("/api/voice"), { method: "POST", headers: { "Content-Type": blob.type || "audio/webm" }, body: blob });
    if (!r.ok) return { ok: false, error: r.status === 403 ? "Voice is not available through a tunnel." : `Voice failed (HTTP ${r.status}).` };
    return (await r.json()) as VoiceReply;
  } catch {
    return { ok: false, error: "The server did not answer." };
  }
}

/** A microphone stream from a tap. Throws a readable message when refused. */
export async function openMic(): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(window.isSecureContext ? "This browser has no microphone API." : "The microphone needs HTTPS (or localhost).");
  }
  try {
    return await navigator.mediaDevices.getUserMedia({ audio: MIC_CONSTRAINTS, video: false });
  } catch (e) {
    throw new Error(micErrorText(e));
  }
}

export function useVoiceInput(upload: (blob: Blob) => Promise<VoiceReply>, paused?: () => boolean) {
  const [mic, setMic] = useState<MicStatus>("off");
  const [micError, setMicError] = useState<string | null>(null);
  const [heard, setHeard] = useState<{ text: string; intent: string | null; seq: number }>({ text: "", intent: null, seq: 0 });
  const stream = useRef<MediaStream | null>(null);
  const capture = useRef<Capture | null>(null);
  const busy = useRef(false);                       // one utterance at a time is out for transcription
  const uploadRef = useRef(upload);
  uploadRef.current = upload;                       // always the latest, so a long-lived capture never uses a stale one
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  const release = useCallback(() => {
    capture.current?.stop();
    capture.current = null;
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    busy.current = false;
  }, []);

  const onUtterance = useCallback((blob: Blob) => {
    if (busy.current) return;
    busy.current = true;
    setMic("processing");
    uploadRef.current(blob).then((reply) => {
      if (reply.status === "unavailable") {          // the server has no speech-to-text key: stop recording
        release();
        setMic("unavailable"); setMicError(reply.error ?? "Voice is not available on the server.");
        return;
      }
      if (reply.ok) { setMicError(null); setHeard((h) => ({ text: reply.text ?? "", intent: reply.intent ?? null, seq: h.seq + 1 })); }
      else setMicError(reply.error ?? "Voice failed.");
    }).catch(() => setMicError("Voice failed."))
      .finally(() => { busy.current = false; setMic((m) => (m === "processing" ? "listening" : m)); });
  }, [release]);

  /** Start listening on `m` (audio only). `why` is the reason there is none (permission refused, no device). */
  const begin = useCallback((m: MediaStream | null, why: string | null) => {
    release();
    if (!m) { setMic("blocked"); setMicError(why); return; }
    // the page's own beeps and spoken prompt are not speech: the detector is paused while they sound
    const cap = startVoiceCapture(m, { onUtterance, paused: () => isQuiet() || !!window.speechSynthesis?.speaking || !!pausedRef.current?.() });
    if (!cap) {
      m.getTracks().forEach((t) => t.stop());
      setMic("unavailable"); setMicError("This browser cannot record audio.");
      return;
    }
    stream.current = m; capture.current = cap;
    setMic("listening"); setMicError(null);
  }, [onUtterance, release]);

  /** Ask for the microphone and start listening (from a tap). */
  const enable = useCallback(async () => {
    try { begin(await openMic(), null); } catch (e) { begin(null, e instanceof Error ? e.message : String(e)); }
  }, [begin]);

  const end = useCallback(() => {
    release();
    setMic("off"); setMicError(null); setHeard({ text: "", intent: null, seq: 0 });
  }, [release]);

  // The same shape the fall dialog already shows for the desktop microphone.
  const voiceState: VoiceState = {
    status: mic === "listening" ? "listening" : mic === "processing" ? "transcribing" : mic === "off" ? "off" : "unavailable",
    transcript: heard.text, intent: heard.intent, error: micError,
  };
  return { mic, micError, heard, begin, enable, end, voiceState };
}

export type VoiceInput = ReturnType<typeof useVoiceInput>;
