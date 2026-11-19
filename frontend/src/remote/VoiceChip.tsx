import { Loader2, Mic, MicOff } from "lucide-react";
import type { MicStatus } from "./useVoiceInput";
import { intentLabel } from "./voice";

export const MIC_TEXT: Record<MicStatus, string> = {
  off: "Mic off", listening: "Listening", processing: "Processing…", blocked: "Mic blocked", unavailable: "Voice unavailable",
};

export const VOICE_HELP = "Say “start my session” or “let’s go”, “stop session”, “show preview”, “hide preview”. During a fall check: “I am fine” or “help”.";

/** The voice status chip, the same in both modes: the microphone state, and under it what was heard — shown BEFORE the
 *  command it triggers is carried out (the page waits a moment, see App.tsx). */
export function VoiceChip({ mic, error, heard, intent = null, offline = false }: { mic: MicStatus; error: string | null; heard: string; intent?: string | null; offline?: boolean }) {
  const bad = mic === "blocked" || mic === "unavailable";
  if (offline) {                                // commands are understood by the server (speech to text), so without it only the buttons work
    return (
      <div className="flex max-w-[60%] flex-col items-end gap-1 text-xs" aria-live="polite">
        <span className="flex items-center gap-2 rounded-full bg-slate-900/80 px-3 py-1 text-amber-300"><MicOff size={12} /> Voice needs a connection</span>
      </div>
    );
  }
  return (
    <div className="flex max-w-[60%] flex-col items-end gap-1 text-xs" aria-live="polite">
      <span className={`flex items-center gap-2 rounded-full bg-slate-900/80 px-3 py-1 ${bad ? "text-amber-300" : ""}`} title={error ?? undefined}>
        {mic === "listening" ? <Mic size={12} className="text-emerald-400" />
          : mic === "processing" ? <Loader2 size={12} className="animate-spin text-emerald-400" /> : <MicOff size={12} />}
        {MIC_TEXT[mic]}
      </span>
      {heard && !bad && (
        <span className="truncate rounded-full bg-slate-900/80 px-3 py-1 text-slate-300">
          Heard: “{heard}” <span className={intent ? "text-emerald-300" : "text-slate-500"}>→ {intentLabel(intent)}</span>
        </span>
      )}
    </div>
  );
}
