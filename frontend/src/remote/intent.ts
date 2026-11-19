/** Client-side twin of backend/app/voice/intent.py `classify_command`, for speech recognised on the device (no server round trip,
 *  so it works offline). Keep the two in step. Session commands only: the fall-check answers ("I'm fine", "help") stay with the
 *  fall dialog's own listener. */

const normalise = (text: string) =>
  text.toLowerCase().replace(/’/g, "'")
    .replace(/\bi'?m\b/g, "i am").replace(/\bcan'?t\b/g, "cannot").replace(/n't\b/g, " not")
    .replace(/[^a-z ]+/g, " ").replace(/\s+/g, " ").trim();

function withinOneEdit(a: string, b: string): boolean {
  if (Math.abs(a.length - b.length) > 1) return false;
  let i = 0, j = 0, edits = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) { i++; j++; continue; }
    if (++edits > 1) return false;
    if (a.length > b.length) i++; else if (a.length < b.length) j++; else { i++; j++; }
  }
  return edits + (a.length - i) + (b.length - j) <= 1;
}

const SESSION_ON = ["start", "begin", "resume", "new"], SESSION_OFF = ["stop", "end", "finish", "quit", "pause"];
const PREVIEW_ON = ["show", "display", "open", "start", "view", "enable"], PREVIEW_OFF = ["hide", "close", "stop", "end", "disable"];
const isVerb = (w: string, vs: string[]) => vs.some((v) => w === v || (w.length >= 4 && v.length >= 4 && withinOneEdit(w, v)));
const isNoun = (w: string, n: string) => w === n || w === n + "s" || withinOneEdit(w, n);

export function classifyCommand(text: string): string | null {
  const s = normalise(text).replace(/\bpre view/g, "preview");
  const words = s.split(" ").filter(Boolean);
  if (words.includes("not")) return null;
  if (/^(please )?(pause|hold( on)?)( please)?$/.test(s)) return "pause_session";
  if (/^(please )?resume( please)?$/.test(s)) return "resume_session";
  if (/\b(let s|lets|let us) go\b/.test(s) && words.length <= 6) return "start_session";
  const session = words.some((w) => isNoun(w, "session")), preview = words.some((w) => isNoun(w, "preview"));
  if (session === preview) return null;
  const [onV, offV] = session ? [SESSION_ON, SESSION_OFF] : [PREVIEW_ON, PREVIEW_OFF];
  const on = words.some((w) => isVerb(w, onV)), off = words.some((w) => isVerb(w, offV));
  if (on === off) return null;
  return session ? (on ? "start_session" : "stop_session") : (on ? "show_preview" : "hide_preview");
}
