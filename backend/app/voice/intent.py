"""Turns what the patient said (the text Whisper returned) into an intent. Pure text logic, no dependencies.

Deliberately asymmetric: a false alarm is acceptable, a wrongly cancelled emergency is not.
- any help / distress / negative word in the sentence -> "help"   (escalate at once; checked first)
- a clear multi-word affirmative                      -> "ok"     (cancel the countdown)
- anything else (bare "okay", "yes", noise)           -> None     (the countdown just continues)

"Help" is matched loosely, because a short soft word is often transcribed as a near-miss: any word containing "elp"
(help, helping, helpi, aelp, yelp ...) or one letter away from "help" (halp, hep, hell, held ...) counts, as do the
Urdu / Hindi words bachao and madad. The leading "I" of an affirmative is optional ("am fine", "i'm ok", "all good",
"doing fine") and a few predictable mishearings of "fine" / "okay" ("five", "kind", "fight", "oky", "ache") are
accepted, but only right after "am" / "i am" / "i'm". A single word on its own never counts as an affirmative, and
any negative wins.
"""
from __future__ import annotations

import re

# "no" / "not" / "bad" also catch "I am not ok". bachao / madad are Urdu / Hindi for "save me" / "help".
_HELP = re.compile(r"\b(emergency|ambulance|hurt|hurts|pain|cannot|stuck|fall|fallen|fell|sick|dizzy|no|not|bad"
                   r"|bachao|bachaao|bacchao|bhachao|madad|madat)\b")

_GOOD = r"(okay|ok|fine|alright|all right|good|well)"      # what the patient means
_HEARD = r"(oky|okey|oke|ache|five|kind|fight)"             # predictable mishearings; "ache" is "okay" in some accents
_OK = re.compile(rf"\b(am ({_GOOD}|{_HEARD})|all good|doing {_GOOD})\b")   # "i am ok" contains "am ok"


def _normalise(text: str) -> str:
    s = text.lower().replace("’", "'")
    s = re.sub(r"\bi'?m\b", "i am", s)          # i'm / im
    s = re.sub(r"\bcan'?t\b", "cannot", s)      # can't / cant
    s = re.sub(r"n't\b", " not", s)             # don't / isn't / won't: any negative must be seen as "not"
    s = re.sub(r"[^a-z ]+", " ", s)             # punctuation or digits
    return re.sub(r"\s+", " ", s).strip()


def _within_one_edit(a: str, b: str) -> bool:
    """True when a and b differ by at most one inserted, deleted or substituted letter."""
    if abs(len(a) - len(b)) > 1:
        return False
    i = j = edits = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(a) > len(b):
            i += 1
        elif len(a) < len(b):
            j += 1
        else:
            i += 1
            j += 1
    return edits + (len(a) - i) + (len(b) - j) <= 1


def _sounds_like_help(word: str) -> bool:
    """help, helps, helping, helpi, aelp, elp, yelp, halp, hep, hell, held ...: the word contains "elp", or is one letter
    away from "help". "hello" is two letters away, so it is not matched."""
    return "elp" in word or (3 <= len(word) <= 5 and _within_one_edit(word, "help"))


def classify(text: str) -> str | None:
    s = _normalise(text)
    if _HELP.search(s) or any(_sounds_like_help(w) for w in s.split()):   # checked first: "no I am five" escalates
        return "help"
    if _OK.search(s):
        return "ok"
    return None


parse_intent = classify         # the name the plans and notes use


# ---- session commands (spoken while NO fall check is open) ------------------------------------------------------
# "start my session" / "stop the session" / "show preview" / "hide preview", with a verb and a noun anywhere in the
# sentence, and one letter of slack on both for Whisper's near-misses ("sesion", "pre view", "shoe preview").
_SESSION_ON = ("start", "begin", "resume", "new")
_SESSION_OFF = ("stop", "end", "finish", "quit", "pause")
_PREVIEW_ON = ("show", "display", "open", "start", "view", "enable")
_PREVIEW_OFF = ("hide", "close", "stop", "end", "disable")


def _is_verb(word: str, verbs: tuple[str, ...]) -> bool:
    return any(word == v or (len(word) >= 4 and len(v) >= 4 and _within_one_edit(word, v)) for v in verbs)


def _is_noun(word: str, noun: str) -> bool:
    return word in (noun, noun + "s") or _within_one_edit(word, noun)


def classify_command(text: str) -> str | None:
    """"start_session" | "stop_session" | "show_preview" | "hide_preview" | None. Any negative ("don't stop the session"),
    both nouns, or both directions at once is ambiguous and gives None: a command must be unmistakable."""
    s = re.sub(r"\bpre view", "preview", _normalise(text))
    words = s.split()
    if "not" in words:
        return None
    if re.search(r"\b(let s|lets|let us) go\b", s) and len(words) <= 6:      # "let's go", "lets go", "okay let us go"
        return "start_session"
    session = any(_is_noun(w, "session") for w in words)
    preview = any(_is_noun(w, "preview") for w in words)
    if session == preview:                       # neither, or both
        return None
    on_verbs, off_verbs = (_SESSION_ON, _SESSION_OFF) if session else (_PREVIEW_ON, _PREVIEW_OFF)
    on = any(_is_verb(w, on_verbs) for w in words)
    off = any(_is_verb(w, off_verbs) for w in words)
    if on == off:
        return None
    return ("start_session" if on else "stop_session") if session else ("show_preview" if on else "hide_preview")
