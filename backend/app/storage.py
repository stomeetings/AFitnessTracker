"""Long-term store for the workouts the phones upload (Phase 4 of the offline-first plan): SQLite, stdlib only, WAL mode.

    sessions  one row per uploaded camera session (primary key = the uuid the phone made, which is what makes a retried upload harmless)
    events    what happened in it (rep, fall, alert, pause ...), in the order the phone recorded them

`insert_batch` is idempotent: a session whose id is already stored is skipped together with its events (INSERT ... ON CONFLICT DO
NOTHING), so a request that is sent twice because the answer was lost creates no duplicates. One lock serialises writers; every batch
is one transaction, so a crash leaves a session either complete or absent.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "rehab.db"      # data/ is git-ignored

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    device_id   TEXT NOT NULL,
    exercise    TEXT NOT NULL,
    started_at  INTEGER NOT NULL,
    ended_at    INTEGER,
    app_version TEXT NOT NULL,
    summary     TEXT NOT NULL,
    reps_total  INTEGER NOT NULL,
    fall_events INTEGER NOT NULL,
    received_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_started ON sessions (started_at DESC);
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    t          REAL NOT NULL,
    type       TEXT NOT NULL,
    payload    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_session ON events (session_id);
"""


def reps_in(session: dict) -> int:
    """Total reps of an uploaded session: the Start..Stop stretches added up (Start resets the counts), else the final counts."""
    stretches = session.get("workouts") or []
    sources = [w.get("limbs") or {} for w in stretches] or [session.get("limbs") or {}]
    return sum(int(limb.get("reps", 0)) for limbs in sources for limb in limbs.values())


class Storage:
    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)   # autocommit: transactions are explicit
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def insert_batch(self, device_id: str, sessions: list[dict]) -> tuple[list[str], list[str]]:
        """Stores the sessions (each with its "events" list) that are not stored yet. Returns (stored ids, already-known ids).
        All or nothing: an error rolls the whole batch back."""
        stored: list[str] = []
        known: list[str] = []
        now = int(time.time() * 1000)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                for s in sessions:
                    events = s.get("events") or []
                    summary = json.dumps({k: s.get(k) for k in ("limbs", "calibration", "workouts")}, separators=(",", ":"))
                    flagged = sum(1 for e in events if e["type"] in ("fall", "alert"))
                    cur = self._db.execute(
                        "INSERT INTO sessions (id, device_id, exercise, started_at, ended_at, app_version, summary, reps_total, fall_events, received_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO NOTHING",
                        (s["id"], device_id, s["exercise"], s["startedAt"], s.get("endedAt"), s.get("appVersion", ""), summary, reps_in(s), flagged, now))
                    if cur.rowcount == 0:
                        known.append(s["id"])                      # a retry of something already kept: nothing to add
                        continue
                    self._db.executemany(
                        "INSERT INTO events (session_id, t, type, payload) VALUES (?, ?, ?, ?)",
                        [(s["id"], e["t"], e["type"], json.dumps(e.get("payload") or {}, separators=(",", ":"))) for e in events])
                    stored.append(s["id"])
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return stored, known

    def list_sessions(self, limit: int = 50, offset: int = 0) -> dict:
        """Newest first, with the totals the dashboard shows."""
        limit, offset = max(1, min(int(limit), 200)), max(0, int(offset))
        with self._lock:
            rows = self._db.execute(
                "SELECT s.id, s.device_id, s.exercise, s.started_at, s.ended_at, s.app_version, s.summary, s.reps_total, s.fall_events, s.received_at, "
                "(SELECT COUNT(*) FROM events e WHERE e.session_id = s.id) AS event_count "
                "FROM sessions s ORDER BY s.started_at DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
            totals = self._db.execute(
                "SELECT COUNT(*) AS sessions, COALESCE(SUM(reps_total), 0) AS reps, COALESCE(SUM(fall_events > 0), 0) AS fall_sessions FROM sessions").fetchone()
        out = []
        for r in rows:
            row = dict(r)
            row["summary"] = json.loads(row["summary"])
            out.append(row)
        return {"sessions": out, "total": totals["sessions"], "summary": {"sessions": totals["sessions"], "reps": totals["reps"], "fall_sessions": totals["fall_sessions"]},
                "limit": limit, "offset": offset}

    def events_of(self, session_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT t, type, payload FROM events WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()
        return [{"t": r["t"], "type": r["type"], "payload": json.loads(r["payload"])} for r in rows]
