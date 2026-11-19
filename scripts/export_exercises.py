"""Writes the exercise table (backend/app/pipeline/rep_counter.EXERCISES, the master) to frontend/src/shared/exercises.json, which
the on-device TypeScript engine (frontend/src/engine/) and the exercise picker import, so the two can never differ.

    backend/.venv/Scripts/python scripts/export_exercises.py            # rewrite the JSON
    backend/.venv/Scripts/python scripts/export_exercises.py --check    # exit 1 if the JSON is out of date (also a test)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.pipeline.rep_counter import EXERCISES  # noqa: E402

TARGET = ROOT / "frontend" / "src" / "shared" / "exercises.json"


def render() -> str:
    """One exercise per line, in the table's order (the picker shows them in this order)."""
    rows = [f"  {json.dumps(name)}: {json.dumps(cfg)}" for name, cfg in EXERCISES.items()]
    return "{\n" + ",\n".join(rows) + "\n}\n"


def current_matches() -> bool:
    """True when the committed JSON holds exactly the Python table (same exercises, same order, same numbers)."""
    return TARGET.is_file() and json.loads(TARGET.read_text(encoding="utf-8")) == EXERCISES \
        and list(json.loads(TARGET.read_text(encoding="utf-8"))) == list(EXERCISES)


if __name__ == "__main__":
    if "--check" in sys.argv:
        ok = current_matches()
        print("exercises.json is up to date" if ok else "exercises.json is OUT OF DATE: run scripts/export_exercises.py")
        sys.exit(0 if ok else 1)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(render(), encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)} ({len(EXERCISES)} exercises)")
