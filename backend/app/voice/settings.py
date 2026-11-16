"""Voice settings: the process environment first, then backend/.env. The file is read on every call, so editing it
needs no restart (the next fall check picks the change up)."""
from __future__ import annotations

from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"      # backend/.env (git-ignored)


def get(name: str, default: str | None = None, env_file: Path | str | None = None) -> str | None:
    """The trimmed value of `name`, or `default` when it is unset or blank."""
    import os
    value = os.environ.get(name, "").strip()
    path = Path(env_file) if env_file else ENV_FILE
    if not value and path.is_file():
        try:
            from dotenv import dotenv_values
            value = (dotenv_values(path).get(name) or "").strip()
        except Exception:  # noqa: BLE001  an unreadable .env just means "not set"
            value = ""
    return value or default


def get_float(name: str, default: float, env_file: Path | str | None = None) -> float:
    try:
        return float(get(name, None, env_file))
    except (TypeError, ValueError):             # unset (None) or not a number: use the default
        return default
