"""Access control for exposing the server through a tunnel or reverse proxy (cloudflared, ngrok, Caddy ...).

Three small pieces:
- token_ok / FailureLimiter: the shared access token for /ws/session (REMOTE_ACCESS_TOKEN), compared in constant time,
  with a cap on wrong guesses.
- origin_allowed: a browser's Origin must be this site (or one listed in REMOTE_ALLOWED_ORIGINS), so another website
  cannot make a visitor's browser open a session.
- TunnelGuard: a request that arrived through a proxy carries forwarding headers (X-Forwarded-For, Cf-Connecting-Ip ...) that
  the proxy adds and the outside client cannot remove. Such requests may only use the remote-session endpoints and a few
  harmless reads. The desktop routes (the server's own camera stream, camera switching, config, fall buttons, telemetry)
  stay reachable only directly (localhost / LAN), so publishing the app through a tunnel does not publish the laptop camera.
"""
from __future__ import annotations

import hmac
import ipaddress
import time
from collections import deque
from collections.abc import Callable, Iterable
from urllib.parse import urlsplit

from starlette.responses import JSONResponse

TUNNEL_HEADERS = (b"x-forwarded-for", b"x-forwarded-host", b"x-forwarded-proto", b"forwarded", b"cf-connecting-ip")
PUBLIC_API = frozenset({"/api/health", "/api/exercises", "/api/exercise_info", "/ws/session"})


def _public_host(host: str) -> bool:
    """True for a public-looking name such as abc.trycloudflare.com. Local is: localhost, a name without dots (a LAN machine
    name), *.local, and loopback / private / link-local addresses."""
    host = host.strip().lower()
    if host.startswith("["):
        host = host[1:].split("]")[0]
    elif host.count(":") == 1:
        host = host.split(":")[0]
    if host in ("", "localhost") or "." not in host and ":" not in host or host.endswith(".local"):
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (ip.is_private or ip.is_loopback or ip.is_link_local)


def is_tunneled(headers: Iterable[tuple[bytes, bytes]]) -> bool:
    """A forwarding header (added by the proxy, which the outside client cannot strip) or a public Host name."""
    headers = list(headers)
    if any(name.lower() in TUNNEL_HEADERS for name, _ in headers):
        return True
    return any(name.lower() == b"host" and _public_host(value.decode("latin-1")) for name, value in headers)


def token_ok(supplied: str, expected: str) -> bool:
    return hmac.compare_digest(supplied.encode(), expected.encode())


def origin_allowed(origin: str | None, hosts: Iterable[str], extra: Iterable[str] = ()) -> bool:
    """No Origin header = not a browser (a script, a test): allowed, the token is what protects it. A browser's Origin
    must match the Host it connected to (or X-Forwarded-Host behind a proxy), or be listed in `extra` ("*" allows all)."""
    if not origin:
        return True
    extra = [e.strip().rstrip("/").lower() for e in extra if e.strip()]
    if "*" in extra:
        return True
    origin = origin.rstrip("/").lower()
    host = urlsplit(origin).netloc
    return origin in extra or (host != "" and host in {h.strip().lower() for h in hosts if h.strip()})


class FailureLimiter:
    """After `max_fails` wrong tokens inside `window_s` seconds every attempt is refused until the window has passed.
    Global, not per client: behind a tunnel every client looks like the proxy. (So someone guessing can also lock out the
    real user for a minute; that is the price of not having per-client identity.)"""

    def __init__(self, max_fails: int = 10, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.max_fails, self.window_s, self.clock = max_fails, window_s, clock
        self._fails: deque[float] = deque()

    def _trim(self) -> None:
        cutoff = self.clock() - self.window_s
        while self._fails and self._fails[0] <= cutoff:
            self._fails.popleft()

    def blocked(self) -> bool:
        self._trim()
        return len(self._fails) >= self.max_fails

    def record(self) -> None:
        self._trim()
        self._fails.append(self.clock())


class TunnelGuard:
    """ASGI middleware: see the module docstring. Static files (anything outside /api and /ws) are left alone."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket") and is_tunneled(scope.get("headers", ())):
            path = scope["path"]
            if path.startswith(("/api/", "/ws/")) and path not in PUBLIC_API:
                if scope["type"] == "websocket":
                    await receive()                                     # the connect event, then refuse
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    await JSONResponse({"detail": "not available through a tunnel"}, status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)
