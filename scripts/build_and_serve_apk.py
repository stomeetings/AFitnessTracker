"""Build the Android wrapper APK for the current tunnel URL and offer it to the phone by QR code (or USB).

The APK is a Capacitor WebView that opens the tunnel URL: the app itself still comes from the backend, so nothing about pose
tracking changes. Quick tunnels get a new address every run, so run this again whenever the tunnel restarts.

    backend/.venv/Scripts/python scripts/build_and_serve_apk.py                       # asks for the URL
    backend/.venv/Scripts/python scripts/build_and_serve_apk.py --url https://x.trycloudflare.com
    backend/.venv/Scripts/python scripts/build_and_serve_apk.py --serve-only          # last APK, just the QR code
    backend/.venv/Scripts/python scripts/build_and_serve_apk.py --adb                 # install over USB instead

URL order: --url, then TUNNEL_URL in the environment or backend/.env, then a prompt.
QR code in the terminal needs `qrcode` (backend/.venv/Scripts/python -m pip install qrcode); without it the link is printed.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
ANDROID = FRONTEND / "android"
CONFIG = FRONTEND / "capacitor.config.json"
MANIFEST = ANDROID / "app" / "src" / "main" / "AndroidManifest.xml"
APK = ANDROID / "app" / "build" / "outputs" / "apk" / "debug" / "app-debug.apk"
SHARE = ANDROID / "app" / "build" / "share"            # only the APK is served from here (android/app/build is git-ignored)
SHARED_NAME = "AI-Fitness-Tracker.apk"
WINDOWS = os.name == "nt"

# What the WebView needs for getUserMedia (camera, microphone for the Whisper voice commands). Capacitor's WebChromeClient turns
# the page's camera/microphone request into the Android permission dialog, but only for permissions declared here.
PERMISSIONS = [
    '<uses-permission android:name="android.permission.INTERNET" />',
    '<uses-permission android:name="android.permission.CAMERA" />',
    '<uses-permission android:name="android.permission.RECORD_AUDIO" />',
    '<uses-permission android:name="android.permission.MODIFY_AUDIO_SETTINGS" />',
    '<uses-feature android:name="android.hardware.camera" android:required="false" />',
]


def run(cmd: str, cwd: Path) -> None:
    print(f"\n> {cmd}   (in {cwd.relative_to(ROOT)})")
    subprocess.run(cmd, cwd=cwd, check=True, shell=True)     # shell: npx/npm/gradlew are .cmd/.bat files on Windows


def env_file_value(name: str) -> str:
    path = ROOT / "backend" / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
    return ""


def tunnel_url(arg: str | None) -> str:
    url = (arg or os.environ.get("TUNNEL_URL") or env_file_value("TUNNEL_URL") or input("Tunnel URL (https://...): ")).strip()
    url = url.rstrip("/")
    if not url.startswith(("https://", "http://")):
        sys.exit(f"not a URL: {url!r}")
    if "?" in url or "token" in url.lower():
        sys.exit("give the bare address: the access token must not be built into the APK (type it in the app once)")
    if url.startswith("http://"):
        print("WARNING: an http:// page is not a secure context, so the WebView will refuse the camera. Use the https tunnel URL.")
    return url


def write_config(url: str) -> None:
    config = {
        "appId": "com.aifitness.tracker",
        "appName": "AI Fitness Tracker",
        "webDir": "dist",
        # The WebView opens the tunnel instead of the bundled files. cleartext only matters for an http:// URL.
        "server": {"url": url, "cleartext": url.startswith("http://")},
        "android": {"allowMixedContent": False},
    }
    CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {CONFIG.relative_to(ROOT)} -> {url}")


def patch_manifest() -> None:
    text = MANIFEST.read_text(encoding="utf-8")
    missing = [line for line in PERMISSIONS if line.split('"')[1] not in text]
    if missing:
        text = text.replace("</manifest>", "".join(f"    {line}\n" for line in missing) + "</manifest>")
        MANIFEST.write_text(text, encoding="utf-8")
    print(f"AndroidManifest.xml: added {len(missing)} permission line(s)")


def patch_wrapper() -> None:
    """Gradle download: the smaller -bin package (no docs/sources, not needed on the command line) and a 2-minute read timeout
    instead of 10 s, which a slow or briefly stalled connection exceeds."""
    path = ANDROID / "gradle" / "wrapper" / "gradle-wrapper.properties"
    text = path.read_text(encoding="utf-8")
    new = re.sub(r"networkTimeout=\d+", "networkTimeout=120000", text.replace("-all.zip", "-bin.zip"))
    if new != text:
        path.write_text(new, encoding="utf-8")
        print("gradle-wrapper.properties: -bin download, 120 s timeout")


def build() -> None:
    for var in ("JAVA_HOME", "ANDROID_HOME"):
        if not os.environ.get(var):
            sys.exit(f"{var} is not set: see NATIVE_DEMO_PLAN.md, step 1 (open a NEW terminal after setting it)")
    if WINDOWS and "trustStoreType" not in os.environ.get("GRADLE_OPTS", ""):
        # Trust what Windows trusts (still full certificate checks). The wrapper that downloads Gradle reads only GRADLE_OPTS, not
        # gradle.properties; the build JVM inherits it too.
        os.environ["GRADLE_OPTS"] = (os.environ.get("GRADLE_OPTS", "") + " -Djavax.net.ssl.trustStoreType=Windows-ROOT").strip()
    if not (FRONTEND / "dist" / "index.html").is_file():
        run("npm run build", FRONTEND)                       # Capacitor insists on webDir existing, even with server.url
    if not ANDROID.is_dir():
        run("npx cap add android", FRONTEND)
    patch_manifest()
    patch_wrapper()
    run("npx cap sync android", FRONTEND)
    # full path: some shells (Claude Code's included) do not look for programs in the current folder
    run(f'"{ANDROID / ("gradlew.bat" if WINDOWS else "gradlew")}" assembleDebug', ANDROID)
    print(f"\nbuilt {APK.relative_to(ROOT)} ({APK.stat().st_size / 1e6:.1f} MB)")


def lan_ip() -> str:
    """The address of the adapter that routes to the LAN. A UDP connect sends nothing; it only picks the route."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def print_qr(link: str) -> None:
    try:
        import qrcode
    except ImportError:
        print("(no QR code: backend/.venv/Scripts/python -m pip install qrcode)")
        return
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    qr = qrcode.QRCode(border=2)
    qr.add_data(link)
    qr.print_ascii(invert=True)


def serve(ip: str, port: int) -> None:
    SHARE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(APK, SHARE / SHARED_NAME)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SHARE))
    handler.func.extensions_map[".apk"] = "application/vnd.android.package-archive"   # so Chrome offers to install it
    link = f"http://{ip}:{port}/{SHARED_NAME}"
    print_qr(link)
    print(f"\nScan with the phone (same Wi-Fi):  {link}\nWrong address? Pass --ip <the Wi-Fi adapter's IPv4 from ipconfig>.")
    print("Ctrl+C stops the server once the APK is installed.\n")
    with http.server.ThreadingHTTPServer(("0.0.0.0", port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("stopped")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", help="the https tunnel address (default: TUNNEL_URL, else asks)")
    ap.add_argument("--serve-only", action="store_true", help="skip the build, serve the last APK")
    ap.add_argument("--adb", action="store_true", help="install over USB (adb install -r) instead of serving a QR code")
    ap.add_argument("--ip", default=None, help="LAN address to put in the QR code (default: detected)")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    if not args.serve_only:
        write_config(tunnel_url(args.url))
        build()
    if not APK.is_file():
        sys.exit(f"no APK at {APK.relative_to(ROOT)}: run without --serve-only first")
    if args.adb:
        run(f'adb install -r "{APK}"', ROOT)
    else:
        serve(args.ip or lan_ip(), args.port)


if __name__ == "__main__":
    main()
