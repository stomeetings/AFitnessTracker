// Puts the on-device pose model's files into public/ so a bundled (offline) build ships them inside the APK:
//   node scripts/copy-assets.mjs           copy the MediaPipe WASM  -> public/mediapipe/wasm and the Lite model -> public/models
//   node scripts/copy-assets.mjs --clean   delete them again (a normal web build must not carry ~20 MB it does not use)
// Run by scripts/build_and_serve_apk.py --bundled; the copies are git-ignored.
import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const wasmFrom = join(root, "node_modules", "@mediapipe", "tasks-vision", "wasm");
const wasmTo = join(root, "public", "mediapipe", "wasm");
const modelFrom = join(root, "..", "backend", "models", "pose_landmarker_lite.task");
const modelTo = join(root, "public", "models", "pose_landmarker_lite.task");

if (process.argv.includes("--clean")) {
  rmSync(join(root, "public", "mediapipe"), { recursive: true, force: true });
  rmSync(join(root, "public", "models"), { recursive: true, force: true });
  console.log("removed public/mediapipe and public/models");
  process.exit(0);
}

if (!existsSync(wasmFrom)) { console.error(`missing ${wasmFrom}: run npm install`); process.exit(1); }
if (!existsSync(modelFrom)) { console.error(`missing ${modelFrom}: run backend/scripts/download_models.py`); process.exit(1); }
mkdirSync(wasmTo, { recursive: true });
cpSync(wasmFrom, wasmTo, { recursive: true });
mkdirSync(dirname(modelTo), { recursive: true });
cpSync(modelFrom, modelTo);
console.log("copied the MediaPipe WASM and the Lite model into public/");
