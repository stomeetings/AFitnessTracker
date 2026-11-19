import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App";
import { requestPersistence } from "./store";
import { startSync } from "./sync";

void requestPersistence();      // ask the browser/WebView not to evict this app's local storage (the saved sessions)
startSync();                    // upload saved sessions whenever the server can be reached (sync/syncQueue.ts): now, on `online`, on app resume

// Production builds only: in dev a worker would cache stale modules. Needs HTTPS (or localhost) to register.
if (import.meta.env.PROD && "serviceWorker" in navigator) {
  window.addEventListener("load", () => { navigator.serviceWorker.register("/sw.js").catch(() => {}); });
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
