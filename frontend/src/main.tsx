import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App";

// Production builds only: in dev a worker would cache stale modules. Needs HTTPS (or localhost) to register.
if (import.meta.env.PROD && "serviceWorker" in navigator) {
  window.addEventListener("load", () => { navigator.serviceWorker.register("/sw.js").catch(() => {}); });
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
