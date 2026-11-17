import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Vite refuses requests whose Host header is not listed here ("Blocked request. This host is not allowed"), which is
// what a tunnel's public name would otherwise hit. A leading dot allows every subdomain.
const allowedHosts = [".trycloudflare.com", ".ngrok-free.app", ".ngrok-free.dev", ".ngrok.app", ".ngrok.io", ".loca.lt"];

// The Host header is deliberately NOT rewritten (no changeOrigin): the backend compares the browser's Origin with it.
const proxy = {
  "/api": "http://127.0.0.1:8000",
  "/models": "http://127.0.0.1:8000",      // the pose model for on-device inference
  "/ws": { target: "ws://127.0.0.1:8000", ws: true },
};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    allowedHosts,
    proxy,
  },
  preview: { allowedHosts, proxy },            // `npm run preview` serves the production build (service worker included) the same way
  test: { include: ["src/**/*.test.ts"] },
});
