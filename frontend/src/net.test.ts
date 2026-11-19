import { describe, expect, it } from "vitest";
import { cleanBase, joinApi, toWsUrl } from "./net";

describe("backend address helpers", () => {
  it("keeps relative paths when there is no base (browser, same origin)", () => {
    expect(joinApi("", "/api/health")).toBe("/api/health");
    expect(toWsUrl("", "/ws/session", { protocol: "https:", host: "x.trycloudflare.com" })).toBe("wss://x.trycloudflare.com/ws/session");
    expect(toWsUrl("", "/ws/session", { protocol: "http:", host: "localhost:5173" })).toBe("ws://localhost:5173/ws/session");
  });
  it("puts the API on the configured base (bundled app)", () => {
    expect(joinApi("https://a.trycloudflare.com/", "/api/health")).toBe("https://a.trycloudflare.com/api/health");
    expect(toWsUrl("https://a.trycloudflare.com", "/ws/session", { protocol: "https:", host: "localhost" })).toBe("wss://a.trycloudflare.com/ws/session");
    expect(toWsUrl("http://192.168.1.5:8000", "/ws/session", { protocol: "https:", host: "localhost" })).toBe("ws://192.168.1.5:8000/ws/session");
  });
  it("accepts only a bare http(s) address as a base", () => {
    expect(cleanBase(" https://a.trycloudflare.com/ ")).toBe("https://a.trycloudflare.com");
    for (const bad of ["", null, undefined, "a.com", "javascript:alert(1)", "https://a.com/?token=x", "https://a.com #x"]) expect(cleanBase(bad)).toBe("");
  });
});
