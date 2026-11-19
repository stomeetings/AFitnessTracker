import { describe, expect, it } from "vitest";
import { classifyCommand } from "./intent";

describe("classifyCommand (twin of the backend)", () => {
  it.each([
    ["Start session", "start_session"], ["let's go", "start_session"], ["stop the session", "stop_session"],
    ["show preview", "show_preview"], ["hide the pre view", "hide_preview"], ["pause", "pause_session"], ["resume", "resume_session"],
    ["don't stop the session", null], ["hello there", null], ["start session and stop session", null],
  ])("%s -> %s", (text, intent) => expect(classifyCommand(text)).toBe(intent));
});
