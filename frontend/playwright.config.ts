import { defineConfig, devices } from "@playwright/test";

/**
 * Smoke-test config. Assumes the FastAPI backend is already running on :8000 —
 * the pages fetch real metrics from it, and testing them against a dead API
 * would only prove the error states render.
 */
export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:5173",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
  ],
  webServer: {
    command: "npm run dev",
    url: "http://localhost:5173",
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
