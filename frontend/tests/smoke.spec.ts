import { test, expect, type ConsoleMessage, type Page } from "@playwright/test";

/**
 * Route smoke test.
 *
 * The build and `tsc` between them prove the app compiles; neither proves it
 * renders. This mounts every route in a real browser and fails on anything
 * that would show a user a broken page: an uncaught exception, a React error,
 * or an empty <main>.
 *
 * Console noise that is expected and not a defect is filtered explicitly below
 * rather than by lowering the bar — an ignore list that swallows real errors
 * would make this test worse than useless.
 */

const ROUTES = [
  { path: "/", name: "Landing", expect: /Reading mental workload/i },
  { path: "/dashboard", name: "Dashboard", expect: /Model dashboard/i },
  { path: "/overview", name: "Overview", expect: /Reading mental workload|Performance Comparison/i },
  { path: "/workflow", name: "Workflow", expect: /From two brain signals/i },
  { path: "/methodology", name: "Methodology", expect: /Methodology/i },
  { path: "/live", name: "Live Prediction", expect: /./ },
  { path: "/session", name: "Session Trend", expect: /./ },
  { path: "/upload", name: "Upload Data", expect: /./ },
  { path: "/insights", name: "Model Insights", expect: /./ },
  { path: "/evaluation", name: "Evaluation", expect: /./ },
  { path: "/runs", name: "Run Archive", expect: /./ },
  { path: "/about", name: "About", expect: /Why measure cognitive load/i },
  { path: "/bios", name: "Bios", expect: /Who built this/i },
  { path: "/faq", name: "FAQ", expect: /Frequently asked questions/i },
];

/** Console messages that are noise, not defects. */
const IGNORE = [
  /Download the React DevTools/i,
  /\[vite\] connect(ing|ed)/i,
  /Recharts.*width\(0\) and height\(0\)/i, // container measured pre-layout
  // The browser logs every non-2xx response as a console error even when the
  // app caught it and degraded on purpose. Failed requests are asserted
  // separately below, against an explicit allowlist, so suppressing the
  // duplicate console line here loses no signal.
  /Failed to load resource/i,
];

/**
 * API requests that are EXPECTED to fail, with the reason. Anything failing
 * outside this list is a defect.
 */
const ALLOWED_FAILURES: { pattern: RegExp; why: string }[] = [
  {
    pattern: /\/api\/model\/confusion-matrix\?model=(eeg|fnirs)&version=binary/,
    why: "single-modality matrices only exist for the 3-class bundle; ModelInsights hides those panels",
  },
  {
    pattern: /\/api\/research\/summary\?version=3class/,
    why: "the 3-class bundle ships no research summary; the Research page falls back",
  },
];

function collectProblems(page: Page) {
  const problems: string[] = [];

  page.on("console", (msg: ConsoleMessage) => {
    if (msg.type() !== "error") return;
    const text = msg.text();
    if (IGNORE.some((re) => re.test(text))) return;
    problems.push(`console.error: ${text}`);
  });

  page.on("pageerror", (err) => {
    problems.push(`pageerror: ${err.message}`);
  });

  page.on("response", (res) => {
    if (res.status() < 400) return;
    const url = res.url();
    if (ALLOWED_FAILURES.some((a) => a.pattern.test(url))) return;
    problems.push(`HTTP ${res.status()}: ${url}`);
  });

  return problems;
}

for (const route of ROUTES) {
  test(`${route.name} (${route.path}) renders without errors`, async ({ page }) => {
    const problems = collectProblems(page);

    await page.goto(route.path, { waitUntil: "networkidle" });

    // The route actually painted something.
    const main = page.locator("main.route-view");
    await expect(main).toBeVisible();
    await expect(main).not.toBeEmpty();
    await expect(page.locator("body")).toContainText(route.expect);

    // React's error boundary output would land here.
    await expect(page.locator("body")).not.toContainText(
      /Something went wrong|Minified React error/i
    );

    expect(problems, `Runtime problems on ${route.path}`).toEqual([]);
  });
}

test("pipeline stepper tracks the section in view", async ({ page }) => {
  await page.goto("/workflow", { waitUntil: "networkidle" });

  const rail = page.getByRole("navigation", { name: /pipeline stages/i });
  await expect(rail).toBeVisible();

  // First stage is active on arrival.
  await expect(rail.locator('[aria-current="step"]')).toContainText(
    /Data collection/i
  );

  // Jumping to a later stage moves the active node.
  await page.locator("#findings").scrollIntoViewIfNeeded();
  await expect(rail.locator('[aria-current="step"]')).toContainText(/Results/i, {
    timeout: 10_000,
  });
});

test("no page scrolls horizontally at mobile width", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 800 });

  for (const route of ROUTES) {
    await page.goto(route.path, { waitUntil: "networkidle" });
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth
    );
    expect(overflow, `${route.path} overflows horizontally by ${overflow}px`).toBeLessThanOrEqual(1);
  }
});
