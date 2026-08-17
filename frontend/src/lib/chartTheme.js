import { useEffect, useState } from "react";

// Recharts can't read CSS custom properties for axis/grid/tick colors, so charts
// that hardcode hex values break under the dark theme. This hook reads the
// resolved token values off <html> and re-reads whenever the theme flips
// (ThemeToggle stamps data-theme on <html>), forcing chart re-render with the
// right colors. Returns a small palette used across every chart.

const readVar = (name, fallback) => {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
};

const readPalette = () => ({
  axis: readVar("--ink-3", "#8a94a1"),
  grid: readVar("--border", "#eef0f3"),
  ink: readVar("--ink", "#101826"),
  ink2: readVar("--ink-2", "#47525f"),
  surface: readVar("--surface", "#ffffff"),
  accent: readVar("--accent", "#4f46e5"),
  eeg: readVar("--eeg", "#6366f1"),
  fnirs: readVar("--fnirs", "#06b6d4"),
  low: readVar("--low", "#0d9488"),
  medium: readVar("--medium", "#d97706"),
  high: readVar("--high", "#dc2626"),
  reference: readVar("--ink-3", "#94a3b8"),
});

export function useChartColors() {
  const [palette, setPalette] = useState(readPalette);

  useEffect(() => {
    const update = () => setPalette(readPalette());
    // Re-read when the theme attribute changes on <html>.
    const obs = new MutationObserver(update);
    obs.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    // Also react to OS scheme changes (theme = "system").
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    mq?.addEventListener?.("change", update);
    return () => {
      obs.disconnect();
      mq?.removeEventListener?.("change", update);
    };
  }, []);

  return palette;
}

// Shared: whether the viewer prefers reduced motion (charts skip draw-in).
export function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(
    () => typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches
  );
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const on = () => setReduced(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return reduced;
}

// Consistent axis tick styling for every Recharts axis.
export const axisTick = (colors) => ({ fill: colors.axis, fontSize: 11 });
