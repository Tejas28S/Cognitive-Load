import { useEffect, useRef, useState } from "react";

const prefersReduced = () =>
  typeof window !== "undefined" &&
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

// Counts up to `value` once the tile scrolls into view. `format(n)` renders each
// frame (e.g. as a percent). Falls back to the final value instantly under
// reduced-motion. Numbers are set in the display serif + tabular figures.
//
// There used to be an `onDark` variant for the hero's dark instrument band.
// The band is light now, so the tile always renders in ink.
export default function MetricStat({
  value,
  format = (n) => n.toFixed(2),
  label,
  sub,
  accent,
  duration = 1100,
}) {
  const [shown, setShown] = useState(prefersReduced() ? value : 0);
  const ref = useRef(null);
  const done = useRef(false);

  useEffect(() => {
    if (prefersReduced() || done.current) {
      setShown(value);
      return;
    }
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      (entries) => {
        if (!entries[0].isIntersecting || done.current) return;
        done.current = true;
        const start = performance.now();
        const tick = (now) => {
          const p = Math.min(1, (now - start) / duration);
          const eased = 1 - Math.pow(1 - p, 3);
          setShown(value * eased);
          if (p < 1) requestAnimationFrame(tick);
          else setShown(value);
        };
        requestAnimationFrame(tick);
      },
      { threshold: 0.25, rootMargin: "0px 0px -10% 0px" }
    );
    obs.observe(el);
    // Safety net: never leave a real metric stuck at 0. If the tile is still
    // off-screen after a moment (e.g. below the fold on mobile), snap to the
    // true value rather than showing a misleading zero.
    const fallback = setTimeout(() => {
      if (!done.current) {
        done.current = true;
        setShown(value);
      }
    }, 1600);
    return () => {
      obs.disconnect();
      clearTimeout(fallback);
    };
  }, [value, duration]);

  return (
    <div ref={ref}>
      <div
        className="stat-value"
        style={{
          color: accent || "var(--ink)",
          fontSize: "clamp(2rem, 1.4rem + 1.8vw, 2.9rem)",
        }}
      >
        {format(shown)}
      </div>
      <div
        className="stat-label"
        style={{ marginTop: 4 }}
      >
        {label}
      </div>
      {sub && (
        <div
          className="stat-sub"
          style={{ color: "var(--ink-2)", marginTop: 3 }}
        >
          {sub}
        </div>
      )}
    </div>
  );
}
