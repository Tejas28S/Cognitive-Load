import { useMemo } from "react";
import { motion, useReducedMotion } from "motion/react";

/**
 * Ambient hero background: the project's actual subject matter rendered as
 * signal, not decoration.
 *
 * Two families of trace, matching the two modalities the model fuses:
 *  - EEG (indigo): fast, spiky, high-frequency — electrical activity.
 *  - fNIRS (cyan): slow, smooth, lagging — the hemodynamic response.
 *
 * Both are drawn from deterministic maths (no RNG) so the figure is identical
 * on every render and across SSR/hydration, and both are kept at low opacity so
 * they read as texture behind the type rather than as a chart competing with
 * it. The hero sits on paper, not on a dark panel, so the grid is ink-on-light
 * and the traces carry a little more weight than they would on dark.
 */

const W = 1200;
const H = 420;

/** Cheap deterministic hash -> [-1, 1]. Stands in for noise without a PRNG. */
function wobble(i: number, seed: number) {
  const x = Math.sin(i * 12.9898 + seed * 78.233) * 43758.5453;
  return (x - Math.floor(x)) * 2 - 1;
}

/** A spiky EEG-like trace: broadband wobble layered over a slow drift. */
function eegPath(seed: number, amp: number, yBase: number) {
  const pts: string[] = [];
  const steps = 240;
  for (let i = 0; i <= steps; i++) {
    const x = (i / steps) * W;
    const drift = Math.sin(i / 26 + seed) * amp * 0.45;
    const fast =
      wobble(i, seed) * amp * 0.55 + wobble(i * 3.1, seed + 5) * amp * 0.3;
    pts.push(`${x.toFixed(1)},${(yBase + drift + fast).toFixed(1)}`);
  }
  return `M${pts.join("L")}`;
}

/** A smooth fNIRS-like trace: a couple of slow sines, no high frequencies. */
function fnirsPath(seed: number, amp: number, yBase: number) {
  const pts: string[] = [];
  const steps = 120;
  for (let i = 0; i <= steps; i++) {
    const x = (i / steps) * W;
    const y =
      yBase +
      Math.sin(i / 15 + seed) * amp +
      Math.sin(i / 38 + seed * 2) * amp * 0.55;
    pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  }
  return `M${pts.join("L")}`;
}

export default function SignalField() {
  const reduced = useReducedMotion();

  const traces = useMemo(
    () => ({
      eeg: [
        { d: eegPath(1.2, 26, 96), opacity: 0.4, width: 1.1, dur: 26 },
        { d: eegPath(4.7, 18, 178), opacity: 0.26, width: 1, dur: 34 },
        { d: eegPath(9.1, 22, 330), opacity: 0.22, width: 1, dur: 30 },
      ],
      fnirs: [
        { d: fnirsPath(0.6, 30, 250), opacity: 0.45, width: 1.7, dur: 40 },
        { d: fnirsPath(3.3, 22, 372), opacity: 0.3, width: 1.5, dur: 48 },
      ],
    }),
    []
  );

  return (
    <div
      aria-hidden="true"
      className="pointer-events-none absolute inset-0 overflow-hidden"
    >
      {/* Faint measurement grid - the graph-paper substrate of a lab record.
          Uses the ink token at low alpha, so it inverts correctly with the
          theme instead of being hardcoded white-on-dark. */}
      <div
        className="absolute inset-0"
        style={{
          backgroundImage:
            "linear-gradient(to right, color-mix(in srgb, var(--ink) 6%, transparent) 1px, transparent 1px), linear-gradient(to bottom, color-mix(in srgb, var(--ink) 6%, transparent) 1px, transparent 1px)",
          backgroundSize: "72px 72px",
          maskImage:
            "radial-gradient(ellipse 80% 60% at 50% 40%, black 30%, transparent 78%)",
        }}
      />

      <svg
        className="absolute inset-x-0 bottom-0 h-full w-full"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        style={{
          maskImage:
            "linear-gradient(to bottom, transparent 0%, black 22%, black 72%, transparent 100%)",
        }}
      >
        {/* Each trace is drawn twice end-to-end and translated by exactly one
            width, so the loop is seamless rather than snapping back. */}
        {traces.fnirs.map((t, i) => (
          <motion.g
            key={`f${i}`}
            animate={reduced ? undefined : { x: [0, -W] }}
            transition={
              reduced
                ? undefined
                : { duration: t.dur, repeat: Infinity, ease: "linear" }
            }
          >
            <path
              d={t.d}
              fill="none"
              stroke="var(--fnirs)"
              strokeWidth={t.width}
              strokeOpacity={t.opacity}
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
            <path
              d={t.d}
              transform={`translate(${W},0)`}
              fill="none"
              stroke="var(--fnirs)"
              strokeWidth={t.width}
              strokeOpacity={t.opacity}
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
          </motion.g>
        ))}

        {traces.eeg.map((t, i) => (
          <motion.g
            key={`e${i}`}
            animate={reduced ? undefined : { x: [0, -W] }}
            transition={
              reduced
                ? undefined
                : { duration: t.dur, repeat: Infinity, ease: "linear" }
            }
          >
            <path
              d={t.d}
              fill="none"
              stroke="var(--eeg)"
              strokeWidth={t.width}
              strokeOpacity={t.opacity}
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
            />
            <path
              d={t.d}
              transform={`translate(${W},0)`}
              fill="none"
              stroke="var(--eeg)"
              strokeWidth={t.width}
              strokeOpacity={t.opacity}
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
            />
          </motion.g>
        ))}
      </svg>
    </div>
  );
}
