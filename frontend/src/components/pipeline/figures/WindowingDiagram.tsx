import { motion, useReducedMotion } from "motion/react";
import { DATASET } from "../../../lib/projectFacts";

/**
 * How a continuous recording becomes model inputs: a 10-second window slid
 * forward in 5-second steps, so consecutive windows overlap by half.
 *
 * The overlap is the point of the figure. It is also precisely why the leakage
 * audit checks *window identity* and not just subject identity: two
 * overlapping windows share raw samples, so letting one into train and its
 * neighbour into test would leak, even with the subject split done correctly.
 */
export default function WindowingDiagram() {
  const reduced = useReducedMotion();
  const { lengthSeconds, strideSeconds } = DATASET.windowing;

  const W = 720;
  const H = 190;
  const trackY = 42;
  const pxPerSec = 26;
  const windows = [0, 1, 2, 3, 4];

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-auto w-full min-w-[560px]"
      role="img"
      aria-label={`A ${lengthSeconds}-second analysis window advancing in ${strideSeconds}-second steps, so each window overlaps the previous one by ${lengthSeconds - strideSeconds} seconds.`}
    >
      {/* Continuous recording track */}
      <text
        x={0}
        y={18}
        className="fill-ink-3 font-mono"
        style={{ fontSize: 11, letterSpacing: "0.08em" }}
      >
        CONTINUOUS RECORDING
      </text>
      <rect
        x={0}
        y={trackY}
        width={W}
        height={16}
        rx={3}
        className="fill-line"
      />

      {/* Time axis ticks every 5 s */}
      {Array.from({ length: 11 }, (_, i) => i * strideSeconds).map((t) => (
        <g key={t}>
          <line
            x1={t * pxPerSec}
            y1={trackY + 16}
            x2={t * pxPerSec}
            y2={trackY + 22}
            className="stroke-line-strong"
            strokeWidth={1}
          />
          <text
            x={t * pxPerSec}
            y={trackY + 34}
            textAnchor="middle"
            className="fill-ink-3 font-mono"
            style={{ fontSize: 10 }}
          >
            {t}s
          </text>
        </g>
      ))}

      {/* The sliding windows, staggered in so the advance reads as motion. */}
      {windows.map((i) => {
        const x = i * strideSeconds * pxPerSec;
        const w = lengthSeconds * pxPerSec;
        const y = 104 + i * 15;

        return (
          <motion.g
            key={i}
            initial={reduced ? false : { opacity: 0, x: -14 }}
            whileInView={reduced ? undefined : { opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{
              duration: 0.45,
              delay: 0.12 * i,
              ease: [0.2, 0.65, 0.2, 1],
            }}
          >
            {/* Tie the window back to the stretch of signal it covers */}
            <rect
              x={x}
              y={trackY}
              width={w}
              height={16}
              rx={3}
              fill="var(--accent)"
              opacity={0.1}
            />
            <rect
              x={x}
              y={y}
              width={w}
              height={11}
              rx={2.5}
              fill="var(--accent)"
              opacity={0.14 + i * 0.05}
              stroke="var(--accent)"
              strokeWidth={1}
            />
            <text
              x={x + w + 10}
              y={y + 9}
              className="fill-ink-2 font-mono"
              style={{ fontSize: 10 }}
            >
              w{i + 1}
            </text>
          </motion.g>
        );
      })}

      {/* Stride annotation between the first two windows */}
      <g>
        <line
          x1={0}
          y1={92}
          x2={strideSeconds * pxPerSec}
          y2={92}
          className="stroke-fnirs"
          strokeWidth={1}
          markerEnd="url(#wd-arrow)"
          markerStart="url(#wd-arrow)"
        />
        <text
          x={strideSeconds * pxPerSec + 12}
          y={96}
          className="fill-fnirs font-mono"
          style={{ fontSize: 10 }}
        >
          {strideSeconds}s stride → {lengthSeconds - strideSeconds}s overlap
        </text>
      </g>

      <defs>
        <marker
          id="wd-arrow"
          markerWidth={6}
          markerHeight={6}
          refX={3}
          refY={3}
          orient="auto"
        >
          <circle cx={3} cy={3} r={2} fill="var(--fnirs)" />
        </marker>
      </defs>
    </svg>
  );
}
