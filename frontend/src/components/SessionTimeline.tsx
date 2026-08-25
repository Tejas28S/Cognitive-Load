import { useMemo, useRef, useState } from "react";
import { CLASS_NAMES, colorForLabel } from "../constants";

type Window = {
  window_index: number;
  timestamp_s?: number;
  predicted_label: string;
  predicted_index: number;
  true_label: number;
  true_label_name: string;
  probabilities?: Record<string, number>;
};

type Run = { start: number; length: number; value: number };

/**
 * A session as a state timeline rather than a line chart.
 *
 * The previous chart plotted class index against window index as a step line
 * with a dot per window. Across a 1,275-window session that is thousands of
 * marks stacked on three y-values: the line becomes a solid block of vertical
 * strokes and nothing about the session is readable.
 *
 * Cognitive load here is a *categorical state over time*, not a quantity, so it
 * is drawn the way categorical sequences are drawn: as bands. Truth and
 * prediction get one lane each, so agreement reads as the two lanes matching,
 * and a third lane marks the errors outright. Long stretches of correct
 * prediction become long blocks of colour instead of visual noise.
 *
 * Class colours come from the class names the served model actually reports,
 * so a binary bundle shows Low/High and nothing else.
 */

/** Collapse a per-window series into runs of equal value. */
function runLengths(values: number[]): Run[] {
  const runs: Run[] = [];
  let start = 0;
  for (let i = 1; i <= values.length; i++) {
    if (i === values.length || values[i] !== values[start]) {
      runs.push({ start, length: i - start, value: values[start] });
      start = i;
    }
  }
  return runs;
}

function Lane({
  runs,
  total,
  colorFor,
  height = 26,
}: {
  runs: Run[];
  total: number;
  colorFor: (v: number) => string;
  height?: number;
}) {
  return (
    <svg
      // One user unit per window, stretched to the container. Widths stay exact
      // at any container size, so runs never drift apart or overlap.
      viewBox={`0 0 ${total} 1`}
      preserveAspectRatio="none"
      style={{ height, width: "100%", display: "block" }}
      shapeRendering="crispEdges"
    >
      {runs.map((r) => (
        <rect
          key={r.start}
          x={r.start}
          y={0}
          width={r.length}
          height={1}
          fill={colorFor(r.value)}
        />
      ))}
    </svg>
  );
}

export default function SessionTimeline({
  session,
  classNames = CLASS_NAMES,
}: {
  session: Window[];
  classNames?: string[];
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);

  const model = useMemo(() => {
    const truth = session.map((w) => w.true_label);
    const pred = session.map((w) => w.predicted_index);
    const wrong = session.map((w) =>
      w.predicted_index === w.true_label ? 0 : 1
    );
    return {
      truthRuns: runLengths(truth),
      predRuns: runLengths(pred),
      errorRuns: runLengths(wrong).filter((r) => r.value === 1),
      confidence: session.map((w) => {
        const p = w.probabilities;
        if (!p) return null;
        const vals = Object.values(p).filter((v) => typeof v === "number");
        return vals.length ? Math.max(...vals) : null;
      }),
    };
  }, [session]);

  if (!session?.length) return null;
  const n = session.length;

  const colorFor = (v: number): string => colorForLabel(classNames[v]);

  const onMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const el = wrapRef.current;
    if (!el) return;
    const box = el.getBoundingClientRect();
    const frac = (e.clientX - box.left) / box.width;
    const idx = Math.max(0, Math.min(n - 1, Math.floor(frac * n)));
    setHover(idx);
  };

  const hovered = hover != null ? session[hover] : null;
  const hoveredConf = hover != null ? model.confidence[hover] : null;
  const hoveredWrong =
    hovered != null && hovered.predicted_index !== hovered.true_label;

  // Confidence polyline. Only drawn when the bundle reports probabilities.
  const confPoints = model.confidence.every((v) => v == null)
    ? null
    : model.confidence
        .map((v, i) => `${i},${1 - (v ?? 0.5)}`)
        .join(" ");

  const axisTicks = Array.from({ length: 6 }, (_, i) =>
    Math.round((i / 5) * (n - 1))
  );

  return (
    <div>
      <div
        ref={wrapRef}
        className="relative"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        <div className="space-y-1.5">
          <LaneRow label="True load">
            <Lane runs={model.truthRuns} total={n} colorFor={colorFor} />
          </LaneRow>

          <LaneRow label="Predicted">
            <Lane runs={model.predRuns} total={n} colorFor={colorFor} />
          </LaneRow>

          <LaneRow label="Errors">
            <div className="relative">
              {/* Correct is the calm ground; only mistakes are inked.
                  The marker is charcoal, not red: "High load" is already red on
                  the class ramp, and a red error bar directly under a red class
                  bar is unreadable as a separate signal. */}
              <div
                className="absolute inset-0 rounded-xs"
                style={{ background: "var(--surface-2)" }}
              />
              <svg
                viewBox={`0 0 ${n} 1`}
                preserveAspectRatio="none"
                style={{ height: 14, width: "100%", display: "block" }}
                shapeRendering="crispEdges"
                className="relative"
              >
                {model.errorRuns.map((r) => (
                  <rect
                    key={r.start}
                    x={r.start}
                    y={0}
                    width={r.length}
                    height={1}
                    fill="var(--ink)"
                    fillOpacity={0.82}
                  />
                ))}
              </svg>
            </div>
          </LaneRow>

          {confPoints && (
            <LaneRow label="Confidence">
              <svg
                viewBox={`0 0 ${n} 1`}
                preserveAspectRatio="none"
                style={{ height: 40, width: "100%", display: "block" }}
              >
                <line
                  x1={0}
                  x2={n}
                  y1={0.5}
                  y2={0.5}
                  stroke="var(--border-strong)"
                  strokeDasharray="4 4"
                  vectorEffect="non-scaling-stroke"
                />
                <polyline
                  points={confPoints}
                  fill="none"
                  stroke="var(--accent)"
                  strokeWidth={1.2}
                  strokeOpacity={0.85}
                  vectorEffect="non-scaling-stroke"
                />
              </svg>
            </LaneRow>
          )}
        </div>

        {/* Crosshair */}
        {hover != null && (
          <div
            aria-hidden="true"
            className="pointer-events-none absolute inset-y-0"
            style={{
              left: `calc(96px + (100% - 96px) * ${(hover + 0.5) / n})`,
              width: 1,
              background: "var(--ink)",
              opacity: 0.45,
            }}
          />
        )}
      </div>

      {/* Axis */}
      <div className="mt-2 flex pl-[96px]">
        <div className="flex w-full justify-between font-mono text-2xs text-ink-3">
          {axisTicks.map((t) => (
            <span key={t}>{t}</span>
          ))}
        </div>
      </div>
      <p className="mt-1 pl-[96px] text-center font-mono text-2xs text-ink-3">
        Window (time through session)
      </p>

      {/* Readout: replaces a floating tooltip so the value has a fixed home. */}
      <div className="mt-4 flex min-h-[2.5rem] flex-wrap items-center gap-x-6 gap-y-2 rounded-sm bg-surface-2 px-4 py-2.5 text-sm">
        {hovered ? (
          <>
            <span className="font-mono text-2xs tracking-wider text-ink-3 uppercase">
              Window {hovered.window_index}
            </span>
            <span className="flex items-center gap-2">
              <Swatch color={colorFor(hovered.true_label)} />
              <span className="text-ink-3">True</span>
              <strong className="text-ink">{hovered.true_label_name}</strong>
            </span>
            <span className="flex items-center gap-2">
              <Swatch color={colorFor(hovered.predicted_index)} />
              <span className="text-ink-3">Predicted</span>
              <strong className="text-ink">{hovered.predicted_label}</strong>
            </span>
            {hoveredConf != null && (
              <span className="text-ink-2">
                confidence{" "}
                <strong className="text-ink tnum">
                  {(hoveredConf * 100).toFixed(0)}%
                </strong>
              </span>
            )}
            <span
              className={`font-medium ${hoveredWrong ? "text-stop" : "text-ok"}`}
            >
              {hoveredWrong ? "incorrect" : "correct"}
            </span>
          </>
        ) : (
          <span className="text-ink-3">
            Hover the timeline to inspect any window.
          </span>
        )}
      </div>

      {/* Legend */}
      <div className="mt-3 flex flex-wrap gap-x-6 gap-y-2 text-xs text-ink-2">
        {classNames.map((name) => (
          <span key={name} className="flex items-center gap-2">
            <Swatch color={colorForLabel(name)} />
            {name} load
          </span>
        ))}
        <span className="flex items-center gap-2">
          <Swatch color="var(--ink)" />
          Misclassified window
        </span>
      </div>
    </div>
  );
}

function LaneRow({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-[84px] shrink-0 text-right font-mono text-2xs tracking-wide text-ink-3 uppercase">
        {label}
      </span>
      <div className="min-w-0 flex-1 overflow-hidden rounded-xs">{children}</div>
    </div>
  );
}

function Swatch({ color }: { color: string }) {
  return (
    <span
      aria-hidden="true"
      className="inline-block h-2.5 w-2.5 shrink-0 rounded-[2px]"
      style={{ background: color }}
    />
  );
}
