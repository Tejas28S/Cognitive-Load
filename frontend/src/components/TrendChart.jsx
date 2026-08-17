import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { CLASS_COLORS, CLASS_NAMES } from "../constants";
import { useChartColors, usePrefersReducedMotion } from "../lib/chartTheme";

const LEVELS = ["Low", "Medium", "High"];

// Predicted cognitive load across a session. y = load level (0/1/2). The
// Medium band is shaded as a literature-informed "sustained-engagement" zone -
// informational only, not a diagnosis or a hard target (Yerkes-Dodson framing).
export default function TrendChart({ session }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  if (!session || session.length === 0) return null;
  const data = session.map((w) => ({
    x: w.window_index,
    predicted: w.predicted_index,
    truth: w.true_label,
    predictedName: w.predicted_label,
    truthName: w.true_label_name,
  }));

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={data} margin={{ top: 8, right: 20, bottom: 24, left: 8 }}>
          {/* Shaded sustained-engagement (Medium) band */}
          <ReferenceArea
            y1={0.5}
            y2={1.5}
            fill={CLASS_COLORS.Medium}
            fillOpacity={0.1}
            stroke="none"
          />
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis
            dataKey="x"
            tick={{ fill: c.axis, fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: c.grid }}
            label={{
              value: "Window (time through session)",
              position: "insideBottom",
              offset: -14,
              fill: c.axis,
              fontSize: 12,
            }}
          />
          <YAxis
            domain={[-0.3, 2.3]}
            ticks={[0, 1, 2]}
            tickFormatter={(v) => LEVELS[v] ?? ""}
            tick={{ fill: c.axis, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={64}
          />
          <Tooltip content={<TrendTooltip />} cursor={{ stroke: c.grid }} />
          <Line
            type="stepAfter"
            dataKey="predicted"
            name="Predicted"
            stroke={c.accent}
            strokeWidth={2}
            dot={{ r: 3, fill: c.accent }}
            isAnimationActive={!reduced}
            animationDuration={700}
          />
          <Scatter dataKey="truth" name="Ground truth" isAnimationActive={!reduced} shape={TruthDot} />
        </ComposedChart>
      </ResponsiveContainer>
      <div style={{ display: "flex", gap: 18, marginTop: 6, flexWrap: "wrap" }}>
        <LegendItem color="var(--accent)" label="Predicted (line)" />
        <LegendItem color="var(--ink-3)" label="Ground truth (dots)" ring />
        <LegendItem color={CLASS_COLORS.Medium} label="Sustained-engagement band" square />
      </div>
    </div>
  );
}

function TruthDot({ cx, cy, payload }) {
  if (cx == null || cy == null) return null;
  return (
    <circle
      cx={cx}
      cy={cy}
      r={4.5}
      fill={CLASS_COLORS[CLASS_NAMES[payload.truth]] || "#94a3b8"}
      stroke="var(--surface)"
      strokeWidth={1.5}
    />
  );
}

function TrendTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tip">
      <div className="chart-tip-h">Window {d.x}</div>
      <div style={{ color: "var(--accent)" }}>Predicted: {d.predictedName}</div>
      <div style={{ color: "var(--ink-2)" }}>Truth: {d.truthName}</div>
    </div>
  );
}

function LegendItem({ color, label, ring, square }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--ink-2)" }}>
      <span
        style={{
          width: 12,
          height: square ? 12 : 12,
          borderRadius: square ? 3 : 999,
          background: square ? color : ring ? color : color,
          opacity: square ? 0.3 : 1,
          border: ring ? "1.5px solid #fff" : "none",
          boxShadow: ring ? "0 0 0 1px #94a3b8" : "none",
        }}
      />
      {label}
    </span>
  );
}
