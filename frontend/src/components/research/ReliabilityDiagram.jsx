import {
  Bar,
  Cell,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import ChartTooltip from "../ChartTooltip";
import { useChartColors, usePrefersReducedMotion } from "../../lib/chartTheme";

// Reliability (calibration) diagram: for each confidence bin, bar height = the
// model's accuracy on windows that fell in that bin, plotted against the bin's
// mean confidence (x). A perfectly calibrated model sits on the diagonal
// y = x. Bars above the line = under-confident; below = over-confident. ECE is
// the confidence-weighted average vertical gap. Data: diagnostics.reliability.
export default function ReliabilityDiagram({ reliability, height = 300 }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  if (!reliability?.bins?.length) return null;

  // Keep only populated bins; x = mean confidence, y = accuracy.
  const data = reliability.bins
    .filter((b) => b.count > 0 && b.avg_confidence != null)
    .map((b) => ({
      x: b.avg_confidence,
      accuracy: b.accuracy,
      gap: b.accuracy - b.avg_confidence,
      count: b.count,
      label: `${(b.bin_lo * 100).toFixed(0)}–${(b.bin_hi * 100).toFixed(0)}%`,
    }));

  const barColor = (gap) => (gap >= 0 ? c.low : c.high);

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 12, right: 18, bottom: 26, left: 6 }}>
          <XAxis
            type="number"
            dataKey="x"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tickFormatter={(v) => v.toFixed(2)}
            tick={{ fill: c.axis, fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: c.grid }}
            label={{ value: "mean confidence in bin →", position: "insideBottom", offset: -14, fill: c.axis, fontSize: 11.5 }}
          />
          <YAxis
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tickFormatter={(v) => v.toFixed(2)}
            tick={{ fill: c.axis, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={40}
            label={{ value: "accuracy", angle: -90, position: "insideLeft", fill: c.axis, fontSize: 11.5 }}
          />
          {/* Perfect-calibration diagonal y = x */}
          <ReferenceLine
            segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
            stroke={c.reference}
            strokeDasharray="5 4"
            ifOverflow="extendDomain"
          />
          <Tooltip
            cursor={{ fill: "rgba(var(--accent-rgb), 0.06)" }}
            content={<ReliabilityTip />}
          />
          <Bar dataKey="accuracy" barSize={16} radius={[3, 3, 0, 0]} isAnimationActive={!reduced} animationDuration={700}>
            {data.map((d, i) => (
              <Cell key={i} fill={barColor(d.gap)} fillOpacity={0.85} />
            ))}
          </Bar>
          <Line
            type="monotone"
            dataKey="accuracy"
            stroke={c.accent}
            strokeWidth={2}
            dot={{ r: 3, fill: c.accent }}
            isAnimationActive={!reduced}
            animationDuration={800}
          />
        </ComposedChart>
      </ResponsiveContainer>
      <div style={{ display: "flex", gap: 16, marginTop: 6, flexWrap: "wrap", fontSize: 12, color: "var(--ink-2)" }}>
        <Legend color={c.reference} label="Perfect calibration (y = x)" dash />
        <Legend color={c.low} label="Under-confident (accuracy > confidence)" />
        <Legend color={c.high} label="Over-confident" />
        <span className="mono" style={{ marginLeft: "auto" }}>
          ECE = {(reliability.ece ?? 0).toFixed(3)}
        </span>
      </div>
    </div>
  );
}

function Legend({ color, label, dash }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <span style={{ width: 14, height: dash ? 0 : 10, borderRadius: 2, background: dash ? "transparent" : color, borderTop: dash ? `2px dashed ${color}` : "none" }} />
      {label}
    </span>
  );
}

function ReliabilityTip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tip">
      <div className="chart-tip-h">Confidence bin {d.label}</div>
      <div className="chart-tip-row">
        <span className="chart-tip-name">Mean confidence</span>
        <span className="chart-tip-val">{d.x.toFixed(3)}</span>
      </div>
      <div className="chart-tip-row">
        <span className="chart-tip-name">Accuracy</span>
        <span className="chart-tip-val">{d.accuracy.toFixed(3)}</span>
      </div>
      <div className="chart-tip-row">
        <span className="chart-tip-name">Windows</span>
        <span className="chart-tip-val">{d.count}</span>
      </div>
    </div>
  );
}
