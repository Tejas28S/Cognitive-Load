import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const AXIS = "#8a94a1";
const GRID = "#eef0f3";
const INK = "#4f46e5";

// Single-series line: adversarial strength (lambda_max) vs Hybrid macro F1.
// This is the controlled on/off comparison (one code path). It falls
// monotonically as lambda rises — the adversary injects noise rather than
// removing a harmful confound. One series → no legend; endpoints direct-labeled.
export default function LambdaSweepChart({ sweep, height = 280 }) {
  if (!sweep) return null;
  const data = Object.keys(sweep)
    .map((k) => ({ lambda: parseFloat(k), f1: sweep[k].f1_mean }))
    .sort((a, b) => a.lambda - b.lambda);
  if (!data.length) return null;

  const first = data[0];
  const last = data[data.length - 1];

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 18, right: 46, bottom: 24, left: 4 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis
            dataKey="lambda"
            type="number"
            domain={[0, 2]}
            ticks={[0, 0.5, 1, 2]}
            tick={{ fill: AXIS, fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "#e4e7eb" }}
            label={{
              value: "adversary strength  λ_max  →",
              position: "insideBottom",
              offset: -12,
              fill: AXIS,
              fontSize: 11.5,
            }}
          />
          <YAxis
            domain={[0.34, 0.44]}
            tick={{ fill: AXIS, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={42}
            tickFormatter={(v) => v.toFixed(2)}
          />
          <Tooltip content={<Tip />} />
          <Line
            type="monotone"
            dataKey="f1"
            stroke={INK}
            strokeWidth={2.5}
            dot={{ r: 4, fill: INK, strokeWidth: 0 }}
            activeDot={{ r: 6 }}
            isAnimationActive={false}
          />
          <ReferenceDot x={first.lambda} y={first.f1} r={0}
            label={{ value: `λ=0 · ${first.f1.toFixed(3)} (off)`, position: "top", fill: "#0f766e", fontSize: 11, fontWeight: 600 }} />
          <ReferenceDot x={last.lambda} y={last.f1} r={0}
            label={{ value: last.f1.toFixed(3), position: "right", fill: "#b91c1c", fontSize: 11, fontWeight: 600 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function Tip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tip">
      <div className="chart-tip-h">λ_max = {d.lambda}</div>
      <div className="chart-tip-row">
        <span className="chart-tip-dot" style={{ background: INK }} />
        <span className="chart-tip-name">Hybrid macro F1</span>
        <span className="chart-tip-val">{d.f1.toFixed(4)}</span>
      </div>
    </div>
  );
}
