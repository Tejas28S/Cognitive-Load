import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useChartColors, usePrefersReducedMotion } from "../../lib/chartTheme";

// Confidence–coverage curve: sort predictions by confidence and, at each
// coverage level (fraction of windows kept, most-confident first), plot the
// accuracy on that kept subset. A model whose confidence is meaningfully ordered
// gets MORE accurate as coverage drops - this is the evidence behind the shipped
// abstain behaviour. Data: diagnostics.coverage_curve.
export default function CoverageCurve({ curve, fullAccuracy, height = 300 }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  if (!curve?.length) return null;

  const data = curve.map((p) => ({
    coverage: p.coverage,
    accuracy: p.accuracy,
    n: p.n,
  }));
  const base = fullAccuracy ?? data.find((d) => d.coverage >= 0.999)?.accuracy;

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={data} margin={{ top: 12, right: 18, bottom: 26, left: 6 }}>
          <defs>
            <linearGradient id="cov-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={c.accent} stopOpacity={0.28} />
              <stop offset="100%" stopColor={c.accent} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis
            dataKey="coverage"
            type="number"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            tick={{ fill: c.axis, fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: c.grid }}
            label={{ value: "coverage (windows kept, most-confident first) →", position: "insideBottom", offset: -14, fill: c.axis, fontSize: 11.5 }}
          />
          <YAxis
            domain={[0.3, "auto"]}
            tickFormatter={(v) => v.toFixed(2)}
            tick={{ fill: c.axis, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={40}
            label={{ value: "accuracy", angle: -90, position: "insideLeft", fill: c.axis, fontSize: 11.5 }}
          />
          {base != null && (
            <ReferenceLine
              y={base}
              stroke={c.reference}
              strokeDasharray="5 4"
              label={{ value: `full-coverage ${base.toFixed(2)}`, position: "insideTopRight", fill: c.axis, fontSize: 10.5 }}
            />
          )}
          <Tooltip cursor={{ stroke: c.grid }} content={<CovTip />} />
          <Area
            type="monotone"
            dataKey="accuracy"
            stroke={c.accent}
            strokeWidth={2.5}
            fill="url(#cov-fill)"
            dot={false}
            activeDot={{ r: 5 }}
            isAnimationActive={!reduced}
            animationDuration={800}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function CovTip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tip">
      <div className="chart-tip-h">Coverage {(d.coverage * 100).toFixed(0)}%</div>
      <div className="chart-tip-row">
        <span className="chart-tip-name">Accuracy on kept subset</span>
        <span className="chart-tip-val">{d.accuracy.toFixed(3)}</span>
      </div>
      <div className="chart-tip-row">
        <span className="chart-tip-name">Windows kept</span>
        <span className="chart-tip-val">{d.n}</span>
      </div>
    </div>
  );
}
