import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useChartColors, usePrefersReducedMotion } from "../../lib/chartTheme";

// Grouped comparison of the LOSO configurations' macro F1 WITH fold-to-fold
// error bars (±f1_std over 26 folds) - the plain table hides how much the
// configs overlap. The served ensemble is highlighted in accent. Data: metrics.loso.
export default function LosoConfigChart({ loso, rows, height = 300 }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  if (!loso) return null;

  const data = rows
    .filter((r) => loso[r.key])
    .map((r) => ({
      label: r.label,
      f1: loso[r.key].f1_mean,
      std: loso[r.key].f1_std ?? 0,
      served: !!r.served,
    }));

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 18, right: 16, bottom: 4, left: 4 }} barCategoryGap="30%">
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis
            dataKey="label"
            tick={{ fill: c.ink2, fontSize: 11.5 }}
            tickLine={false}
            axisLine={{ stroke: c.grid }}
            interval={0}
            angle={0}
          />
          <YAxis
            domain={[0, 0.6]}
            tick={{ fill: c.axis, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={38}
            tickFormatter={(v) => v.toFixed(2)}
          />
          <Tooltip cursor={{ fill: "rgba(var(--accent-rgb), 0.06)" }} content={<Tip />} />
          <Bar dataKey="f1" name="macro F1" radius={[4, 4, 0, 0]} maxBarSize={64} isAnimationActive={!reduced} animationDuration={700}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.served ? c.accent : c.eeg} fillOpacity={d.served ? 1 : 0.55} />
            ))}
            <ErrorBar dataKey="std" width={5} strokeWidth={1.5} stroke={c.ink2} direction="y" />
            <LabelList dataKey="f1" position="top" formatter={(v) => v.toFixed(3)} style={{ fill: c.ink2, fontSize: 11, fontWeight: 600 }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function Tip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tip">
      <div className="chart-tip-h">{d.label}{d.served ? " · served" : ""}</div>
      <div className="chart-tip-row">
        <span className="chart-tip-name">Macro F1 (26-fold mean)</span>
        <span className="chart-tip-val">{d.f1.toFixed(4)}</span>
      </div>
      <div className="chart-tip-row">
        <span className="chart-tip-name">Fold std (±)</span>
        <span className="chart-tip-val">{d.std.toFixed(3)}</span>
      </div>
    </div>
  );
}
