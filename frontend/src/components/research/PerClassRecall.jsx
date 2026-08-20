import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { CLASS_COLORS, CLASS_NAMES } from "../../constants";
import { useChartColors, usePrefersReducedMotion } from "../../lib/chartTheme";

// Per-class recall [Low, Medium, High] across LOSO configs. Surfaces the "High
// load is hardest" story hidden inside the recall_mean arrays: bars grouped by
// class, one series per config. Data: metrics.loso[*].recall_mean.
export default function PerClassRecall({ loso, rows, height = 300, classNames = CLASS_NAMES }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  if (!loso) return null;

  const configs = rows.filter(
    (r) => loso[r.key]?.recall_mean?.length === classNames.length
  );
  if (!configs.length) return null;
  // One row per class, one key per config.
  const data = classNames.map((cls, ci) => {
    const row = { cls };
    configs.forEach((r) => {
      row[r.key] = loso[r.key].recall_mean[ci];
    });
    return row;
  });

  // Categorical hue per config from the compare palette range.
  const palette = [c.eeg, c.fnirs, c.accent, c.medium];

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 18, right: 12, bottom: 4, left: 4 }} barGap={2} barCategoryGap="26%">
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis
            dataKey="cls"
            tick={({ x, y, payload }) => (
              <text x={x} y={y + 14} textAnchor="middle" fontSize={12} fontWeight={600} fill={CLASS_COLORS[payload.value]}>
                {payload.value}
              </text>
            )}
            tickLine={false}
            axisLine={{ stroke: c.grid }}
          />
          <YAxis domain={[0, 1]} tick={{ fill: c.axis, fontSize: 11 }} tickLine={false} axisLine={false} width={36} tickFormatter={(v) => v.toFixed(1)} />
          <Tooltip cursor={{ fill: "rgba(var(--accent-rgb), 0.06)" }} content={<Tip />} />
          <Legend verticalAlign="top" align="left" height={28} iconType="circle" iconSize={9} wrapperStyle={{ fontSize: 12, color: "var(--ink-2)" }} />
          {configs.map((r, i) => (
            <Bar key={r.key} dataKey={r.key} name={r.label} fill={palette[i % palette.length]} radius={[3, 3, 0, 0]} maxBarSize={30} isAnimationActive={!reduced} animationDuration={700}>
              {configs.length === 1 && (
                <LabelList dataKey={r.key} position="top" formatter={(v) => v.toFixed(2)} style={{ fill: c.ink2, fontSize: 10.5, fontWeight: 600 }} />
              )}
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function Tip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="chart-tip">
      <div className="chart-tip-h">{label} load · recall</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="chart-tip-row">
          <span className="chart-tip-dot" style={{ background: p.color || p.fill }} />
          <span className="chart-tip-name">{p.name}</span>
          <span className="chart-tip-val">{Number(p.value).toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
}
