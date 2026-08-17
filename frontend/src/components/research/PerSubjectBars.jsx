import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useChartColors, usePrefersReducedMotion } from "../../lib/chartTheme";

// Per-subject accuracy on the held-out demo windows - makes the documented
// subject-variability ceiling concrete: the same frozen model swings widely
// from one person to the next. Dashed line = mean across subjects.
// Data: diagnostics.per_subject.
export default function PerSubjectBars({ perSubject, height = 280 }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  if (!perSubject?.length) return null;

  const data = perSubject.map((s) => ({
    subject: `S${s.subject_id}`,
    accuracy: s.accuracy,
    n: s.n,
    conf: s.mean_confidence,
  }));
  const mean = data.reduce((a, b) => a + b.accuracy, 0) / data.length;

  // Color by accuracy: teal (good) → amber → red (weak), keyed to the class ramp.
  const barColor = (a) => (a >= 0.5 ? c.low : a >= 0.4 ? c.medium : c.high);

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 18, right: 14, bottom: 4, left: 4 }} barCategoryGap="26%">
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis dataKey="subject" tick={{ fill: c.ink2, fontSize: 12, fontWeight: 600 }} tickLine={false} axisLine={{ stroke: c.grid }} />
          <YAxis domain={[0, 0.8]} tick={{ fill: c.axis, fontSize: 11 }} tickLine={false} axisLine={false} width={36} tickFormatter={(v) => v.toFixed(1)} />
          <ReferenceLine y={mean} stroke={c.reference} strokeDasharray="5 4"
            label={{ value: `mean ${mean.toFixed(2)}`, position: "right", fill: c.axis, fontSize: 10.5 }} />
          <Tooltip cursor={{ fill: "rgba(var(--accent-rgb), 0.06)" }} content={<Tip />} />
          <Bar dataKey="accuracy" radius={[4, 4, 0, 0]} maxBarSize={54} isAnimationActive={!reduced} animationDuration={700}>
            {data.map((d, i) => (
              <Cell key={i} fill={barColor(d.accuracy)} />
            ))}
            <LabelList dataKey="accuracy" position="top" formatter={(v) => v.toFixed(2)} style={{ fill: c.ink2, fontSize: 11, fontWeight: 600 }} />
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
      <div className="chart-tip-h">Subject {d.subject.slice(1)}</div>
      <div className="chart-tip-row"><span className="chart-tip-name">Accuracy</span><span className="chart-tip-val">{d.accuracy.toFixed(3)}</span></div>
      <div className="chart-tip-row"><span className="chart-tip-name">Mean confidence</span><span className="chart-tip-val">{d.conf.toFixed(3)}</span></div>
      <div className="chart-tip-row"><span className="chart-tip-name">Windows</span><span className="chart-tip-val">{d.n}</span></div>
    </div>
  );
}
