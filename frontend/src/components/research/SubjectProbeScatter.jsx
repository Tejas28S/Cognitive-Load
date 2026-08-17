import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { COMPARE_COLORS } from "../../constants";
import { useChartColors, usePrefersReducedMotion } from "../../lib/chartTheme";

// Per-subject subject-identity recovery from the embedding, all 26 LOSO folds -
// the granular view behind the mean-only probe bar. Each subject has a baseline
// and a DANN point; the dashed line is chance. Both hug chance across nearly
// every subject → almost no decodable identity for DANN to strip.
// Data: dann.subject_invariance_probe.per_fold.
export default function SubjectProbeScatter({ probe, height = 300 }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  const perFold = probe?.per_fold;
  if (!perFold?.length) return null;

  const chance = probe.chance_mean ?? perFold[0]?.chance ?? 0.05;
  const data = [...perFold]
    .sort((a, b) => a.subject - b.subject)
    .map((f) => ({
      subject: f.subject,
      baseline: f.baseline_probe_acc,
      dann: f.dann_probe_acc,
    }));

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 14, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis
            dataKey="subject"
            tick={{ fill: c.axis, fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: c.grid }}
            interval={0}
            label={{ value: "subject (LOSO fold)", position: "insideBottom", offset: -12, fill: c.axis, fontSize: 11.5 }}
          />
          <YAxis
            domain={[0, "auto"]}
            tick={{ fill: c.axis, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={40}
            tickFormatter={(v) => v.toFixed(2)}
          />
          <ReferenceLine y={chance} stroke={c.reference} strokeDasharray="5 4"
            label={{ value: `chance ${chance.toFixed(2)}`, position: "insideTopRight", fill: c.axis, fontSize: 10.5 }} />
          <Tooltip cursor={{ stroke: c.grid }} content={<Tip />} />
          {/* faint connectors so paired points read together */}
          <Line dataKey="baseline" name="Baseline" stroke={COMPARE_COLORS[0]} strokeWidth={1} dot={{ r: 3, fill: COMPARE_COLORS[0] }} isAnimationActive={!reduced} animationDuration={700} legendType="circle" />
          <Scatter dataKey="dann" name="DANN" fill={COMPARE_COLORS[1]} isAnimationActive={!reduced} />
        </ComposedChart>
      </ResponsiveContainer>
      <div style={{ display: "flex", gap: 18, marginTop: 6, fontSize: 12, color: "var(--ink-2)" }}>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><span style={{ width: 10, height: 10, borderRadius: 999, background: COMPARE_COLORS[0] }} /> Baseline encoder</span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><span style={{ width: 10, height: 10, borderRadius: 999, background: COMPARE_COLORS[1] }} /> DANN encoder</span>
      </div>
    </div>
  );
}

function Tip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tip">
      <div className="chart-tip-h">Subject {d.subject}</div>
      <div className="chart-tip-row"><span className="chart-tip-dot" style={{ background: COMPARE_COLORS[0] }} /><span className="chart-tip-name">Baseline</span><span className="chart-tip-val">{Number(d.baseline).toFixed(3)}</span></div>
      <div className="chart-tip-row"><span className="chart-tip-dot" style={{ background: COMPARE_COLORS[1] }} /><span className="chart-tip-name">DANN</span><span className="chart-tip-val">{Number(d.dann).toFixed(3)}</span></div>
    </div>
  );
}
