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
import { COMPARE_COLORS } from "../../constants";

const AXIS = "#8a94a1";
const GRID = "#eef0f3";

// Two small panels that carry the protocol-taxonomy argument:
//   (a) within-subject vs LOSO — the gap is small once per-subject normalized,
//       so the honest LOSO number is not propped up by subject leakage.
//   (b) 3-class vs binary (LOSO) — collapsing to binary lifts F1 ~+0.20, which
//       is what actually explains the distance to literature's 60-90%.
// Both are macro F1 on the SAME RandomForest + features (only the protocol changes).
export default function ProtocolBenchmarkChart({ protocol }) {
  if (!protocol) return null;

  const leakage = [
    {
      cond: "No per-subject norm",
      within: protocol.three_class_no_normalization?.within_subject?.f1_mean,
      loso: protocol.three_class_no_normalization?.leave_one_subject_out?.f1_mean,
    },
    {
      cond: "Per-subject norm (locked)",
      within: protocol.three_class_per_subject_norm?.within_subject?.f1_mean,
      loso: protocol.three_class_per_subject_norm?.leave_one_subject_out?.f1_mean,
    },
  ];

  const difficulty = [
    {
      task: "3-class (L/M/H)",
      within: protocol.three_class_per_subject_norm?.within_subject?.f1_mean,
      loso: protocol.three_class_per_subject_norm?.leave_one_subject_out?.f1_mean,
    },
    {
      task: "Binary (Low vs High)",
      within: protocol.binary_low_vs_high_per_subject_norm?.within_subject?.f1_mean,
      loso: protocol.binary_low_vs_high_per_subject_norm?.leave_one_subject_out?.f1_mean,
    },
  ];

  const series = [
    { key: "within", name: "Within-subject", color: COMPARE_COLORS[1] },
    { key: "loso", name: "LOSO (cross-subject)", color: COMPARE_COLORS[0] },
  ];

  return (
    <div className="grid grid-2" style={{ gap: 22 }}>
      <Panel
        title="Subject leakage (3-class)"
        note="Small gap → the LOSO figure is honest, not leakage-inflated."
        data={leakage}
        catKey="cond"
        series={series}
      />
      <Panel
        title="Task difficulty (per-subject normalized)"
        note="Binary lifts F1 ~+0.20 — the real gap to easier literature protocols."
        data={difficulty}
        catKey="task"
        series={series}
      />
    </div>
  );
}

function Panel({ title, note, data, catKey, series }) {
  return (
    <div>
      <div className="eyebrow" style={{ marginBottom: 4 }}>{title}</div>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={250}>
          <BarChart
            data={data}
            margin={{ top: 12, right: 8, bottom: 4, left: 4 }}
            barGap={2}
            barCategoryGap="30%"
          >
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis
              dataKey={catKey}
              tick={{ fill: "#55606d", fontSize: 11.5 }}
              tickLine={false}
              axisLine={{ stroke: "#e4e7eb" }}
              interval={0}
            />
            <YAxis
              domain={[0, 0.7]}
              tick={{ fill: AXIS, fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={34}
            />
            <Tooltip
              cursor={{ fill: "rgba(79,70,229,0.05)" }}
              content={<Tip />}
            />
            <Legend
              verticalAlign="top"
              align="left"
              height={26}
              iconType="circle"
              iconSize={9}
              wrapperStyle={{ fontSize: 12, color: "#47525f" }}
            />
            {series.map((s) => (
              <Bar
                key={s.key}
                dataKey={s.key}
                name={s.name}
                fill={s.color}
                radius={[4, 4, 0, 0]}
                maxBarSize={46}
                isAnimationActive={false}
              >
                <LabelList
                  dataKey={s.key}
                  position="top"
                  formatter={(v) => (v == null ? "" : Number(v).toFixed(2))}
                  style={{ fill: s.color, fontSize: 10.5, fontWeight: 600 }}
                />
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="muted" style={{ fontSize: "0.82rem", marginTop: 6 }}>{note}</p>
    </div>
  );
}

function Tip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="chart-tip">
      <div className="chart-tip-h">{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="chart-tip-row">
          <span className="chart-tip-dot" style={{ background: p.color }} />
          <span className="chart-tip-name">{p.name}</span>
          <span className="chart-tip-val">{Number(p.value).toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
}
