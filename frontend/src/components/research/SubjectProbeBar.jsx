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
import { COMPARE_COLORS, REFERENCE_GRAY } from "../../constants";

const AXIS = "#8a94a1";
const GRID = "#eef0f3";

// Can a fresh classifier still recover subject identity from the learned
// embedding? Lower = more subject-invariant. The diagnostic that proves the
// mechanism: the BASELINE is already near chance (~0.09 vs 0.05), so there is
// almost no decodable subject identity for the adversary to remove — which is
// why DANN can't help here. Two bars (baseline, DANN); chance = dashed line.
export default function SubjectProbeBar({ probe, height = 250 }) {
  if (!probe) return null;
  const chance = probe.chance_mean ?? 0.05;
  const data = [
    { who: "Baseline encoder", acc: probe.baseline_probe_acc_mean, color: COMPARE_COLORS[0] },
    { who: "DANN encoder", acc: probe.dann_probe_acc_mean, color: COMPARE_COLORS[1] },
  ];

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 16, right: 12, bottom: 4, left: 4 }} barCategoryGap="34%">
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis
            dataKey="who"
            tick={{ fill: "#55606d", fontSize: 12.5, fontWeight: 500 }}
            tickLine={false}
            axisLine={{ stroke: "#e4e7eb" }}
          />
          <YAxis
            domain={[0, 0.14]}
            tick={{ fill: AXIS, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={40}
            tickFormatter={(v) => v.toFixed(2)}
          />
          <Tooltip cursor={{ fill: "rgba(79,70,229,0.05)" }} content={<Tip chance={chance} />} />
          <ReferenceLine
            y={chance}
            stroke={REFERENCE_GRAY}
            strokeDasharray="5 4"
            label={{
              value: `chance ${chance.toFixed(2)}`,
              position: "right",
              fill: "#6b7280",
              fontSize: 11,
            }}
          />
          <Bar dataKey="acc" radius={[4, 4, 0, 0]} maxBarSize={90} isAnimationActive={false}>
            {data.map((d) => (
              <Cell key={d.who} fill={d.color} />
            ))}
            <LabelList
              dataKey="acc"
              position="top"
              formatter={(v) => Number(v).toFixed(3)}
              style={{ fill: "#47525f", fontSize: 11.5, fontWeight: 600 }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function Tip({ active, payload, chance }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tip">
      <div className="chart-tip-h">{d.who}</div>
      <div className="chart-tip-row">
        <span className="chart-tip-dot" style={{ background: d.color }} />
        <span className="chart-tip-name">Subject-id recovery</span>
        <span className="chart-tip-val">{Number(d.acc).toFixed(3)}</span>
      </div>
      <div className="chart-tip-row">
        <span className="chart-tip-dot" style={{ background: "#94a3b8" }} />
        <span className="chart-tip-name">Chance</span>
        <span className="chart-tip-val">{chance.toFixed(3)}</span>
      </div>
    </div>
  );
}
