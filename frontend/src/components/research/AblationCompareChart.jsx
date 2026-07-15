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

const AXIS = "#8a94a1";
const GRID = "#eef0f3";

// Reusable grouped-bar comparison for the decision-gate ablations (HbR 468 vs
// 936, classical RF/XGB/SVM). Magnitude → bars; identity of each config → a
// validated categorical hue. Legend always present; values direct-labeled.
export default function AblationCompareChart({
  data,
  categoryKey,
  series,
  height = 300,
  domain = [0, 0.6],
  unit = "macro F1",
}) {
  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={data}
          margin={{ top: 10, right: 12, bottom: 4, left: 4 }}
          barGap={2}
          barCategoryGap="28%"
        >
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis
            dataKey={categoryKey}
            tick={{ fill: "#55606d", fontSize: 12.5, fontWeight: 500 }}
            tickLine={false}
            axisLine={{ stroke: "#e4e7eb" }}
          />
          <YAxis
            domain={domain}
            tick={{ fill: AXIS, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={38}
          />
          <Tooltip
            cursor={{ fill: "rgba(79,70,229,0.05)" }}
            content={<GroupTooltip series={series} unit={unit} />}
          />
          <Legend
            verticalAlign="top"
            align="left"
            height={30}
            iconType="circle"
            iconSize={9}
            wrapperStyle={{ fontSize: 12.5, color: "#47525f" }}
          />
          {series.map((s) => (
            <Bar
              key={s.key}
              dataKey={s.key}
              name={s.name}
              fill={s.color}
              radius={[4, 4, 0, 0]}
              maxBarSize={54}
              isAnimationActive={false}
            >
              {series.length <= 2 && (
                <LabelList
                  dataKey={s.key}
                  position="top"
                  formatter={(v) => (v == null ? "" : Number(v).toFixed(3))}
                  style={{ fill: s.color, fontSize: 11, fontWeight: 600 }}
                />
              )}
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function GroupTooltip({ active, payload, label, unit }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="chart-tip">
      <div className="chart-tip-h">{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="chart-tip-row">
          <span className="chart-tip-dot" style={{ background: p.color }} />
          <span className="chart-tip-name">{p.name}</span>
          <span className="chart-tip-val">
            {Number(p.value).toFixed(3)} {unit}
          </span>
        </div>
      ))}
    </div>
  );
}
