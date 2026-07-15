import {
  Bar,
  BarChart,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

// Horizontal bar chart of mean |SHAP| importance by feature category.
// Single-series magnitude → one hue (indigo), sorted descending.
export default function ShapChart({ categories, height = 300 }) {
  if (!categories || categories.length === 0) return null;
  const data = [...categories]
    .sort((a, b) => b.mean_abs_shap - a.mean_abs_shap)
    .map((c) => ({ category: c.category, value: c.mean_abs_shap }));

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 60, bottom: 4, left: 8 }}
        >
          <XAxis type="number" hide />
          <YAxis
            type="category"
            dataKey="category"
            width={170}
            tickLine={false}
            axisLine={false}
            tick={{ fill: "#55606d", fontSize: 12 }}
          />
          <Tooltip
            formatter={(v) => [Number(v).toFixed(4), "mean |SHAP|"]}
            contentStyle={{ borderRadius: 8, border: "1px solid #e4e7eb", fontSize: 12 }}
          />
          <Bar dataKey="value" fill="#4f46e5" radius={[4, 4, 4, 4]} barSize={18} isAnimationActive={false}>
            <LabelList
              dataKey="value"
              position="right"
              formatter={(v) => Number(v).toFixed(3)}
              style={{ fill: "#8a94a1", fontSize: 11 }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
