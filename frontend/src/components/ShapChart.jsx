import {
  Bar,
  BarChart,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import ChartTooltip from "./ChartTooltip";
import { useChartColors, usePrefersReducedMotion } from "../lib/chartTheme";

// Horizontal bar chart of mean |SHAP| importance by feature category.
// Single-series magnitude → one hue (indigo), sorted descending.
export default function ShapChart({ categories, height = 300 }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  if (!categories || categories.length === 0) return null;
  const data = [...categories]
    .sort((a, b) => b.mean_abs_shap - a.mean_abs_shap)
    .map((cat) => ({ category: cat.category, value: cat.mean_abs_shap }));

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
            tick={{ fill: c.ink2, fontSize: 12 }}
          />
          <Tooltip
            cursor={{ fill: c.grid, fillOpacity: 0.3 }}
            content={<ChartTooltip format={(v) => Number(v).toFixed(4)} />}
          />
          <Bar dataKey="value" name="mean |SHAP|" fill={c.accent} radius={[4, 4, 4, 4]} barSize={18} isAnimationActive={!reduced} animationDuration={700}>
            <LabelList
              dataKey="value"
              position="right"
              formatter={(v) => Number(v).toFixed(3)}
              style={{ fill: c.axis, fontSize: 11 }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
