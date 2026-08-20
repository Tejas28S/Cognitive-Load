import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";
import { CLASS_COLORS, CLASS_NAMES } from "../constants";
import { useChartColors, usePrefersReducedMotion } from "../lib/chartTheme";

// Horizontal probability bars, one per class. The predicted (max) class is
// drawn at full opacity; the others are dimmed so identity reads instantly.
export default function ProbabilityBars({ probabilities, predicted, classNames = CLASS_NAMES }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  if (!probabilities) return null;
  const data = classNames.map((name) => ({
    name,
    prob: probabilities[name] ?? 0,
  }));
  const top = predicted || data.reduce((a, b) => (b.prob > a.prob ? b : a)).name;

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={160}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 56, bottom: 4, left: 8 }}
        >
          <XAxis type="number" domain={[0, 1]} hide />
          <YAxis
            type="category"
            dataKey="name"
            width={70}
            tickLine={false}
            axisLine={false}
            tick={{ fill: c.ink2, fontSize: 13, fontWeight: 600 }}
          />
          <Bar dataKey="prob" radius={[4, 4, 4, 4]} barSize={26} isAnimationActive={!reduced} animationDuration={650}>
            {data.map((d) => (
              <Cell
                key={d.name}
                fill={CLASS_COLORS[d.name]}
                fillOpacity={d.name === top ? 1 : 0.32}
              />
            ))}
            <LabelList
              dataKey="prob"
              position="right"
              formatter={(v) => `${(v * 100).toFixed(1)}%`}
              style={{ fill: c.ink2, fontSize: 12, fontWeight: 600 }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
