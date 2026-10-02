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

// Recall per class from the model's own 26-fold LOSO evaluation - i.e. "of
// everyone who was genuinely in this band, how often did the model catch it."
// The predicted class is drawn at full opacity so the report reads as "here
// is how much to trust THIS verdict specifically," not just a general table.
export default function ClassRecallChart({ perClass, highlight, classNames = CLASS_NAMES }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  if (!perClass) return null;
  const data = classNames
    .filter((name) => perClass[name]?.recall != null)
    .map((name) => ({ name, recall: perClass[name].recall }));
  if (!data.length) return null;

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={Math.max(90, data.length * 44)}>
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
          <Bar dataKey="recall" radius={[4, 4, 4, 4]} barSize={22} isAnimationActive={!reduced} animationDuration={650}>
            {data.map((d) => (
              <Cell
                key={d.name}
                fill={CLASS_COLORS[d.name]}
                fillOpacity={d.name === highlight ? 1 : 0.32}
              />
            ))}
            <LabelList
              dataKey="recall"
              position="right"
              formatter={(v) => `${(v * 100).toFixed(0)}%`}
              style={{ fill: c.ink2, fontSize: 12, fontWeight: 600 }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
