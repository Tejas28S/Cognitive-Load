import {
  Bar,
  BarChart,
  LabelList,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";
import { CLASS_COLORS } from "../../constants";
import { useChartColors, usePrefersReducedMotion } from "../../lib/chartTheme";

// Small-multiples of the top feature categories driving EACH class, from the
// per-class SHAP/attribution breakdown. Shows that every class leans on
// fNIRS features, but which specific families differ. Data: shap.per_class,
// a dict keyed by whatever class names the served bundle actually predicts
// ("Low (0-back)" / "Medium (2-back)" / "High (3-back)" for 3-class, "Low" /
// "High" for binary) - driven entirely by perClass's own keys, not a
// hardcoded class list, so this renders correctly for either bundle.

// Best-effort short label + color lookup: strips a "(N-back)" suffix if
// present, else uses the key verbatim (binary keys are already short).
function shortLabel(key) {
  return key.replace(/\s*\(.*\)\s*$/, "");
}

export default function ShapPerClass({ perClass, topN = 6 }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();
  if (!perClass) return null;
  const keys = Object.keys(perClass);

  return (
    <div className="grid grid-3" style={{ gap: 16 }}>
      {keys.map((key) => {
        const short = shortLabel(key);
        const list = perClass[key];
        if (!list) return null;
        const data = [...list]
          .sort((a, b) => b.mean_abs_shap - a.mean_abs_shap)
          .slice(0, topN)
          .map((d) => ({ category: d.category, value: d.mean_abs_shap }));
        return (
          <div key={key}>
            <div className="eyebrow" style={{ marginBottom: 6, color: CLASS_COLORS[short] || c.accent }}>
              {short} load
            </div>
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={190}>
                <BarChart data={data} layout="vertical" margin={{ top: 2, right: 42, bottom: 2, left: 4 }}>
                  <XAxis type="number" hide />
                  <YAxis
                    type="category"
                    dataKey="category"
                    width={116}
                    tickLine={false}
                    axisLine={false}
                    tick={{ fill: c.ink2, fontSize: 10.5 }}
                  />
                  <Bar dataKey="value" fill={CLASS_COLORS[short]} radius={[3, 3, 3, 3]} barSize={13} isAnimationActive={!reduced} animationDuration={650}>
                    <LabelList dataKey="value" position="right" formatter={(v) => v.toFixed(2)} style={{ fill: c.axis, fontSize: 10 }} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      })}
    </div>
  );
}
