// Shared Recharts tooltip, styled with the .chart-tip* classes in index.css so
// it follows the light/dark theme automatically. Pass an optional `format`
// (value) => string and an optional `labelFormat` (label) => string.
export default function ChartTooltip({ active, payload, label, format, labelFormat, unit }) {
  if (!active || !payload || !payload.length) return null;
  const fmt = format || ((v) => (typeof v === "number" ? v.toFixed(3) : v));
  return (
    <div className="chart-tip">
      {label != null && (
        <div className="chart-tip-h">{labelFormat ? labelFormat(label) : label}</div>
      )}
      {payload.map((p, i) => (
        <div className="chart-tip-row" key={i}>
          <span className="chart-tip-dot" style={{ background: p.color || p.fill }} />
          <span className="chart-tip-name">{p.name}</span>
          <span className="chart-tip-val">
            {fmt(p.value)}
            {unit ? ` ${unit}` : ""}
          </span>
        </div>
      ))}
    </div>
  );
}
