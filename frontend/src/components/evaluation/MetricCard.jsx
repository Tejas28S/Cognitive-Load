// One metric: its value, its name, and a plain-English explanation.
//
// The value always comes from the run's own metrics.json. When a metric is not
// mathematically defined for the run's number of classes (Top-3 on a binary
// task, one-vs-one AUC on two classes), the backend sends the reason and this
// tile shows that reason instead of inventing a number.

const fmtValue = (v, digits) => {
  if (v === null || v === undefined || Number.isNaN(v)) return null;
  return Number(v).toFixed(digits);
};

export default function MetricCard({
  label,
  value,
  explanation,
  unavailableReason,
  digits = 4,
  emphasis = false,
  suffix,
  chance,
}) {
  const shown = fmtValue(value, digits);
  const isUnavailable = shown === null;

  return (
    <div
      className="card"
      style={{
        padding: "14px 16px",
        display: "flex",
        flexDirection: "column",
        gap: 6,
        borderColor: emphasis ? "rgba(var(--accent-rgb), 0.45)" : undefined,
      }}
    >
      <div
        className="stat-label"
        style={{ fontSize: 11, letterSpacing: "0.04em", textTransform: "uppercase" }}
      >
        {label}
      </div>

      {isUnavailable ? (
        <div
          className="muted"
          style={{ fontSize: 12, lineHeight: 1.45, fontStyle: "italic", minHeight: 30 }}
        >
          {unavailableReason || "Not available for this run."}
        </div>
      ) : (
        <div
          className="mono"
          style={{
            fontSize: emphasis ? "1.75rem" : "1.4rem",
            fontWeight: 600,
            fontVariantNumeric: "tabular-nums",
            color: emphasis ? "var(--accent)" : "var(--ink)",
            lineHeight: 1.1,
          }}
        >
          {shown}
          {suffix && (
            <span className="muted" style={{ fontSize: "0.7em", marginLeft: 4 }}>
              {suffix}
            </span>
          )}
        </div>
      )}

      {chance !== undefined && chance !== null && !isUnavailable && (
        <div className="mono muted" style={{ fontSize: 10.5 }}>
          chance {Number(chance).toFixed(digits)}
        </div>
      )}

      {explanation && (
        <div className="muted" style={{ fontSize: 11.5, lineHeight: 1.5 }}>
          {explanation}
        </div>
      )}
    </div>
  );
}
