import { CLASS_NAMES } from "../constants";

// Confusion heatmap, sized to the served model (2x2 or 3x3). Rows = true class, cols = predicted. Cell color is a
// single-hue sequential ramp (magnitude), intensity by row share so the
// diagonal (correct) reads at a glance; the raw count is printed in each cell.
export default function ConfusionMatrixChart({ matrix, title, classNames = CLASS_NAMES }) {
  if (!matrix) return null;
  const rowTotals = matrix.map((row) => row.reduce((a, b) => a + b, 0));

  const cellColor = (v, rowIdx) => {
    const share = rowTotals[rowIdx] ? v / rowTotals[rowIdx] : 0;
    // Accent ramp keyed to the theme's accent RGB (works in light + dark).
    return `rgba(var(--accent-rgb), ${0.08 + 0.82 * share})`;
  };
  const textColor = (v, rowIdx) => {
    const share = rowTotals[rowIdx] ? v / rowTotals[rowIdx] : 0;
    // High-share cells get white ink; low-share cells sit on near-surface, so
    // use the theme's primary ink (readable in both light and dark).
    return share > 0.5 ? "#fff" : "var(--ink)";
  };

  return (
    <div>
      {title && <h3 style={{ marginBottom: 12 }}>{title}</h3>}
      <div className="chart-wrap">
        <div style={{ display: "inline-grid", gridTemplateColumns: `auto repeat(${classNames.length}, 56px)`, gap: 4 }}>
          <div />
          {classNames.map((c) => (
            <div key={c} style={{ textAlign: "center", fontSize: 11, fontWeight: 600, color: "var(--ink-muted)" }}>
              {c}
            </div>
          ))}
          {matrix.map((row, r) => (
            <RowFragment
              key={r}
              label={classNames[r]}
              row={row}
              r={r}
              cellColor={cellColor}
              textColor={textColor}
            />
          ))}
        </div>
      </div>
      <div className="stat-sub" style={{ marginTop: 10 }}>
        Rows = true label · columns = predicted · shade = share of that true class.
      </div>
    </div>
  );
}

function RowFragment({ label, row, r, cellColor, textColor }) {
  return (
    <>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "flex-end",
          paddingRight: 8,
          fontSize: 11,
          fontWeight: 600,
          color: "var(--ink-muted)",
        }}
      >
        {label}
      </div>
      {row.map((v, c) => (
        <div
          key={c}
          title={`True ${label} → predicted ${["Low", "Medium", "High"][c]}: ${v}`}
          style={{
            height: 56,
            borderRadius: 6,
            background: cellColor(v, r),
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontWeight: 700,
            fontSize: 15,
            fontVariantNumeric: "tabular-nums",
            color: textColor(v, r),
            outline: r === c ? "2px solid rgba(var(--accent-rgb), 0.55)" : "none",
            outlineOffset: -2,
          }}
        >
          {v}
        </div>
      ))}
    </>
  );
}
