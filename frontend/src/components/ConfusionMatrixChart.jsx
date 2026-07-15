import { CLASS_NAMES } from "../constants";

// 3x3 confusion heatmap. Rows = true class, cols = predicted. Cell color is a
// single-hue sequential ramp (magnitude), intensity by row share so the
// diagonal (correct) reads at a glance; the raw count is printed in each cell.
export default function ConfusionMatrixChart({ matrix, title }) {
  if (!matrix) return null;
  const rowTotals = matrix.map((row) => row.reduce((a, b) => a + b, 0));

  const cellColor = (v, rowIdx) => {
    const share = rowTotals[rowIdx] ? v / rowTotals[rowIdx] : 0;
    // indigo ramp: light surface -> #4f46e5
    return `rgba(79, 70, 229, ${0.08 + 0.82 * share})`;
  };
  const textColor = (v, rowIdx) => {
    const share = rowTotals[rowIdx] ? v / rowTotals[rowIdx] : 0;
    return share > 0.5 ? "#fff" : "#1c2530";
  };

  return (
    <div>
      {title && <h3 style={{ marginBottom: 12 }}>{title}</h3>}
      <div className="chart-wrap">
        <div style={{ display: "inline-grid", gridTemplateColumns: "auto repeat(3, 56px)", gap: 4 }}>
          <div />
          {CLASS_NAMES.map((c) => (
            <div key={c} style={{ textAlign: "center", fontSize: 11, fontWeight: 600, color: "var(--ink-muted)" }}>
              {c}
            </div>
          ))}
          {matrix.map((row, r) => (
            <RowFragment
              key={r}
              label={CLASS_NAMES[r]}
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
            outline: r === c ? "2px solid rgba(79,70,229,0.55)" : "none",
            outlineOffset: -2,
          }}
        >
          {v}
        </div>
      ))}
    </>
  );
}
