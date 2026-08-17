// Dumbbell chart for comparing two metrics.
// Each row is a metric; the connecting bar shows the gap.
export default function GapDumbbell({ 
  rows, 
  domain = [0, 0.7], 
  label1 = "Fixed split (optimistic)", 
  label2 = "LOSO (honest, unseen subject)",
  gapLabel = "leakage gap",
  gapDirection = "fixed-loso"
}) {
  if (!rows?.length) return null;
  const [lo, hi] = domain;
  const pos = (v) => `${((v - lo) / (hi - lo)) * 100}%`;

  return (
    <div className="dumbbell">
      {rows.map((r) => {
        const gap = gapDirection === "loso-fixed" ? r.loso - r.fixed : r.fixed - r.loso;
        return (
          <div key={r.metric} className="dumbbell-row">
            <div className="dumbbell-label">{r.metric}</div>
            <div className="dumbbell-track">
              <span
                className="dumbbell-connect"
                style={{ left: pos(Math.min(r.fixed, r.loso)), width: `calc(${pos(Math.max(r.fixed, r.loso))} - ${pos(Math.min(r.fixed, r.loso))})` }}
              />
              <span className="dumbbell-dot fixed" style={{ left: pos(r.fixed) }} title={`${label1}: ${r.fixed?.toFixed(3)}`} />
              <span className="dumbbell-dot loso" style={{ left: pos(r.loso) }} title={`${label2}: ${r.loso?.toFixed(3)}`} />
            </div>
            <div className="dumbbell-vals mono">
              <span className="dumbbell-gap">{gap > 0 ? "+" : ""}{gap.toFixed(3)}</span>
            </div>
          </div>
        );
      })}
      <div className="dumbbell-legend">
        <span><span className="dumbbell-dot fixed inline" /> {label1}</span>
        <span><span className="dumbbell-dot loso inline" /> {label2}</span>
        <span className="muted">bar = {gapLabel}</span>
      </div>
    </div>
  );
}
