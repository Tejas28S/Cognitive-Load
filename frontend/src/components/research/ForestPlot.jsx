// Effect-size "forest" plot for the paired LOSO significance tests. Each row is
// one comparison; the marker sits at its mean macro-F1/AUC difference on a shared
// axis centered at zero (the dashed no-effect line). A comparison is called
// significant only when BOTH the paired t-test and Wilcoxon clear p < 0.05 -
// significant markers are filled/accented, non-significant ones are hollow/muted.
// Data: metrics.significance (or research.significance) - a map of test objects.
export default function ForestPlot({ tests, order }) {
  if (!tests) return null;
  const keys = (order || Object.keys(tests)).filter((k) => tests[k]);
  const rows = keys.map((k) => {
    const t = tests[k];
    const sig = t.t_p < 0.05 && t.wilcoxon_p < 0.05;
    return {
      key: k,
      label: t.comparison || k,
      diff: t.mean_diff ?? 0,
      t_p: t.t_p,
      w_p: t.wilcoxon_p,
      n: t.n,
      sig,
    };
  });

  const maxAbs = Math.max(0.02, ...rows.map((r) => Math.abs(r.diff))) * 1.25;
  // Map a diff in [-maxAbs, maxAbs] to a 0–100% track position.
  const pos = (d) => ((d + maxAbs) / (2 * maxAbs)) * 100;

  return (
    <div className="forest">
      <div className="forest-axis">
        <span>← favours reference</span>
        <span className="mono">Δ (mean difference)</span>
        <span>favours variant →</span>
      </div>
      {rows.map((r) => (
        <div key={r.key} className={`forest-row${r.sig ? " is-sig" : ""}`}>
          <div className="forest-label">{r.label}</div>
          <div className="forest-track">
            <span className="forest-zero" />
            <span
              className="forest-marker"
              style={{ left: `${pos(r.diff)}%` }}
              title={`Δ ${r.diff >= 0 ? "+" : ""}${r.diff.toFixed(4)}`}
            />
          </div>
          <div className="forest-stats mono">
            <span className="forest-diff">{r.diff >= 0 ? "+" : ""}{r.diff.toFixed(3)}</span>
            <span className={`forest-badge ${r.sig ? "sig" : "ns"}`}>
              {r.sig ? "significant" : "n.s."}
            </span>
            <span className="forest-p">t p={fmtP(r.t_p)} · W p={fmtP(r.w_p)}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

const fmtP = (p) => (p == null ? "-" : p < 0.001 ? "<0.001" : p.toFixed(3));
