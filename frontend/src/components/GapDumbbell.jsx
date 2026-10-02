// Dumbbell chart for a before/after comparison, one row per metric.
//
// The two ends are two shades of ONE hue, not two different hues: this is the
// same quantity measured twice, not two categories, and the shading is what
// makes "it moved right" the first thing you read. The track carries a real
// tick scale, because a bar whose length means nothing is decoration.
//
// Both endpoint values are printed beside every row. That is deliberate - the
// pale end of the ramp sits under 3:1 against the surface, so the numbers, not
// the fill, are what carry the value.
export default function GapDumbbell({
  rows,
  domain = [0, 0.7],
  label1 = "Fixed split (optimistic)",
  label2 = "LOSO (honest, unseen subject)",
  gapLabel = "leakage gap",
  gapDirection = "fixed-loso",
  /** "gain" - a positive delta is good. "penalty" - a positive delta is bad. */
  gapTone = "penalty",
  ticks,
  decimals = 3,
}) {
  if (!rows?.length) return null;
  const [lo, hi] = domain;
  const span = hi - lo || 1;
  // Clamped, so a value outside the domain pins to the edge instead of
  // escaping the track and littering the card with orphaned dots.
  const pct = (v) => Math.min(100, Math.max(0, ((v - lo) / span) * 100));
  const has = (v) => typeof v === "number" && Number.isFinite(v);

  const tickVals =
    ticks || Array.from({ length: 5 }, (_, i) => lo + (span * i) / 4);

  return (
    <div className="dumbbell">
      <div className="dumbbell-plot">
        {rows.map((r) => {
          if (!has(r.fixed) || !has(r.loso)) return null;
          const gap = gapDirection === "loso-fixed" ? r.loso - r.fixed : r.fixed - r.loso;
          const good = gapTone === "gain" ? gap > 0 : gap < 0;
          const a = pct(r.fixed);
          const b = pct(r.loso);
          return (
            <div key={r.metric} className="dumbbell-row">
              <div className="dumbbell-label">{r.metric}</div>
              <div className="dumbbell-track">
                {tickVals.map((t) => (
                  <span key={t} className="dumbbell-grid" style={{ left: `${pct(t)}%` }} />
                ))}
                <span
                  className="dumbbell-connect"
                  style={{ left: `${Math.min(a, b)}%`, width: `${Math.abs(b - a)}%` }}
                />
                <span
                  className="dumbbell-dot is-before"
                  style={{ left: `${a}%` }}
                  title={`${label1}: ${r.fixed.toFixed(decimals)}`}
                />
                <span
                  className="dumbbell-dot is-after"
                  style={{ left: `${b}%` }}
                  title={`${label2}: ${r.loso.toFixed(decimals)}`}
                />
              </div>
              <div className="dumbbell-vals">
                <span className="dumbbell-from">{r.fixed.toFixed(decimals)}</span>
                <span className="dumbbell-arrow" aria-hidden="true">→</span>
                <span className="dumbbell-to">{r.loso.toFixed(decimals)}</span>
              </div>
              <div className={`dumbbell-gap ${good ? "is-good" : "is-bad"}`}>
                {gap > 0 ? "+" : ""}
                {gap.toFixed(decimals)}
              </div>
            </div>
          );
        })}

        {/* Scale, aligned to the track column so the ticks sit under the marks. */}
        <div className="dumbbell-row dumbbell-axis">
          <div />
          <div className="dumbbell-track">
            {tickVals.map((t) => (
              <span key={t} className="dumbbell-tick" style={{ left: `${pct(t)}%` }}>
                {t.toFixed(2)}
              </span>
            ))}
          </div>
          <div />
          <div />
        </div>
      </div>

      <div className="dumbbell-legend">
        <span><i className="dumbbell-key is-before" /> {label1}</span>
        <span><i className="dumbbell-key is-after" /> {label2}</span>
        <span className="muted">bar = {gapLabel}</span>
      </div>
    </div>
  );
}
