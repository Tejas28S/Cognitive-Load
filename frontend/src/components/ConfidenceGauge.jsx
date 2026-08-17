// Radial confidence gauge (semi-circular) for a single prediction. The arc fills
// to the model's confidence; the shaded wedge below the abstain threshold marks
// the "not sure enough to commit" zone. Colour shifts teal→amber→red as
// confidence drops toward abstain. Pure SVG, theme-aware, animated draw-in.
const R = 80;
const CX = 100;
const CY = 100;
const CIRC = Math.PI * R; // half-circle arc length

// Point on the semicircle (180°→0°) for a value in [0,1].
function polar(v) {
  const angle = Math.PI * (1 - v); // 0 → left (180°), 1 → right (0°)
  return { x: CX + R * Math.cos(angle), y: CY - R * Math.sin(angle) };
}

export default function ConfidenceGauge({ confidence = 0, abstainThreshold = 0.45, abstain }) {
  const conf = Math.max(0, Math.min(1, confidence));
  const color = abstain ? "var(--stop)" : conf >= 0.6 ? "var(--low)" : "var(--medium)";
  const dash = CIRC * conf;

  const threshPt = polar(abstainThreshold);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      <svg viewBox="0 0 200 120" width="100%" style={{ maxWidth: 260 }} role="img"
        aria-label={`Confidence ${(conf * 100).toFixed(0)} percent`}>
        {/* track */}
        <path d={`M20,100 A${R},${R} 0 0 1 180,100`} fill="none" stroke="var(--border)" strokeWidth="12" strokeLinecap="round" />
        {/* abstain zone (0 → threshold) */}
        <path
          d={`M20,100 A${R},${R} 0 0 1 ${threshPt.x},${threshPt.y}`}
          fill="none"
          stroke="var(--stop)"
          strokeOpacity="0.18"
          strokeWidth="12"
          strokeLinecap="round"
        />
        {/* value arc */}
        <path
          d={`M20,100 A${R},${R} 0 0 1 180,100`}
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={`${dash} ${CIRC}`}
          className="gauge-arc"
        />
        {/* threshold tick */}
        <circle cx={threshPt.x} cy={threshPt.y} r="3" fill="var(--stop)" />
        <text x="100" y="86" textAnchor="middle" fontFamily="var(--font-display)" fontWeight="700" fontSize="26" fill="var(--ink)">
          {(conf * 100).toFixed(0)}%
        </text>
        <text x="100" y="104" textAnchor="middle" fontFamily="var(--font-mono)" fontSize="9" fill="var(--ink-3)" letterSpacing="0.08em">
          CONFIDENCE
        </text>
      </svg>
      <div style={{ fontSize: "0.82rem", color: "var(--ink-2)", marginTop: 2 }}>
        {abstain ? (
          <span style={{ color: "var(--stop)", fontWeight: 600 }}>Below threshold - model abstains</span>
        ) : (
          <span>Abstain threshold {(abstainThreshold * 100).toFixed(0)}%</span>
        )}
      </div>
      <style>{`
        .gauge-arc { transition: stroke-dasharray 0.7s var(--ease); }
        @media (prefers-reduced-motion: reduce) { .gauge-arc { transition: none; } }
      `}</style>
    </div>
  );
}
