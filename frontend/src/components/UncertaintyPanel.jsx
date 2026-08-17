import { fmtPct } from "../constants";

// Trustworthiness surface for a single prediction. The served Hybrid+RF
// ensemble naturally exposes confidence, entropy, and - because it has two
// members - how much the Hybrid net and the Random Forest DISAGREE. On an
// honest 3-class LOSO task (~0.44 macro F1) an over-confident single label
// would be misleading; showing uncertainty (and abstaining when the model is
// not sure) is the deployment-grade, trustworthy behavior.
export default function UncertaintyPanel({ uncertainty }) {
  if (!uncertainty) return null;
  const {
    confidence,
    margin,
    entropy,
    disagreement,
    abstain,
    abstain_confidence_threshold: confThr,
  } = uncertainty;

  const barColor = abstain ? "#dc2626" : confidence >= 0.6 ? "#0d9488" : "#d97706";

  const Stat = ({ label, value, hint }) => (
    <div style={{ flex: 1, minWidth: 92 }}>
      <div style={{ fontSize: 12, color: "var(--ink-3)" }}>{label}</div>
      <div style={{ fontSize: "1.05rem", fontWeight: 700, fontFamily: "var(--font-mono)" }}>{value}</div>
      {hint && <div style={{ fontSize: 11, color: "var(--ink-3)" }}>{hint}</div>}
    </div>
  );

  return (
    <div>
      {abstain && (
        <div
          style={{
            display: "flex",
            gap: 8,
            alignItems: "center",
            padding: "8px 12px",
            borderRadius: 8,
            background: "rgba(220, 38, 38, 0.10)",
            border: "1px solid rgba(220, 38, 38, 0.35)",
            marginBottom: 12,
            fontSize: "0.88rem",
          }}
        >
          <span style={{ fontWeight: 700, color: "#dc2626" }}>Low confidence</span>
          <span className="muted">
            below the {fmtPct(confThr)} threshold - treat this as “unsure,” not a
            firm call.
          </span>
        </div>
      )}

      <div style={{ marginBottom: 10 }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
          <span className="muted">Confidence (top class probability)</span>
          <strong>{fmtPct(confidence)}</strong>
        </div>
        <div
          style={{
            height: 10,
            borderRadius: 6,
            background: "var(--border)",
            overflow: "hidden",
            marginTop: 4,
          }}
        >
          <div style={{ width: `${confidence * 100}%`, height: "100%", background: barColor }} />
        </div>
      </div>

      <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
        <Stat label="Top-2 margin" value={fmtPct(margin)} hint="gap to runner-up" />
        <Stat label="Entropy" value={fmtPct(entropy)} hint="0 = certain, 100% = uniform" />
        {disagreement != null && (
          <Stat
            label="Hybrid↔RF disagreement"
            value={fmtPct(disagreement)}
            hint="ensemble members diverging"
          />
        )}
      </div>
    </div>
  );
}
