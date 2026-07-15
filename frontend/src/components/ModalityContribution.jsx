import { MODALITY_COLORS } from "../constants";

// Two-segment bar: the model's learned gate weight on each modality's
// representation. This is "how much the model trusts each stream," NOT "which
// modality is more accurate" — a distinct thing (see the label copy below).
export default function ModalityContribution({ eegShare, fnirsShare, compact }) {
  const eeg = eegShare ?? 0.5;
  const fnirs = fnirsShare ?? 0.5;
  const total = eeg + fnirs || 1;
  const eegPct = (eeg / total) * 100;
  const fnirsPct = (fnirs / total) * 100;

  return (
    <div>
      <div
        style={{
          display: "flex",
          height: 26,
          borderRadius: 8,
          overflow: "hidden",
          border: "1px solid var(--border)",
        }}
      >
        <div
          style={{
            width: `${eegPct}%`,
            background: MODALITY_COLORS.eeg,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#fff",
            fontSize: 12,
            fontWeight: 600,
          }}
        >
          {eegPct >= 14 ? `EEG ${eegPct.toFixed(0)}%` : ""}
        </div>
        <div
          style={{
            width: `${fnirsPct}%`,
            background: MODALITY_COLORS.fnirs,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#fff",
            fontSize: 12,
            fontWeight: 600,
          }}
        >
          {fnirsPct >= 14 ? `fNIRS ${fnirsPct.toFixed(0)}%` : ""}
        </div>
      </div>
      {!compact && (
        <div className="stat-sub" style={{ marginTop: 8 }}>
          The model's learned trust in each modality's representation — not a
          claim about which sensor is more accurate.
        </div>
      )}
    </div>
  );
}
