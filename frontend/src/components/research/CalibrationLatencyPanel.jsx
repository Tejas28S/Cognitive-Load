// Compact serving-side panel: calibration (ECE), confidence, and latency for the
// live served bundle (468) vs the HbR prototype (936), both measured through the
// real backend on the demo split. Reinforces that HbR was rejected for
// parsimony, not because it couldn't be served. Data: research.hbr_served_split.
const ROWS = [
  { key: "macro_f1", label: "Macro F1", fmt: (v) => v.toFixed(3), better: "up" },
  { key: "ece", label: "Calibration error (ECE)", fmt: (v) => v.toFixed(3), better: "down" },
  { key: "mean_confidence", label: "Mean confidence", fmt: (v) => v.toFixed(3) },
  { key: "latency_ms_mean", label: "Latency mean", fmt: (v) => `${v.toFixed(1)} ms`, better: "down" },
  { key: "latency_ms_p95", label: "Latency p95", fmt: (v) => `${v.toFixed(1)} ms`, better: "down" },
];

export default function CalibrationLatencyPanel({ served }) {
  const live = served?.live_468;
  const proto = served?.proto_936;
  if (!live) return null;

  return (
    <div className="chart-wrap">
      <table className="data">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Served · EEG+HbO (468)</th>
            {proto && <th>Prototype · +HbR (936)</th>}
          </tr>
        </thead>
        <tbody>
          {ROWS.map((r) => {
            const a = live[r.key];
            const b = proto?.[r.key];
            if (a == null) return null;
            return (
              <tr key={r.key}>
                <td>{r.label}</td>
                <td>{r.fmt(a)}</td>
                {proto && <td>{b == null ? "-" : r.fmt(b)}</td>}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="muted" style={{ fontSize: "0.82rem", marginTop: 8 }}>
        Both bundles run end-to-end through the same backend on the held-out demo
        windows. The 936 prototype is servable but does not clear the decision
        gate - so the leaner 468 model ships.
      </p>
    </div>
  );
}
