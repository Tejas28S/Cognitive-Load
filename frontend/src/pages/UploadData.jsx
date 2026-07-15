import { useRef, useState } from "react";
import { uploadCsv, uploadTemplateUrl } from "../api/client";
import { CLASS_COLORS, CLASS_NAMES } from "../constants";

// Compact 3-segment probability bar for a table row.
function MiniProb({ probabilities }) {
  return (
    <div style={{ display: "flex", height: 14, borderRadius: 4, overflow: "hidden", minWidth: 120 }}>
      {CLASS_NAMES.map((c) => (
        <div
          key={c}
          title={`${c}: ${((probabilities[c] ?? 0) * 100).toFixed(1)}%`}
          style={{ width: `${(probabilities[c] ?? 0) * 100}%`, background: CLASS_COLORS[c] }}
        />
      ))}
    </div>
  );
}

function ClassDistribution({ dist, total }) {
  return (
    <div style={{ display: "grid", gap: 8 }}>
      {CLASS_NAMES.map((c) => {
        const n = dist[c] ?? 0;
        const pct = total ? (n / total) * 100 : 0;
        return (
          <div key={c} style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ width: 70, fontSize: 13, fontWeight: 600, color: CLASS_COLORS[c] }}>{c}</span>
            <div style={{ flex: 1, height: 18, background: "var(--surface-2)", borderRadius: 5, overflow: "hidden" }}>
              <div style={{ width: `${pct}%`, height: "100%", background: CLASS_COLORS[c] }} />
            </div>
            <span style={{ width: 74, textAlign: "right", fontSize: 13, fontVariantNumeric: "tabular-nums" }}>
              {n} ({pct.toFixed(0)}%)
            </span>
          </div>
        );
      })}
    </div>
  );
}

export default function UploadData() {
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const onUpload = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const data = await uploadCsv(file);
      setResult(data);
    } catch (e) {
      const detail = e?.response?.data?.detail;
      if (detail && typeof detail === "object") {
        setError({
          message: detail.message,
          missing: detail.missing_required || [],
          unexpected: detail.unexpected_columns || [],
        });
      } else {
        setError({ message: detail || e?.message || "Upload failed", missing: [], unexpected: [] });
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="container">
      <div className="page-header">
        <div className="eyebrow eyebrow--accent" style={{ marginBottom: 8 }}>Live demo</div>
        <h1>Upload Data</h1>
        <p>
          Upload a CSV of <strong>already-extracted TU Berlin features</strong> and
          get an ensemble prediction for every row.
        </p>
      </div>

      <div className="note" style={{ marginBottom: 20 }}>
        This expects already-extracted features in the project's format
        (<strong>180 EEG + 468 HbO columns</strong>, plus <span className="mono">subject_id</span> and
        optionally <span className="mono">Cognitive_Load</span>) — <strong>not raw sensor
        readings</strong>. Feature extraction from a different dataset or device
        (e.g. a Stroop EEG+fNIRS recording) would need a separate,
        dataset-specific pipeline; that is not currently supported. The
        standard HbR (<span className="mono">hbr_*</span>) and <span className="mono">task</span> columns
        may be present — they are ignored by the model.
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="controls-row">
          <div>
            <label className="field">CSV file</label>
            <input
              ref={inputRef}
              type="file"
              accept=".csv"
              onChange={(e) => {
                setFile(e.target.files?.[0] || null);
                setResult(null);
                setError(null);
              }}
            />
          </div>
          <button className="btn" onClick={onUpload} disabled={!file || busy}>
            {busy ? "Predicting…" : "Upload & predict"}
          </button>
          <a href={uploadTemplateUrl} download style={{ fontSize: "0.88rem", fontWeight: 600 }}>
            ↓ Download example CSV
          </a>
        </div>
        <p className="card-sub" style={{ marginTop: 12, marginBottom: 0 }}>
          Not sure of the format? The example CSV shows the exact expected
          columns with a few real rows. Predictions come from the served
          Hybrid + RF ensemble.
        </p>
      </div>

      {error && (
        <div className="card" style={{ borderColor: "#fca5a5", marginBottom: 20 }}>
          <h3 style={{ color: "#b91c1c", marginTop: 0 }}>Upload rejected</h3>
          <p style={{ color: "var(--ink-2)" }}>{error.message}</p>
          {error.missing.length > 0 && (
            <div>
              <div style={{ fontWeight: 600, fontSize: "0.85rem", marginBottom: 4 }}>
                Missing required columns ({error.missing.length} shown up to 50):
              </div>
              <div className="mono" style={{ fontSize: "0.78rem", color: "#b91c1c", wordBreak: "break-word" }}>
                {error.missing.join(", ")}
              </div>
            </div>
          )}
        </div>
      )}

      {result && (
        <>
          <div className="card" style={{ marginBottom: 20 }}>
            <h2>Summary</h2>
            <div className="grid grid-2" style={{ alignItems: "start" }}>
              <div>
                <div style={{ display: "grid", gap: 6, marginBottom: 12 }}>
                  <Row label="Rows processed" value={result.summary.rows_processed} />
                  <Row
                    label="Accuracy vs uploaded labels"
                    value={
                      result.summary.accuracy_vs_uploaded_labels == null
                        ? "— (no label column)"
                        : `${(result.summary.accuracy_vs_uploaded_labels * 100).toFixed(1)}%`
                    }
                  />
                  <Row label="Served model" value="Hybrid + RF ensemble" />
                </div>
                {result.normalization !== "per_subject" && (
                  <div className="note" style={{ fontSize: "0.82rem" }}>
                    {result.normalization_note}
                  </div>
                )}
                {result.summary.unexpected_columns?.length > 0 && (
                  <div className="note info" style={{ fontSize: "0.8rem", marginTop: 10 }}>
                    {result.summary.unexpected_columns.length} unrecognized column(s)
                    were ignored (e.g. {result.summary.unexpected_columns.slice(0, 3).join(", ")}
                    …).
                  </div>
                )}
              </div>
              <div>
                <div style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--ink-2)", marginBottom: 10 }}>
                  Predicted class distribution
                </div>
                <ClassDistribution
                  dist={result.summary.class_distribution}
                  total={result.summary.rows_processed}
                />
              </div>
            </div>
          </div>

          <div className="card">
            <h2>Per-row predictions</h2>
            <div className="chart-wrap" style={{ maxHeight: 460, overflowY: "auto" }}>
              <table className="data">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Subject</th>
                    <th>Predicted</th>
                    <th>Probabilities [L·M·H]</th>
                    <th>True label</th>
                  </tr>
                </thead>
                <tbody>
                  {result.predictions.map((r) => (
                    <tr key={r.row_index}>
                      <td>{r.row_index}</td>
                      <td>{r.subject_id ?? "—"}</td>
                      <td style={{ textAlign: "left" }}>
                        <span
                          className="badge"
                          style={{ background: CLASS_COLORS[r.predicted_label], color: "#fff" }}
                        >
                          {r.predicted_label}
                        </span>
                      </td>
                      <td style={{ textAlign: "left" }}>
                        <MiniProb probabilities={r.probabilities} />
                      </td>
                      <td
                        style={{
                          textAlign: "left",
                          color: r.true_label_name ? CLASS_COLORS[r.true_label_name] : "var(--ink-muted)",
                          fontWeight: r.true_label_name ? 600 : 400,
                        }}
                      >
                        {r.true_label_name ?? "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.9rem", padding: "4px 0" }}>
      <span className="muted">{label}</span>
      <strong style={{ fontVariantNumeric: "tabular-nums" }}>{value}</strong>
    </div>
  );
}
