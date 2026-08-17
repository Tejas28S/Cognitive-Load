import { useRef, useState } from "react";
import { uploadCsv, uploadMat, uploadTemplateUrl } from "../api/client";
import ExportMenu from "../components/ExportMenu";
import { stampFilename } from "../lib/exportData";
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
  
  // New state for raw mat uploads
  const [eegFile, setEegFile] = useState(null);
  const [nirsFile, setNirsFile] = useState(null);
  const [uploadMode, setUploadMode] = useState("csv"); // 'csv' or 'mat'

  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);
  const eegInputRef = useRef(null);
  const nirsInputRef = useRef(null);

  const onUploadCsv = async () => {
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

  const onUploadMat = async () => {
    if (!eegFile || !nirsFile) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const data = await uploadMat(eegFile, nirsFile);
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
          Upload either a CSV of <strong>already-extracted features</strong>, or raw <strong>.mat recordings</strong> (EEG and fNIRS) to have the features extracted automatically on the server.
        </p>
      </div>

      <div style={{ display: "flex", gap: 16, marginBottom: 20 }}>
        <button className={`btn ${uploadMode === 'csv' ? 'btn-solid' : 'btn-outline'}`} onClick={() => { setUploadMode('csv'); setResult(null); setError(null); }}>Upload Extracted CSV</button>
        <button className={`btn ${uploadMode === 'mat' ? 'btn-solid' : 'btn-outline'}`} onClick={() => { setUploadMode('mat'); setResult(null); setError(null); }}>Upload Raw MATLAB (.mat)</button>
      </div>

      {uploadMode === "csv" && (
        <div className="note" style={{ marginBottom: 20 }}>
          This expects already-extracted features in the project's format
          (<strong>180 EEG + 468 HbO columns</strong>, plus <span className="mono">subject_id</span> and
          optionally <span className="mono">Cognitive_Load</span>). The
          standard HbR (<span className="mono">hbr_*</span>) and <span className="mono">task</span> columns
          may be present - they are ignored by the model.
        </div>
      )}
      
      {uploadMode === "mat" && (
        <div className="note" style={{ marginBottom: 20 }}>
          Upload two continuous recordings (<strong>cnt_nback.mat</strong>) from the TU Berlin dataset - one for EEG and one for fNIRS. The server will dynamically extract sliding windows (Power Spectral Densities and Slopes) before running the prediction. This process can take a few minutes.
        </div>
      )}

      {uploadMode === "csv" && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div 
            className="card-flat" 
            style={{ 
              border: "2px dashed var(--border-strong)", 
              borderRadius: "var(--r)",
              textAlign: "center", 
              padding: "40px 20px",
              cursor: "pointer",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 12,
              transition: "background 0.2s, border-color 0.2s",
              marginBottom: 20
            }}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); e.currentTarget.style.borderColor = "var(--accent)"; e.currentTarget.style.background = "var(--accent-soft)"; }}
            onDragLeave={(e) => { e.preventDefault(); e.currentTarget.style.borderColor = "var(--border-strong)"; e.currentTarget.style.background = "var(--surface-2)"; }}
            onDrop={(e) => { 
              e.preventDefault(); 
              e.currentTarget.style.borderColor = "var(--border-strong)"; 
              e.currentTarget.style.background = "var(--surface-2)";
              if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                setFile(e.dataTransfer.files[0]);
                setResult(null);
                setError(null);
              }
            }}
          >
            <div style={{ background: "var(--surface)", width: 48, height: 48, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "var(--shadow-sm)" }}>
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
            </div>
            <div style={{ fontSize: "1.05rem" }}>
              <span style={{ fontWeight: 600, color: "var(--accent)" }}>Click to browse</span>
              <span style={{ color: "var(--ink-2)", marginLeft: 6 }}>or drag CSV here</span>
            </div>
            {file && (
              <div className="badge" style={{ background: "var(--surface)", border: "1px solid var(--border)", padding: "6px 12px", marginTop: 4 }}>
                Selected: <span style={{ fontWeight: 600, color: "var(--ink)", marginLeft: 4 }}>{file.name}</span>
              </div>
            )}
            <input
              ref={inputRef}
              type="file"
              accept=".csv"
              style={{ display: "none" }}
              onChange={(e) => {
                setFile(e.target.files?.[0] || null);
                setResult(null);
                setError(null);
              }}
            />
          </div>

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 16 }}>
            <button className="btn btn-solid" disabled={!file || busy} onClick={onUploadCsv}>
              {busy ? "Processing CSV…" : "Upload & Predict"}
            </button>
            <a href={uploadTemplateUrl()} className="btn btn-outline" download="upload_template.csv" style={{ fontSize: "0.9rem" }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
              Download CSV Template
            </a>
          </div>
        </div>
      )}

      {uploadMode === "mat" && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="grid grid-2" style={{ gap: 16, marginBottom: 20 }}>
            {/* EEG Upload Block */}
            <div 
              className="card-flat" 
              style={{ 
                border: "2px dashed var(--border-strong)", 
                borderRadius: "var(--r)",
                textAlign: "center", 
                padding: "20px",
                cursor: "pointer",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 8,
                background: "var(--surface-2)"
              }}
              onClick={() => eegInputRef.current?.click()}
            >
              <div style={{ fontWeight: 600, color: "var(--accent)" }}>EEG File (.mat)</div>
              {eegFile ? (
                <div className="badge">{eegFile.name}</div>
              ) : (
                <div style={{ fontSize: "0.9rem", color: "var(--ink-2)" }}>Click to select</div>
              )}
              <input
                ref={eegInputRef}
                type="file"
                accept=".mat"
                style={{ display: "none" }}
                onChange={(e) => {
                  setEegFile(e.target.files?.[0] || null);
                  setResult(null);
                  setError(null);
                }}
              />
            </div>
            
            {/* fNIRS Upload Block */}
            <div 
              className="card-flat" 
              style={{ 
                border: "2px dashed var(--border-strong)", 
                borderRadius: "var(--r)",
                textAlign: "center", 
                padding: "20px",
                cursor: "pointer",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 8,
                background: "var(--surface-2)"
              }}
              onClick={() => nirsInputRef.current?.click()}
            >
              <div style={{ fontWeight: 600, color: "var(--accent)" }}>fNIRS File (.mat)</div>
              {nirsFile ? (
                <div className="badge">{nirsFile.name}</div>
              ) : (
                <div style={{ fontSize: "0.9rem", color: "var(--ink-2)" }}>Click to select</div>
              )}
              <input
                ref={nirsInputRef}
                type="file"
                accept=".mat"
                style={{ display: "none" }}
                onChange={(e) => {
                  setNirsFile(e.target.files?.[0] || null);
                  setResult(null);
                  setError(null);
                }}
              />
            </div>
          </div>

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 16 }}>
            <button className="btn btn-solid" disabled={!eegFile || !nirsFile || busy} onClick={onUploadMat}>
              {busy ? "Extracting & Predicting…" : "Upload & Predict"}
            </button>
          </div>
        </div>
      )}

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
                        ? "- (no label column)"
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
            <div className="card-toolbar">
              <h2 style={{ margin: 0 }}>Per-row predictions</h2>
              <ExportMenu
                csvRows={result.predictions.map((r) => ({
                  row_index: r.row_index,
                  subject_id: r.subject_id ?? "",
                  predicted_label: r.predicted_label,
                  true_label: r.true_label_name ?? "",
                  p_low: r.probabilities?.Low,
                  p_medium: r.probabilities?.Medium,
                  p_high: r.probabilities?.High,
                }))}
                jsonData={result}
                filename={stampFilename(["upload-predictions"])}
              />
            </div>
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
                      <td>{r.subject_id ?? "-"}</td>
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
                        {r.true_label_name ?? "-"}
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
