import { useRef, useState } from "react";
import ProbabilityBars from "../components/ProbabilityBars";
import ModalityContribution from "../components/ModalityContribution";
import UncertaintyPanel from "../components/UncertaintyPanel";
import ConfidenceGauge from "../components/ConfidenceGauge";
import ExportMenu from "../components/ExportMenu";
import { getSubjects, predict } from "../api/client";
import { useApi, Loading, ErrorBox } from "../useApi";
import { CLASS_COLORS, CLASS_NAMES } from "../constants";
import { stampFilename } from "../lib/exportData";

function PredictionResultPanel({ title, result, version }) {
  if (!result) return <div className="card muted">No prediction run yet.</div>;
  const resultRef = useRef(null);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div className="card-toolbar">
        <h2 style={{ fontSize: "1.2rem", margin: 0 }}>{title}</h2>
        <ExportMenu
          pngRef={resultRef}
          csvRows={[
            {
              subject_id: result.subject_id,
              window_index: result.window_index,
              predicted_label: result.predicted_label,
              ...Object.fromEntries(
                Object.keys(result.probabilities).map((c) => [`p_${c.toLowerCase()}`, result.probabilities[c]])
              ),
              true_label: result.true_label_name ?? "",
              eeg_share: result.attention_weights?.eeg_share,
              fnirs_share: result.attention_weights?.fnirs_share,
            },
          ]}
          jsonData={result}
          filename={stampFilename([`live-${version}`, `subject-${result.subject_id}`, `w${result.window_index}`])}
        />
      </div>
      <div className="card" ref={resultRef}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 16 }}>
          <span
            className="badge"
            style={{
              background: CLASS_COLORS[result.predicted_label] || "var(--accent)",
              color: "#fff",
              fontSize: "0.95rem",
              padding: "5px 14px",
            }}
          >
            {result.predicted_label}
          </span>
          <span className="muted" style={{ fontSize: "0.85rem" }}>
            Ground truth:{" "}
            <strong style={{ color: CLASS_COLORS[result.true_label_name] || "var(--ink)" }}>
              {result.true_label_name ?? "-"}
            </strong>
          </span>
        </div>
        <ProbabilityBars probabilities={result.probabilities} predicted={result.predicted_label} />
        
        <div style={{ marginTop: 24 }}>
            <h4 style={{ margin: "0 0 8px 0" }}>Confidence & Uncertainty</h4>
            <ConfidenceGauge
              confidence={result.uncertainty?.confidence}
              abstainThreshold={result.uncertainty?.abstain_confidence_threshold}
              abstain={result.uncertainty?.abstain}
            />
            <div style={{ height: 8 }} />
            <UncertaintyPanel uncertainty={result.uncertainty} />
        </div>

        <div style={{ marginTop: 24 }}>
            <h4 style={{ margin: "0 0 8px 0" }}>Modality Attention</h4>
            <ModalityContribution
              eegShare={result.attention_weights?.eeg_share || 0.5}
              fnirsShare={result.attention_weights?.fnirs_share || 0.5}
            />
        </div>
      </div>
    </div>
  );
}

export default function LivePrediction() {
  const { data: subjectsBin, loading: lBin, error: eBin } = useApi(() => getSubjects("binary"), []);
  const { data: subjects3c, loading: l3c, error: e3c } = useApi(() => getSubjects("3class"), []);

  const [subjectId, setSubjectId] = useState(null);
  const [windowIdxBin, setWindowIdxBin] = useState(0);
  const [windowIdx3c, setWindowIdx3c] = useState(0);
  
  const [resultBin, setResultBin] = useState(null);
  const [result3c, setResult3c] = useState(null);
  const [busy, setBusy] = useState(false);
  const [predErr, setPredErr] = useState(null);

  if (lBin || l3c) return <div className="container"><Loading /></div>;
  if (eBin || e3c) return <div className="container"><ErrorBox error={eBin || e3c} /></div>;

  const currentBin = subjectsBin?.find((s) => s.subject_id === subjectId) || subjectsBin[0];
  const current3c = subjects3c?.find((s) => s.subject_id === subjectId) || subjects3c[0];
  
  const activeSubject = currentBin?.subject_id;
  const numWindowsBin = currentBin?.num_windows || 0;
  const numWindows3c = current3c?.num_windows || 0;

  const runPredict = async () => {
    setBusy(true);
    setPredErr(null);
    try {
      const [r3c, rBin] = await Promise.all([
        predict({ subject_id: activeSubject, window_index: windowIdx3c }, "3class"),
        predict({ subject_id: activeSubject, window_index: windowIdxBin }, "binary")
      ]);
      setResult3c({ ...r3c, subject_id: activeSubject, window_index: windowIdx3c });
      setResultBin({ ...rBin, subject_id: activeSubject, window_index: windowIdxBin });
    } catch (e) {
      setPredErr(e?.message || "Prediction failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="container" style={{ maxWidth: 1200 }}>
      <div className="page-header">
        <div className="eyebrow eyebrow--accent" style={{ marginBottom: 8 }}>Live demo</div>
        <h1>Live Prediction Comparison</h1>
        <p>
          Compare the failing <strong>Legacy 3-Class Model</strong> directly against the new <strong>Binary (Low/High) Model</strong> using a real held-out test window from the same subject.
        </p>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="controls-row" style={{ flexWrap: "wrap" }}>
          <div>
            <label className="field">Demo subject (Both Models)</label>
            <select
              value={activeSubject}
              onChange={(e) => {
                setSubjectId(Number(e.target.value));
                setWindowIdxBin(0);
                setWindowIdx3c(0);
                setResultBin(null);
                setResult3c(null);
              }}
            >
              {subjectsBin.map((s) => (
                <option key={s.subject_id} value={s.subject_id}>
                  Subject {s.subject_id}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="field">Window (3-Class Model)</label>
            <select value={windowIdx3c} onChange={(e) => setWindowIdx3c(Number(e.target.value))}>
              {Array.from({ length: numWindows3c }, (_, i) => (
                <option key={i} value={i}>
                  W{i} · truth: {current3c.true_label_names[i]}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="field">Window (Binary Model)</label>
            <select value={windowIdxBin} onChange={(e) => setWindowIdxBin(Number(e.target.value))}>
              {Array.from({ length: numWindowsBin }, (_, i) => (
                <option key={i} value={i}>
                  W{i} · truth: {currentBin.true_label_names[i]}
                </option>
              ))}
            </select>
          </div>
          <button className="btn btn-solid" onClick={runPredict} disabled={busy} style={{ alignSelf: "flex-end" }}>
            {busy ? "Predicting Both…" : "Predict Both"}
          </button>
        </div>
      </div>

      {predErr && <div className="error">{predErr}</div>}

      <div className="grid grid-2">
        <PredictionResultPanel title="Legacy 3-Class Prediction" result={result3c} version="3class" />
        <PredictionResultPanel title="New Binary Prediction" result={resultBin} version="binary" />
      </div>
    </div>
  );
}
