import { useState } from "react";
import ProbabilityBars from "../components/ProbabilityBars";
import ModalityContribution from "../components/ModalityContribution";
import UncertaintyPanel from "../components/UncertaintyPanel";
import { getSubjects, predict } from "../api/client";
import { useApi, Loading, ErrorBox } from "../useApi";
import { CLASS_COLORS } from "../constants";

export default function LivePrediction() {
  const { data: subjects, loading, error } = useApi(getSubjects);
  const [subjectId, setSubjectId] = useState(null);
  const [windowIdx, setWindowIdx] = useState(0);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [predErr, setPredErr] = useState(null);

  if (loading) return <div className="container"><Loading /></div>;
  if (error) return <div className="container"><ErrorBox error={error} /></div>;

  const current = subjects.find((s) => s.subject_id === subjectId) || subjects[0];
  const activeSubject = current?.subject_id;
  const numWindows = current?.num_windows || 0;

  const runPredict = async () => {
    setBusy(true);
    setPredErr(null);
    try {
      const r = await predict({ subject_id: activeSubject, window_index: windowIdx });
      setResult(r);
    } catch (e) {
      setPredErr(e?.message || "Prediction failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="container">
      <div className="page-header">
        <div className="eyebrow eyebrow--accent" style={{ marginBottom: 8 }}>Live demo</div>
        <h1>Live Prediction</h1>
        <p>
          Pick a real held-out test window and run it through the served{" "}
          <strong>Hybrid + RF ensemble</strong> (the frozen Hybrid network's
          softmax averaged with the Random Forest). The ground-truth label is
          shown for reference — the model never sees it.
        </p>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="controls-row">
          <div>
            <label className="field">Demo subject</label>
            <select
              value={activeSubject}
              onChange={(e) => {
                setSubjectId(Number(e.target.value));
                setWindowIdx(0);
                setResult(null);
              }}
            >
              {subjects.map((s) => (
                <option key={s.subject_id} value={s.subject_id}>
                  Subject {s.subject_id} ({s.num_windows} windows)
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="field">Window</label>
            <select value={windowIdx} onChange={(e) => setWindowIdx(Number(e.target.value))}>
              {Array.from({ length: numWindows }, (_, i) => (
                <option key={i} value={i}>
                  Window {i} · truth: {current.true_label_names[i]}
                </option>
              ))}
            </select>
          </div>
          <button className="btn" onClick={runPredict} disabled={busy}>
            {busy ? "Predicting…" : "Predict"}
          </button>
        </div>
      </div>

      {predErr && <div className="error">{predErr}</div>}

      {result && (
        <div className="grid grid-2">
          <div className="card">
            <h2>Prediction</h2>
            <p className="card-sub" style={{ marginTop: -4 }}>
              Served model: Hybrid + RF ensemble
            </p>
            <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 16 }}>
              <span
                className="badge"
                style={{
                  background: CLASS_COLORS[result.predicted_label],
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
                  {result.true_label_name ?? "—"}
                </strong>
              </span>
            </div>
            <ProbabilityBars probabilities={result.probabilities} predicted={result.predicted_label} />
          </div>

          <div className="card">
            <h2>Confidence &amp; uncertainty</h2>
            <p className="card-sub">
              How much to trust this call. The Hybrid+RF ensemble exposes its
              own confidence, entropy, and internal disagreement — and abstains
              when it is not sure rather than forcing a low-confidence guess.
            </p>
            <UncertaintyPanel uncertainty={result.uncertainty} />
          </div>

          <div className="card">
            <h2>Modality contribution</h2>
            <p className="card-sub">
              How the Hybrid component's attention gate weighted each
              brain-signal stream for this window (the ensemble's Random Forest
              has no attention).
            </p>
            <ModalityContribution
              eegShare={result.attention_weights.eeg_share}
              fnirsShare={result.attention_weights.fnirs_share}
            />
          </div>
        </div>
      )}

      {!result && !predErr && (
        <div className="muted" style={{ padding: "8px 4px" }}>
          Choose a subject and window, then press <strong>Predict</strong>.
        </div>
      )}
    </div>
  );
}
