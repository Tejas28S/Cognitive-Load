import { useRef, useState } from "react";
import ProbabilityBars from "../components/ProbabilityBars";
import ModalityContribution from "../components/ModalityContribution";
import UncertaintyPanel from "../components/UncertaintyPanel";
import ConfidenceGauge from "../components/ConfidenceGauge";
import ExportMenu from "../components/ExportMenu";
import PredictionReport from "../components/PredictionReport";
import { getSubjects, predict } from "../api/client";
import { useApi, Loading, ErrorBox } from "../useApi";
import { useModelVersion, bundleLabel } from "../lib/modelVersion";
import ModelProvenanceNote from "../components/ModelProvenanceNote";
import { CLASS_COLORS } from "../constants";
import { stampFilename } from "../lib/exportData";

const errMsg = (e) =>
  e?.response?.data?.detail || e?.message || "Prediction failed";

function PredictionResultPanel({ title, result, version, classNames, error }) {
  const resultRef = useRef(null);
  if (error) {
    return (
      <div className="card">
        <h2 style={{ fontSize: "1.2rem", marginTop: 0 }}>{title}</h2>
        <div className="error" style={{ margin: 0 }}>{error}</div>
      </div>
    );
  }
  if (!result) return <div className="card muted">No prediction run yet.</div>;

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
        <ProbabilityBars
          probabilities={result.probabilities}
          predicted={result.predicted_label}
          classNames={classNames || undefined}
        />

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

      <PredictionReport result={result} classNames={classNames} label={title} version={version} />
    </div>
  );
}

export default function LivePrediction() {
  // Single-model demo: the header's model switch (2-class / 3-class) picks
  // which bundle this page talks to. No side-by-side comparison anymore -
  // switching the toggle just re-points everything below at the other bundle.
  const { version, active } = useModelVersion();

  const { data: subjects, loading, error } = useApi(
    () => getSubjects(version),
    [version]
  );

  const [subjectId, setSubjectId] = useState(null);
  const [windowIdx, setWindowIdx] = useState(0);

  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  if (loading) return <div className="container"><Loading /></div>;
  if (error) return <div className="container"><ErrorBox error={error} /></div>;

  const subjectOptions = subjects || [];
  const current =
    subjectOptions.find((s) => s.subject_id === subjectId) || subjectOptions[0];
  const activeSubject = current?.subject_id;

  const numWindows = current?.num_windows || 0;
  const w = windowIdx < numWindows ? windowIdx : 0;

  const label = bundleLabel(active) || version;

  const runPredict = async () => {
    if (activeSubject == null) return;
    setBusy(true);
    setErr(null);

    try {
      const r = await predict({ subject_id: activeSubject, window_index: w }, version);
      setResult({ ...r, subject_id: activeSubject, window_index: w });
    } catch (e) {
      setResult(null);
      setErr(errMsg(e));
    }

    setBusy(false);
  };

  return (
    <div className="container" style={{ maxWidth: 1200 }}>
      <div className="page-header">
        <div className="eyebrow eyebrow--accent" style={{ marginBottom: 8 }}>Live demo</div>
        <h1>Live Prediction</h1>
        <p>
          Run the <strong>{label} model</strong> on a real held-out test window.
        </p>
      </div>

      <ModelProvenanceNote style={{ marginBottom: 20 }} />

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="controls-row" style={{ flexWrap: "wrap" }}>
          <div>
            <label className="field">Demo subject</label>
            <select
              value={activeSubject ?? ""}
              onChange={(e) => {
                setSubjectId(Number(e.target.value));
                setWindowIdx(0);
                setResult(null);
                setErr(null);
              }}
            >
              {subjectOptions.map((s) => (
                <option key={s.subject_id} value={s.subject_id}>
                  Subject {s.subject_id}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="field">Window</label>
            <select value={w} onChange={(e) => setWindowIdx(Number(e.target.value))}>
              {Array.from({ length: numWindows }, (_, i) => (
                <option key={i} value={i}>
                  W{i} · truth: {current.true_label_names[i]}
                </option>
              ))}
            </select>
          </div>
          <button className="btn btn-solid" onClick={runPredict} disabled={busy} style={{ alignSelf: "flex-end" }}>
            {busy ? "Predicting…" : "Predict"}
          </button>
        </div>
      </div>

      <div className="grid">
        <PredictionResultPanel
          title={`${label} prediction`}
          result={result}
          error={err}
          version={version}
          classNames={active?.class_names}
        />
      </div>
    </div>
  );
}
