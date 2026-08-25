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

function PredictionResultPanel({ title, result, version, classNames }) {
  // Hook before the early return: this panel now flips between "no result" and
  // "result" as the model switch changes, so a conditional hook would break.
  const resultRef = useRef(null);
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
  // The switch picks the primary model. If a second bundle is also servable it
  // is shown alongside, so the page works as a single-model demo or as the
  // 2-class vs 3-class comparison depending on what the backend has.
  const { version, bundles, available } = useModelVersion();
  const activeBundle = bundles.find((b) => b.version === version) || null;
  const otherBundle = available.find((b) => b.version !== version) || null;
  const otherVersion = otherBundle?.version || null;

  const { data: subjectsA, loading: lA, error: eA } = useApi(
    () => getSubjects(version),
    [version]
  );
  const { data: subjectsB, loading: lB, error: eB } = useApi(
    () => (otherVersion ? getSubjects(otherVersion) : Promise.resolve(null)),
    [otherVersion]
  );

  const [subjectId, setSubjectId] = useState(null);
  const [windowIdxA, setWindowIdxA] = useState(0);
  const [windowIdxB, setWindowIdxB] = useState(0);

  const [resultA, setResultA] = useState(null);
  const [resultB, setResultB] = useState(null);
  const [busy, setBusy] = useState(false);
  const [predErr, setPredErr] = useState(null);

  if (lA || lB) return <div className="container"><Loading /></div>;
  if (eA) return <div className="container"><ErrorBox error={eA} /></div>;

  const hasOther = Boolean(otherVersion) && Boolean(subjectsB) && !eB;

  const currentA = subjectsA?.find((s) => s.subject_id === subjectId) || subjectsA[0];
  const currentB = subjectsB?.find((s) => s.subject_id === subjectId) || subjectsB?.[0];

  const activeSubject = currentA?.subject_id;
  const numWindowsA = currentA?.num_windows || 0;
  const numWindowsB = currentB?.num_windows || 0;

  const labelA = bundleLabel(activeBundle) || version;
  const labelB = bundleLabel(otherBundle) || otherVersion;

  const runPredict = async () => {
    setBusy(true);
    setPredErr(null);
    try {
      const [rA, rB] = await Promise.all([
        predict({ subject_id: activeSubject, window_index: windowIdxA }, version),
        hasOther
          ? predict({ subject_id: activeSubject, window_index: windowIdxB }, otherVersion)
          : null,
      ]);
      setResultA({ ...rA, subject_id: activeSubject, window_index: windowIdxA });
      if (rB) setResultB({ ...rB, subject_id: activeSubject, window_index: windowIdxB });
      else setResultB(null);
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
        <h1>{hasOther ? "Live Prediction Comparison" : "Live Prediction"}</h1>
        <p>
          {hasOther ? (
            <>
              Run the <strong>{labelA} model</strong> against the <strong>{labelB} model</strong> on
              a real held-out test window from the same subject. The model switch decides which one
              leads.
            </>
          ) : (
            <>
              Run the <strong>{labelA} model</strong> on a real held-out test window. No second
              bundle is available to compare against.
            </>
          )}
        </p>
      </div>

      <ModelProvenanceNote style={{ marginBottom: 20 }} />

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="controls-row" style={{ flexWrap: "wrap" }}>
          <div>
            <label className="field">{hasOther ? "Demo subject (both models)" : "Demo subject"}</label>
            <select
              value={activeSubject}
              onChange={(e) => {
                setSubjectId(Number(e.target.value));
                setWindowIdxA(0);
                setWindowIdxB(0);
                setResultA(null);
                setResultB(null);
              }}
            >
              {subjectsA.map((s) => (
                <option key={s.subject_id} value={s.subject_id}>
                  Subject {s.subject_id}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="field">{hasOther ? `Window (${labelA})` : "Window"}</label>
            <select value={windowIdxA} onChange={(e) => setWindowIdxA(Number(e.target.value))}>
              {Array.from({ length: numWindowsA }, (_, i) => (
                <option key={i} value={i}>
                  W{i} · truth: {currentA.true_label_names[i]}
                </option>
              ))}
            </select>
          </div>
          {hasOther && (
            <div>
              <label className="field">Window ({labelB})</label>
              <select value={windowIdxB} onChange={(e) => setWindowIdxB(Number(e.target.value))}>
                {Array.from({ length: numWindowsB }, (_, i) => (
                  <option key={i} value={i}>
                    W{i} · truth: {currentB.true_label_names[i]}
                  </option>
                ))}
              </select>
            </div>
          )}
          <button className="btn btn-solid" onClick={runPredict} disabled={busy} style={{ alignSelf: "flex-end" }}>
            {busy ? (hasOther ? "Predicting both…" : "Predicting…") : hasOther ? "Predict both" : "Predict"}
          </button>
        </div>
      </div>

      {predErr && <div className="error">{predErr}</div>}

      <div className={hasOther ? "grid grid-2" : "grid"}>
        <PredictionResultPanel
          title={`${labelA} prediction`}
          result={resultA}
          version={version}
          classNames={activeBundle?.class_names}
        />
        {hasOther && (
          <PredictionResultPanel
            title={`${labelB} prediction`}
            result={resultB}
            version={otherVersion}
            classNames={otherBundle?.class_names}
          />
        )}
      </div>
    </div>
  );
}
