import { useRef } from "react";
import { Link } from "react-router-dom";
import ExportMenu from "./ExportMenu";
import ProbabilityBars from "./ProbabilityBars";
import ConfusionMatrixChart from "./ConfusionMatrixChart";
import ClassRecallChart from "./ClassRecallChart";
import { CLASS_COLORS, CLASS_TINTS, CLASS_NAMES } from "../constants";
import { stampFilename } from "../lib/exportData";
import { getMetrics, getConfusion } from "../api/client";
import { useApi } from "../useApi";
import {
  REPORT_BASIS,
  MEANING,
  GUIDANCE,
  LIMITATIONS,
  FOOTER_DISCLAIMER,
  recallSentence,
} from "../lib/reportGuidance";

// A structured, exportable summary of one prediction: verdict, plain-language
// meaning, the model's own reliability numbers right next to the verdict (not
// buried in a footnote), workload-management guidance, and limitations shown
// by default. See REPORT_GENERATION_INSTRUCTIONS.md for the framing rules -
// this is a workload-management report, never clinical/diagnostic language.
export default function PredictionReport({ result, classNames, label, version }) {
  const reportRef = useRef(null);
  // Hook before the early return (mirrors LivePrediction's PredictionResultPanel):
  // this component mounts before a prediction exists, so a conditional hook
  // would violate the rules of hooks once `result` later arrives.
  const { data: metrics } = useApi(() => getMetrics(version), [version]);
  const { data: confusion } = useApi(() => getConfusion("hybrid", version), [version]);

  if (!result) return null;

  const cls = result.predicted_label;
  const names = classNames && classNames.length ? classNames : CLASS_NAMES;
  const clsIndex = names.indexOf(cls);
  const meaning = MEANING[cls];
  const guidance = GUIDANCE[cls];
  const confidence = result.uncertainty?.confidence;
  const abstain = Boolean(result.uncertainty?.abstain);

  const loso = metrics?.loso?.hybrid;
  const recall =
    metrics?.per_class?.[cls]?.recall ??
    (clsIndex >= 0 ? loso?.recall_mean?.[clsIndex] : null) ??
    null;
  const overallAcc = loso?.acc_mean;
  const overallAccStd = loso?.acc_std;
  const perSubject = metrics?.per_subject_summary;

  const exportPayload = {
    badge: "Research demo · not a medical device",
    generated_at: new Date().toISOString(),
    subject_id: result.subject_id,
    window_index: result.window_index,
    model: label,
    version,
    predicted_label: cls,
    confidence,
    abstain,
    interpretation: meaning,
    reliability: {
      class_recall: recall,
      overall_loso_accuracy: overallAcc,
      overall_loso_accuracy_std: overallAccStd,
      per_subject_spread: perSubject || null,
      per_class: metrics?.per_class || null,
      confusion_matrix: confusion ? { labels: confusion.labels, matrix: confusion.matrix } : null,
    },
    probabilities: result.probabilities,
    workload_management_guidance: guidance,
    limitations: LIMITATIONS,
    disclaimer: FOOTER_DISCLAIMER,
  };

  const filename = stampFilename([`report-${version}`, `subject-${result.subject_id}`, `w${result.window_index}`]);

  return (
    <div className="report" ref={reportRef}>
      <div className="card-toolbar">
        <div className="report-head">
          <div className="report-head-row">
            <div className="eyebrow eyebrow--accent">Assessment Report</div>
            <span className="report-badge">Research demo · not a medical device</span>
          </div>
          <h2 style={{ margin: "6px 0 4px" }}>Cognitive Load Assessment</h2>
          <div className="report-meta">
            Generated {new Date().toLocaleString()} · Subject {result.subject_id} ·
            {" "}Window {result.window_index} · {label} model ({version}) · LOSO-evaluated
          </div>
        </div>
        <ExportMenu
          pngRef={reportRef}
          pdfRef={reportRef}
          jsonData={exportPayload}
          filename={filename}
        />
      </div>

      {abstain && (
        <div className="note" style={{ marginTop: 16 }}>
          Low-confidence call: this window's prediction fell below the model's own
          confidence threshold. Treat the class below as tentative, not a firm result.
        </div>
      )}

      <div
        className="report-verdict"
        style={{
          marginTop: 16,
          borderLeftColor: CLASS_COLORS[cls] || "var(--accent)",
          background: CLASS_TINTS[cls] || "var(--surface-2)",
        }}
      >
        <div className="report-verdict-main">
          <span className="report-verdict-class" style={{ color: CLASS_COLORS[cls] || "var(--ink)" }}>
            {cls}
          </span>
          {confidence != null && (
            <span className="muted">{(confidence * 100).toFixed(0)}% confidence</span>
          )}
        </div>
        {recall != null && (
          <div className="report-verdict-reliability">{recallSentence(cls, recall)}</div>
        )}
      </div>

      <div className="report-section">
        <h3>Class probabilities</h3>
        <ProbabilityBars
          probabilities={result.probabilities}
          predicted={cls}
          classNames={names}
        />
      </div>

      {meaning && (
        <div className="report-section">
          <h3>What this means</h3>
          <p>{meaning}</p>
        </div>
      )}

      <div className="report-section">
        <h3>What this is based on</h3>
        <p>{REPORT_BASIS}</p>
      </div>

      <div className="report-section">
        <h3>Model reliability</h3>
        <div className="report-stat-row">
          {overallAcc != null && (
            <div className="report-stat">
              <span className="report-stat-num">
                {(overallAcc * 100).toFixed(1)}%
                {overallAccStd != null && (
                  <span className="report-stat-std"> ± {(overallAccStd * 100).toFixed(1)}</span>
                )}
              </span>
              <span className="report-stat-label">Overall cross-subject (LOSO) accuracy</span>
            </div>
          )}
          {recall != null && (
            <div className="report-stat">
              <span className="report-stat-num">{(recall * 100).toFixed(0)}%</span>
              <span className="report-stat-label">Recall for {cls}</span>
            </div>
          )}
          {perSubject && (
            <div className="report-stat">
              <span className="report-stat-num">
                {(perSubject.worst.accuracy * 100).toFixed(0)}–{(perSubject.best.accuracy * 100).toFixed(0)}%
              </span>
              <span className="report-stat-label">Range across {perSubject.n_subjects} subjects</span>
            </div>
          )}
        </div>
        {perSubject && <p className="muted report-stat-note">{perSubject.note}</p>}

        {metrics?.per_class && (
          <div style={{ marginTop: 18 }}>
            <h4 style={{ margin: "0 0 8px 0", fontSize: "0.95rem" }}>Recall by class</h4>
            <ClassRecallChart perClass={metrics.per_class} highlight={cls} classNames={names} />
          </div>
        )}

        {confusion?.matrix && (
          <div style={{ marginTop: 18 }}>
            <ConfusionMatrixChart
              matrix={confusion.matrix}
              classNames={confusion.labels || names}
              title="Confusion matrix (26-fold LOSO, all subjects)"
            />
          </div>
        )}
      </div>

      {guidance && (
        <div className="report-section">
          <h3>Workload-management guidance</h3>
          <ul className="report-list">
            {guidance.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="report-section">
        <h3>Limitations</h3>
        <ul className="report-list report-list--plain">
          {LIMITATIONS.map((item, i) => (
            <li key={i}>{item}</li>
          ))}
        </ul>
      </div>

      <div className="note plain report-disclaimer">
        {FOOTER_DISCLAIMER} See{" "}
        <Link to="/methodology">Methodology</Link> for the model's real cross-subject
        accuracy and limitations.
      </div>
    </div>
  );
}
