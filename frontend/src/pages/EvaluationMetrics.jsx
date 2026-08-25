import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import ConfusionMatrixChart from "../components/ConfusionMatrixChart";
import Figure from "../components/research/Figure";
import MetricCard from "../components/evaluation/MetricCard";
import RunSelector, { StatusBadge } from "../components/evaluation/RunSelector";
import {
  ExperimentComparisonChart,
  FeatureImportanceChart,
  PerSubjectChart,
  PrChart,
  RocChart,
  TrainingCurves,
  TsneScatter,
} from "../components/evaluation/EvalCharts";
import {
  getRun,
  getRunConfusion,
  getRunCurves,
  getRunExperiments,
  getRunFeatureImportance,
  getRunHistory,
  getRunMetrics,
  getRunPerSubject,
  getRunRepresentations,
  getRuns,
} from "../api/client";
import { useApi, ErrorBox, Loading } from "../useApi";

const n = (v, d = 4) =>
  v === null || v === undefined || Number.isNaN(v) ? "-" : Number(v).toFixed(d);

// Metric groups. Order and labels are fixed; every VALUE is read from the run.
const OVERALL = [
  ["accuracy", "Accuracy"],
  ["balanced_accuracy", "Balanced accuracy"],
  ["cohen_kappa", "Cohen's Kappa"],
  ["mcc", "Matthews Corr. Coef."],
];
const AVERAGED = [
  ["precision_macro", "Precision, macro"],
  ["recall_macro", "Recall, macro"],
  ["f1_macro", "F1, macro"],
  ["precision_weighted", "Precision, weighted"],
  ["recall_weighted", "Recall, weighted"],
  ["f1_weighted", "F1, weighted"],
  ["precision_micro", "Precision, micro"],
  ["recall_micro", "Recall, micro"],
  ["f1_micro", "F1, micro"],
];
const PROBABILISTIC = [
  ["roc_auc_macro_ovr", "ROC-AUC, macro OVR"],
  ["roc_auc_macro_ovo", "ROC-AUC, macro OVO"],
  ["average_precision_macro", "Average precision, macro"],
  ["log_loss", "Log loss"],
  ["brier_score", "Brier score"],
];

function Section({ id, num, title, lede, children, right }) {
  return (
    <section className="section" id={id} style={{ scrollMarginTop: 90 }}>
      <div className="section-head" style={{ alignItems: "flex-start" }}>
        <div>
          <div className="eyebrow" style={{ marginBottom: 6 }}>
            {num != null ? `${String(num).padStart(2, "0")} · ` : ""}
            {title}
          </div>
          {lede && <p className="lede" style={{ maxWidth: "62ch", marginTop: 0 }}>{lede}</p>}
        </div>
        {right}
      </div>
      {children}
    </section>
  );
}

// `enabled` must be part of the dependency list: it starts false (the run's
// artifact manifest has not loaded yet) and flips true once it has. Without it
// in deps, the fetch would run once while disabled and never re-run.
function useSection(fn, deps, enabled = true) {
  return useApi(() => (enabled ? fn() : Promise.resolve(null)), [...deps, enabled]);
}

export default function EvaluationMetrics() {
  const [params, setParams] = useSearchParams();
  const runsQuery = useApi(() => getRuns({ limit: 200 }), []);
  const runs = useMemo(() => runsQuery.data?.runs || [], [runsQuery.data]);

  const urlRun = params.get("run");
  const [runId, setRunId] = useState(urlRun || "");

  // Default to the newest completed run so the page opens on real results.
  useEffect(() => {
    if (runId || runs.length === 0) return;
    const done = runs.find((r) => r.status === "completed" && r.has_summary);
    setRunId(done?.run_id || runs[0].run_id);
  }, [runs, runId]);

  useEffect(() => {
    if (runId && params.get("run") !== runId) {
      setParams({ run: runId }, { replace: true });
    }
  }, [runId]); // eslint-disable-line react-hooks/exhaustive-deps

  const selected = runs.find((r) => r.run_id === runId);

  const detail = useSection(() => getRun(runId), [runId], !!runId);
  const available = detail.data?.available || {};

  const metrics = useSection(() => getRunMetrics(runId), [runId], !!runId && !!available.metrics);
  const confusion = useSection(() => getRunConfusion(runId), [runId], !!runId && !!available.confusion_matrix);
  const curves = useSection(() => getRunCurves(runId), [runId], !!runId && !!available.curves);
  const history = useSection(() => getRunHistory(runId), [runId], !!runId && !!available.training_history);
  const perSubject = useSection(() => getRunPerSubject(runId), [runId], !!runId && !!available.loso);
  const shap = useSection(() => getRunFeatureImportance(runId), [runId], !!runId && !!available.feature_importance);
  const reps = useSection(() => getRunRepresentations(runId), [runId], !!runId && !!available.representations);
  const experiments = useSection(() => getRunExperiments(runId), [runId], !!runId && !!available.experiments);

  const [tsneVariant, setTsneVariant] = useState("selected_model");
  const [tsneColor, setTsneColor] = useState("y_true_name");

  if (runsQuery.loading) return <div className="container"><Loading /></div>;
  if (runsQuery.error) return <div className="container"><ErrorBox error={runsQuery.error} /></div>;

  const summary = detail.data?.summary || {};
  const dsInfo = detail.data?.dataset_info || {};
  const cfg = detail.data?.config || {};
  const status = detail.data?.status || {};

  // Class names come from the run's own artifacts — never from a constant.
  const classNames =
    confusion.data?.labels ||
    metrics.data?.metrics?.class_names ||
    dsInfo.class_names ||
    [];
  const nClasses = classNames.length;
  const chance = nClasses ? 1 / nClasses : null;

  const m = metrics.data?.metrics;
  const explain = metrics.data?.explanations || {};
  const unavailable = m?.unavailable || {};
  const loso = metrics.data?.loso;
  const foldAgg = loso?.fold_aggregate;
  const lockbox = loso?.lockbox;
  const gap = loso?.generalization_gap;
  const baselineCmp = loso?.baseline_comparison;

  const noRuns = runs.length === 0;

  return (
    <div className="container">
      <div className="page-header">
        <div className="eyebrow eyebrow--accent" style={{ marginBottom: 8 }}>Evaluation</div>
        <h1>Evaluation Metrics</h1>
        <p className="lede">
          Every number on this page is read from the artifacts of the selected run
          under <span className="mono">outputs/runs/</span>. Each metric is computed
          from that run's saved <span className="mono">predictions.csv</span>, so any
          value here can be recomputed from the predictions themselves. Nothing is
          hardcoded in this page.
        </p>
      </div>

      {/* ── run selector ── */}
      <div
        className="card"
        style={{
          display: "flex",
          gap: 24,
          flexWrap: "wrap",
          alignItems: "flex-start",
          justifyContent: "space-between",
          marginBottom: 26,
        }}
      >
        <RunSelector runs={runs} value={runId} onChange={setRunId} />
        {selected && (
          <div style={{ display: "flex", gap: 26, flexWrap: "wrap" }}>
            <Meta label="Classification mode" value={selected.classification_mode} />
            <Meta label="Classes" value={classNames.join(" / ") || "-"} />
            <Meta label="Subjects" value={selected.n_subjects} />
            <Meta label="Windows" value={selected.n_samples?.toLocaleString?.()} />
            <Meta label="Experiments" value={selected.n_experiments} />
          </div>
        )}
      </div>

      {noRuns && (
        <div className="note">
          The run archive is empty. Run{" "}
          <span className="mono">.venv/bin/python run_campaign.py</span> to produce a
          run, then reload this page.
        </div>
      )}

      {status.status === "failed" && (
        <div className="error" style={{ marginBottom: 22 }}>
          This run <strong>failed</strong> during the{" "}
          <span className="mono">{status.stage}</span> stage:{" "}
          {status.error?.message || "no message recorded"}. Its metadata, config and
          logs are preserved; sections below show only what it produced before failing.
        </div>
      )}
      {status.status === "running" && (
        <div className="note" style={{ marginBottom: 22 }}>
          This run is still <strong>in progress</strong> (stage:{" "}
          <span className="mono">{status.stage}</span>). Sections appear as its
          artifacts are written.
        </div>
      )}
      {(status.warnings || []).length > 0 && (
        <div className="note" style={{ marginBottom: 22 }}>
          <strong>Run warnings:</strong>
          <ul style={{ margin: "6px 0 0", paddingLeft: 20 }}>
            {status.warnings.map((w, i) => (
              <li key={i} style={{ fontSize: 12.5 }}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {/* ── 1. protocol header ── */}
      {summary.protocol && (
        <Section
          id="protocol"
          num={1}
          title="Model, protocol and dataset"
          lede="What was evaluated, on which subjects, and under which protocol."
        >
          <div className="grid grid-2">
            <div className="card">
              <h3 style={{ marginTop: 0 }}>Selected model</h3>
              <KV k="Experiment" v={summary.model?.selected_experiment} mono />
              <KV k="Parameters" v={summary.model?.n_params?.toLocaleString?.()} mono />
              <KV k="Encoder" v={summary.model?.spec?.encoder} mono />
              <KV k="Fusion" v={summary.model?.spec?.fusion} mono />
              <KV k="Mean best epoch" v={n(summary.model?.mean_best_epoch, 1)} mono />
              <KV k="Checkpoint metric" v={summary.protocol?.checkpoint_metric} />
            </div>
            <div className="card">
              <h3 style={{ marginTop: 0 }}>Evaluation protocol</h3>
              <KV k="Headline" v={summary.protocol?.headline} />
              <KV k="Selection" v={summary.protocol?.selection} />
              <KV
                k="Lockbox subjects"
                v={(summary.protocol?.lockbox_subjects || []).join(", ")}
                mono
              />
              <KV
                k="Development subjects"
                v={(summary.protocol?.development_subjects || []).join(", ")}
                mono
              />
              <div className="muted" style={{ fontSize: 11.5, marginTop: 10, lineHeight: 1.55 }}>
                The lockbox subjects were never used to pick an architecture,
                hyperparameter, epoch or threshold. They are evaluated once.
              </div>
            </div>
          </div>

          <div className="card" style={{ marginTop: 18 }}>
            <h3 style={{ marginTop: 0 }}>Dataset</h3>
            <div className="grid grid-3">
              <KV k="File" v={dsInfo.csv_basename} mono />
              <KV k="Windows" v={dsInfo.n_samples?.toLocaleString?.()} mono />
              <KV k="Subjects" v={dsInfo.n_subjects} mono />
              <KV k="Trials" v={dsInfo.n_trials} mono />
              <KV k="Window length" v={dsInfo.window_sec ? `${dsInfo.window_sec}s` : "-"} mono />
              <KV
                k="Classes"
                v={(dsInfo.class_names || []).map((c, i) =>
                  `${c} (${dsInfo.class_semantics?.[i] ?? "?"})`).join(", ")}
              />
              <KV k="EEG features" v={dsInfo.feature_schema?.n_eeg} mono />
              <KV k="fNIRS features" v={dsInfo.feature_schema?.n_fnirs} mono />
              <KV
                k="Class distribution"
                v={Object.entries(dsInfo.class_distribution || {})
                  .map(([k2, v]) => `${k2}=${v}`)
                  .join(", ")}
                mono
              />
            </div>
            {dsInfo.preprocessing && (
              <div className="muted mono" style={{ fontSize: 11, marginTop: 12 }}>
                preprocessing: {dsInfo.preprocessing.order} · per-subject norm:{" "}
                {dsInfo.preprocessing.per_subject_norm} (
                {dsInfo.preprocessing.per_subject_norm_stage})
              </div>
            )}
          </div>
        </Section>
      )}

      {/* ── 2. headline: held-out vs validation ── */}
      {foldAgg && (
        <Section
          id="headline"
          num={2}
          title="Held-out performance"
          lede="Cross-subject results: the mean and spread over folds, where each fold's model never saw the evaluated subject. Validation performance is reported separately below, it is what selected the checkpoint, not an estimate of generalization."
        >
          <div className="grid grid-4">
            <MetricCard
              label="Macro F1, held-out"
              value={foldAgg.f1_macro?.mean}
              suffix={`± ${n(foldAgg.f1_macro?.std, 3)}`}
              explanation={explain.f1_macro}
              emphasis
            />
            <MetricCard
              label="Accuracy, held-out"
              value={foldAgg.accuracy?.mean}
              suffix={`± ${n(foldAgg.accuracy?.std, 3)}`}
              explanation={explain.accuracy}
              chance={chance}
            />
            <MetricCard
              label="ROC-AUC, held-out"
              value={foldAgg.roc_auc?.mean}
              suffix={`± ${n(foldAgg.roc_auc?.std, 3)}`}
              explanation={explain.roc_auc_macro_ovr}
            />
            <MetricCard
              label="MCC, held-out"
              value={foldAgg.mcc?.mean}
              suffix={`± ${n(foldAgg.mcc?.std, 3)}`}
              explanation={explain.mcc}
            />
          </div>

          {(lockbox || gap) && (
            <div className="grid grid-2" style={{ marginTop: 18 }}>
              {lockbox && (
                <div className="card">
                  <h3 style={{ marginTop: 0 }}>Lockbox, subjects never used for selection</h3>
                  <p className="muted" style={{ fontSize: 12, marginTop: 0 }}>
                    Subjects {(lockbox.subjects || []).join(", ")}. Evaluated once, after
                    the model was chosen. This is the estimate least affected by
                    selection.
                  </p>
                  <table className="mono" style={{ width: "100%", fontSize: 12 }}>
                    <tbody>
                      {["f1_macro", "accuracy", "balanced_accuracy", "roc_auc", "mcc"].map((k) => (
                        <tr key={k}>
                          <td style={{ padding: "3px 0", color: "var(--ink-2)" }}>
                            {k.replace(/_/g, " ")}
                          </td>
                          <td style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                            {n(lockbox.fold_aggregate?.[k]?.mean)} ±{" "}
                            {n(lockbox.fold_aggregate?.[k]?.std, 3)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {gap && (
                <div className="card">
                  <h3 style={{ marginTop: 0 }}>Generalization gap</h3>
                  <p className="muted" style={{ fontSize: 12, marginTop: 0 }}>
                    Macro F1 at each fold's selected epoch. A large train−held-out gap
                    is the overfitting signal.
                  </p>
                  <table className="mono" style={{ width: "100%", fontSize: 12 }}>
                    <tbody>
                      {[
                        ["train", gap.train_f1_macro_mean],
                        ["validation", gap.val_f1_macro_mean],
                        ["held-out", gap.heldout_f1_macro_mean],
                        ["train − validation", gap.train_minus_val],
                        ["train − held-out", gap.train_minus_heldout],
                        ["validation − held-out", gap.val_minus_heldout],
                      ].map(([k, v]) => (
                        <tr key={k}>
                          <td style={{ padding: "3px 0", color: "var(--ink-2)" }}>{k}</td>
                          <td style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                            {n(v)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {baselineCmp && (
            <div className="card" style={{ marginTop: 18 }}>
              <h3 style={{ marginTop: 0 }}>Selected model vs baseline</h3>
              <p className="muted" style={{ fontSize: 12, marginTop: 0 }}>
                Both trained and evaluated on identical folds, so the comparison is
                paired.
              </p>
              <div style={{ overflowX: "auto" }}>
                <table className="mono" style={{ width: "100%", fontSize: 11.5, borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ textAlign: "left", color: "var(--ink-2)" }}>
                      <th style={{ padding: "6px 8px" }}>Protocol</th>
                      <th style={{ padding: "6px 8px", textAlign: "right" }}>Selected</th>
                      <th style={{ padding: "6px 8px", textAlign: "right" }}>Baseline</th>
                      <th style={{ padding: "6px 8px", textAlign: "right" }}>Δ</th>
                      <th style={{ padding: "6px 8px", textAlign: "right" }}>paired t p</th>
                      <th style={{ padding: "6px 8px", textAlign: "right" }}>Wilcoxon p</th>
                      <th style={{ padding: "6px 8px", textAlign: "right" }}>W/L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ["LOSO (all subjects)", foldAgg?.f1_macro?.mean,
                        baselineCmp.loso26?.f1_macro?.mean, baselineCmp.paired_loso26],
                      ["Lockbox", lockbox?.fold_aggregate?.f1_macro?.mean,
                        baselineCmp.lockbox?.f1_macro?.mean, baselineCmp.paired_lockbox],
                    ].map(([label, a, b, t]) => (
                      <tr key={label} style={{ borderTop: "1px solid var(--rule)" }}>
                        <td style={{ padding: "6px 8px" }}>{label}</td>
                        <td style={{ padding: "6px 8px", textAlign: "right" }}>{n(a)}</td>
                        <td style={{ padding: "6px 8px", textAlign: "right" }}>{n(b)}</td>
                        <td style={{ padding: "6px 8px", textAlign: "right",
                          color: t?.mean_diff > 0 ? "#0f766e" : t?.mean_diff < 0 ? "#b91c1c" : undefined }}>
                          {t?.mean_diff != null ? (t.mean_diff > 0 ? "+" : "") + n(t.mean_diff) : "-"}
                        </td>
                        <td style={{ padding: "6px 8px", textAlign: "right" }}>{n(t?.t_p, 3)}</td>
                        <td style={{ padding: "6px 8px", textAlign: "right" }}>{n(t?.wilcoxon_p, 3)}</td>
                        <td style={{ padding: "6px 8px", textAlign: "right" }}>
                          {t ? `${t.wins}/${t.losses}` : "-"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </Section>
      )}

      {/* ── 3. all metrics ── */}
      {m && (
        <Section
          id="metrics"
          num={3}
          title="Classification metrics"
          lede={`Pooled over every held-out window (${m.n_samples?.toLocaleString?.()} predictions across all folds). Metrics that are not mathematically defined for ${nClasses} classes state why instead of showing a number.`}
        >
          <h3 style={{ marginBottom: 10 }}>Overall</h3>
          <div className="grid grid-4">
            {OVERALL.map(([k, label]) => (
              <MetricCard key={k} label={label} value={m[k]} explanation={explain[k]}
                unavailableReason={unavailable[k]} chance={k === "accuracy" ? chance : undefined} />
            ))}
          </div>

          <h3 style={{ margin: "24px 0 10px" }}>Averaged precision / recall / F1</h3>
          <div className="grid grid-3">
            {AVERAGED.map(([k, label]) => (
              <MetricCard key={k} label={label} value={m[k]} explanation={explain[k]}
                unavailableReason={unavailable[k]} />
            ))}
          </div>

          <h3 style={{ margin: "24px 0 10px" }}>Probability quality</h3>
          <div className="grid grid-3">
            {PROBABILISTIC.map(([k, label]) => (
              <MetricCard key={k} label={label} value={m[k]} explanation={explain[k]}
                unavailableReason={unavailable[k]} />
            ))}
            {[3, 5].map((k) => {
              const key = `top_${k}_accuracy`;
              const val = m.top_k_accuracy?.[`top_${k}`];
              return (
                <MetricCard
                  key={key}
                  label={`Top-${k} accuracy`}
                  value={val}
                  explanation={explain.top_k_accuracy}
                  unavailableReason={unavailable[key]}
                />
              );
            })}
          </div>
        </Section>
      )}

      {/* ── 4. per class ── */}
      {m?.per_class && (
        <Section
          id="per-class"
          num={4}
          title="Per-class metrics"
          lede="Precision, recall, F1 and support for each class, with the raw confusion counts behind them."
        >
          <div className="card" style={{ overflowX: "auto" }}>
            <table className="mono" style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ textAlign: "left", color: "var(--ink-2)" }}>
                  {["Class", "Precision", "Recall", "F1", "Specificity", "Support", "TP", "FP", "FN", "TN"]
                    .map((h, i) => (
                      <th key={h} style={{ padding: "8px", textAlign: i === 0 ? "left" : "right" }}>{h}</th>
                    ))}
                </tr>
              </thead>
              <tbody>
                {m.per_class.map((c) => (
                  <tr key={c.class_index} style={{ borderTop: "1px solid var(--rule)" }}>
                    <td style={{ padding: "8px", fontWeight: 600 }}>{c.class_name}</td>
                    {["precision", "recall", "f1", "specificity"].map((k) => (
                      <td key={k} style={{ padding: "8px", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                        {n(c[k])}
                      </td>
                    ))}
                    {["support", "tp", "fp", "fn", "tn"].map((k) => (
                      <td key={k} style={{ padding: "8px", textAlign: "right", color: "var(--ink-2)" }}>
                        {c[k]}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      {/* ── 5. confusion matrix ── */}
      {confusion.data && (
        <Section
          id="confusion"
          num={5}
          title="Confusion matrix"
          lede={confusion.data.protocol}
        >
          <Figure num={1} title="Confusion matrix, pooled held-out windows"
            source={`outputs/runs/${runId}/confusion_matrix/confusion_matrix.json`}>
            <ConfusionMatrixChart
              matrix={confusion.data.matrix}
              classNames={confusion.data.labels}
            />
          </Figure>
        </Section>
      )}

      {/* ── 6/7. ROC + PR ── */}
      {curves.data && (
        <>
          <Section id="roc" num={6} title="ROC analysis"
            lede="Computed from the saved held-out probabilities across every fold.">
            <Figure num={2} title="ROC curve" source={`outputs/runs/${runId}/curves/curves.json`}>
              <RocChart curves={curves.data} />
            </Figure>
          </Section>
          <Section id="pr" num={7} title="Precision-Recall analysis"
            lede="The dashed line is the positive-rate baseline a random classifier would achieve.">
            <Figure num={3} title="Precision-Recall curve" source={`outputs/runs/${runId}/curves/curves.json`}>
              <PrChart curves={curves.data} />
            </Figure>
          </Section>
        </>
      )}

      {/* ── 8. per subject ── */}
      {perSubject.data?.per_subject?.length > 0 && (
        <Section
          id="subjects"
          num={8}
          title="Per-subject held-out performance"
          lede={`${perSubject.data.protocol}. Each bar is one subject the model never trained on, the spread across subjects is the real measure of cross-person robustness.`}
        >
          <Figure num={4} title="Macro F1 per held-out subject"
            source={`outputs/runs/${runId}/metrics/per_subject_metrics.csv`}>
            <PerSubjectChart
              rows={perSubject.data.per_subject}
              metric="f1_macro"
              chance={chance}
              mean={perSubject.data.fold_aggregate?.f1_macro?.mean}
            />
          </Figure>
          <div className="card" style={{ marginTop: 16, overflowX: "auto" }}>
            <table className="mono" style={{ width: "100%", fontSize: 11.5, borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ textAlign: "left", color: "var(--ink-2)" }}>
                  {["Subject", "Windows", "Accuracy", "Macro F1", "Balanced acc", "Kappa", "MCC", "ROC-AUC", "Mean conf."]
                    .map((h, i) => (
                      <th key={h} style={{ padding: "7px 8px", textAlign: i === 0 ? "left" : "right" }}>{h}</th>
                    ))}
                </tr>
              </thead>
              <tbody>
                {perSubject.data.per_subject.map((r) => (
                  <tr key={r.subject_id} style={{ borderTop: "1px solid var(--rule)" }}>
                    <td style={{ padding: "7px 8px", fontWeight: 600 }}>S{r.subject_id}</td>
                    <td style={{ padding: "7px 8px", textAlign: "right", color: "var(--ink-2)" }}>{r.n_windows}</td>
                    {["accuracy", "f1_macro", "balanced_accuracy", "cohen_kappa", "mcc", "roc_auc", "mean_confidence"]
                      .map((k) => (
                        <td key={k} style={{ padding: "7px 8px", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                          {n(r[k], 3)}
                        </td>
                      ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      {/* ── 9. training curves ── */}
      {history.data && (
        <Section
          id="curves"
          num={9}
          title="Training vs validation curves"
          lede="Averaged over folds. Folds stop at different epochs, so later points average fewer folds, the tooltip reports how many. Validation is what selected each checkpoint; it is not held-out performance."
        >
          <div className="grid grid-2">
            <Figure num={5} title="Loss" source={`outputs/runs/${runId}/curves/training_history.json`}>
              <TrainingCurves history={history.data} metric="loss" label="loss" />
            </Figure>
            <Figure num={6} title="Macro F1" source={`outputs/runs/${runId}/curves/training_history.json`}>
              <TrainingCurves history={history.data} metric="f1_macro" label="macro F1" />
            </Figure>
            <Figure num={7} title="Accuracy" source={`outputs/runs/${runId}/curves/training_history.json`}>
              <TrainingCurves history={history.data} metric="accuracy" label="accuracy" />
            </Figure>
            <Figure num={8} title="Macro recall" source={`outputs/runs/${runId}/curves/training_history.json`}>
              <TrainingCurves history={history.data} metric="recall_macro" label="macro recall" />
            </Figure>
          </div>
        </Section>
      )}

      {/* ── 10. feature importance ── */}
      {shap.data && (
        <Section
          id="shap"
          num={10}
          title="Feature importance"
          lede={`${shap.data.method}, computed on this run's own trained model over ${shap.data.n_explained} held-out windows.`}
        >
          <div className="grid grid-4" style={{ marginBottom: 18 }}>
            {["EEG", "fNIRS", "HbO", "HbR"].map((k) => (
              <MetricCard
                key={k}
                label={`${k} share of |SHAP|`}
                value={shap.data.modality_share?.[k]}
                digits={3}
                explanation={`Fraction of total attribution mass carried by ${k} features.`}
              />
            ))}
          </div>
          <div className="grid grid-2">
            <Figure num={9} title="Top individual features"
              source={`outputs/runs/${runId}/feature_importance/feature_ranking.csv`}>
              <FeatureImportanceChart items={(shap.data.top_features || []).slice(0, 22)} />
            </Figure>
            <Figure num={10} title="Feature families"
              source={`outputs/runs/${runId}/feature_importance/shap_values.json`}>
              <FeatureImportanceChart
                items={(shap.data.categories || []).slice(0, 22)}
                labelKey="category"
              />
            </Figure>
          </div>
        </Section>
      )}

      {/* ── 11. t-SNE ── */}
      {reps.data && (
        <Section
          id="tsne"
          num={11}
          title="Learned representation (t-SNE)"
          lede="Every variant is trained on the same development subjects and embedded on the same unseen subjects, so the pictures differ only by the model. Colour by workload to see class structure; colour by subject to see whether the representation still encodes who the person is."
          right={
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              <select className="mono" value={tsneVariant} onChange={(e) => setTsneVariant(e.target.value)}
                style={selectStyle}>
                {Object.keys(reps.data.variants || {}).map((k) => (
                  <option key={k} value={k}>{k.replace(/_/g, " ")}</option>
                ))}
              </select>
              <select className="mono" value={tsneColor} onChange={(e) => setTsneColor(e.target.value)}
                style={selectStyle}>
                <option value="y_true_name">colour by workload</option>
                <option value="subject_id">colour by subject</option>
              </select>
            </div>
          }
        >
          <Figure num={11} title={`t-SNE, ${tsneVariant.replace(/_/g, " ")}`}
            source={`outputs/runs/${runId}/representations/tsne_${tsneVariant}.csv`}>
            <TsneScatter
              points={reps.data.variants?.[tsneVariant]?.points}
              colorBy={tsneColor}
            />
          </Figure>
          <div className="card" style={{ marginTop: 16 }}>
            <h3 style={{ marginTop: 0 }}>Subject-invariance probe</h3>
            <p className="muted" style={{ fontSize: 12, marginTop: 0 }}>
              A fresh classifier trained to recover subject identity from each
              representation. Accuracy near chance means the representation has
              largely discarded who the person is.
            </p>
            <table className="mono" style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ textAlign: "left", color: "var(--ink-2)" }}>
                  <th style={{ padding: "6px 8px" }}>Representation</th>
                  <th style={{ padding: "6px 8px", textAlign: "right" }}>Probe accuracy</th>
                  <th style={{ padding: "6px 8px", textAlign: "right" }}>Chance</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(reps.data.variants || {}).map(([k, v]) => {
                  const p = v.subject_invariance_probe;
                  return (
                    <tr key={k} style={{ borderTop: "1px solid var(--rule)" }}>
                      <td style={{ padding: "6px 8px" }}>{k.replace(/_/g, " ")}</td>
                      <td style={{ padding: "6px 8px", textAlign: "right" }}>
                        {p ? `${n(p.accuracy_mean, 3)} ± ${n(p.accuracy_std, 3)}` : "-"}
                      </td>
                      <td style={{ padding: "6px 8px", textAlign: "right", color: "var(--ink-2)" }}>
                        {p ? n(p.chance_uniform, 3) : "-"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      {/* ── 12. experiments ── */}
      {experiments.data?.experiments?.length > 0 && (
        <Section
          id="experiments"
          num={12}
          title="Experiment comparison"
          lede="Every arm this run trained, ranked by development-LOSO macro F1. Rejected arms are kept, a negative result is evidence, not clutter."
        >
          <Figure num={12} title="All experiments, development LOSO"
            source={`outputs/runs/${runId}/experiments/leaderboard.csv`}>
            <ExperimentComparisonChart
              experiments={experiments.data.experiments}
              baselineId="A0_baseline"
            />
          </Figure>

          {experiments.data.stage_log?.length > 0 && (
            <div className="card" style={{ marginTop: 18, overflowX: "auto" }}>
              <h3 style={{ marginTop: 0 }}>Stage log, what was accepted and what was rejected</h3>
              <table className="mono" style={{ width: "100%", fontSize: 11.5, borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ textAlign: "left", color: "var(--ink-2)" }}>
                    <th style={{ padding: "6px 8px" }}>Stage</th>
                    <th style={{ padding: "6px 8px" }}>Winner</th>
                    <th style={{ padding: "6px 8px" }}>Changed</th>
                    <th style={{ padding: "6px 8px", textAlign: "right" }}>Before</th>
                    <th style={{ padding: "6px 8px", textAlign: "right" }}>After</th>
                    <th style={{ padding: "6px 8px", textAlign: "right" }}>Δ</th>
                  </tr>
                </thead>
                <tbody>
                  {experiments.data.stage_log.map((s) => (
                    <tr key={s.stage} style={{ borderTop: "1px solid var(--rule)" }}>
                      <td style={{ padding: "6px 8px" }}>{s.stage}</td>
                      <td style={{ padding: "6px 8px" }}>{s.winner}</td>
                      <td style={{ padding: "6px 8px" }}>{s.changed ? "accepted" : "rejected"}</td>
                      <td style={{ padding: "6px 8px", textAlign: "right" }}>{n(s.score_before)}</td>
                      <td style={{ padding: "6px 8px", textAlign: "right" }}>{n(s.score_after)}</td>
                      <td style={{ padding: "6px 8px", textAlign: "right",
                        color: s.delta > 0 ? "#0f766e" : undefined }}>
                        {s.delta > 0 ? "+" : ""}{n(s.delta)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="card" style={{ marginTop: 18, overflowX: "auto" }}>
            <h3 style={{ marginTop: 0 }}>Leaderboard</h3>
            <table className="mono" style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ textAlign: "left", color: "var(--ink-2)" }}>
                  {["Experiment", "Stage", "Macro F1", "Accuracy", "AUC", "MCC", "Params", "Gap", "vs base", "p"]
                    .map((h, i) => (
                      <th key={h} style={{ padding: "6px 8px", textAlign: i < 2 ? "left" : "right" }}>{h}</th>
                    ))}
                </tr>
              </thead>
              <tbody>
                {experiments.data.experiments.map((e) => (
                  <tr key={e.experiment_id} style={{ borderTop: "1px solid var(--rule)" }}>
                    <td style={{ padding: "6px 8px" }} title={e.note}>{e.experiment_id}</td>
                    <td style={{ padding: "6px 8px", color: "var(--ink-2)" }}>{e.stage}</td>
                    <td style={{ padding: "6px 8px", textAlign: "right", fontWeight: 600 }}>
                      {n(e.f1_macro_mean)}
                    </td>
                    <td style={{ padding: "6px 8px", textAlign: "right" }}>{n(e.accuracy_mean, 3)}</td>
                    <td style={{ padding: "6px 8px", textAlign: "right" }}>{n(e.roc_auc_mean, 3)}</td>
                    <td style={{ padding: "6px 8px", textAlign: "right" }}>{n(e.mcc_mean, 3)}</td>
                    <td style={{ padding: "6px 8px", textAlign: "right", color: "var(--ink-2)" }}>
                      {e.n_params?.toLocaleString?.()}
                    </td>
                    <td style={{ padding: "6px 8px", textAlign: "right", color: "var(--ink-2)" }}>
                      {n(e.generalization_gap?.train_minus_heldout, 3)}
                    </td>
                    <td style={{ padding: "6px 8px", textAlign: "right",
                      color: e.vs_baseline?.mean_diff > 0 ? "#0f766e" : e.vs_baseline?.mean_diff < 0 ? "#b91c1c" : undefined }}>
                      {e.vs_baseline?.mean_diff != null
                        ? (e.vs_baseline.mean_diff > 0 ? "+" : "") + n(e.vs_baseline.mean_diff, 3)
                        : "-"}
                    </td>
                    <td style={{ padding: "6px 8px", textAlign: "right", color: "var(--ink-2)" }}>
                      {n(e.vs_baseline?.t_p, 3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      {/* ── 13. run details ── */}
      {detail.data && (
        <Section
          id="details"
          num={13}
          title="Run details"
          lede="Exactly how this run was produced, so it can be reproduced."
        >
          <div className="grid grid-2">
            <div className="card">
              <h3 style={{ marginTop: 0 }}>Provenance</h3>
              <KV k="Run ID" v={runId} mono />
              <KV k="Status" v={<StatusBadge status={status.status} stale={selected?.stale} />} />
              <KV k="Created" v={status.created_at} mono />
              <KV k="Git commit" v={detail.data.metadata?.git?.commit?.slice(0, 12)} mono />
              <KV k="Git branch" v={detail.data.metadata?.git?.branch} mono />
              <KV k="Working tree" v={detail.data.metadata?.git?.dirty ? "dirty" : "clean"} mono />
              <KV k="Seed" v={cfg.seed} mono />
              <KV k="Torch" v={detail.data.metadata?.environment?.torch} mono />
              <KV k="Python" v={detail.data.metadata?.environment?.python} mono />
              <KV k="Platform" v={detail.data.metadata?.environment?.platform} mono />
            </div>
            <div className="card">
              <h3 style={{ marginTop: 0 }}>Leakage audit</h3>
              <p className="muted" style={{ fontSize: 12, marginTop: 0 }}>
                Assertions run before every fold trained. A failure aborts the run
                rather than producing a score.
              </p>
              {(detail.data.leakage_audit?.dataset_level?.checks || []).map((c) => (
                <div key={c.check} style={{ display: "flex", gap: 8, alignItems: "flex-start", marginBottom: 7 }}>
                  <span className="mono" style={{
                    color: c.passed ? "#0f766e" : "#b91c1c", fontWeight: 700, fontSize: 11,
                  }}>
                    {c.passed ? "PASS" : "FAIL"}
                  </span>
                  <span style={{ fontSize: 11.5 }}>
                    <span className="mono">{c.check}</span>
                    <span className="muted">, {c.detail}</span>
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="card" style={{ marginTop: 18 }}>
            <h3 style={{ marginTop: 0 }}>Configuration</h3>
            <pre className="mono" style={{
              fontSize: 11, overflowX: "auto", background: "var(--surface-2, transparent)",
              padding: 12, borderRadius: 8, margin: 0, maxHeight: 420,
            }}>
              {JSON.stringify(summary.model?.spec || cfg.baseline_spec || cfg, null, 2)}
            </pre>
          </div>
        </Section>
      )}

      {detail.loading && <Loading />}
    </div>
  );
}

const selectStyle = {
  padding: "6px 9px",
  borderRadius: 8,
  border: "1px solid var(--rule)",
  background: "var(--surface)",
  color: "var(--ink)",
  fontSize: 11.5,
};

function Meta({ label, value }) {
  return (
    <div>
      <div className="stat-label" style={{ fontSize: 10.5, textTransform: "uppercase", letterSpacing: "0.04em" }}>
        {label}
      </div>
      <div className="mono" style={{ fontSize: 13, marginTop: 2 }}>{value ?? "-"}</div>
    </div>
  );
}

function KV({ k, v, mono }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 14, padding: "4px 0", fontSize: 12.5 }}>
      <span className="muted" style={{ flexShrink: 0 }}>{k}</span>
      <span className={mono ? "mono" : ""} style={{ textAlign: "right", wordBreak: "break-word" }}>
        {v ?? "-"}
      </span>
    </div>
  );
}
