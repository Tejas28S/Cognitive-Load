import ConfusionMatrixChart from "../components/ConfusionMatrixChart";
import ShapChart from "../components/ShapChart";
import ModalityContribution from "../components/ModalityContribution";
import Figure from "../components/research/Figure";
import LosoConfigChart from "../components/research/LosoConfigChart";
import PerClassRecall from "../components/research/PerClassRecall";
import ShapPerClass from "../components/research/ShapPerClass";
import PerSubjectBars from "../components/research/PerSubjectBars";
import { getConfusion, getShap, getAttention, getMetrics, getDiagnostics } from "../api/client";
import { useApi, Loading, ErrorBox } from "../useApi";
import { MODALITY_COLORS, fmt } from "../constants";

const loadAll = () =>
  Promise.all([
    getConfusion("hybrid"),
    getConfusion("eeg"),
    getConfusion("fnirs"),
    getShap(),
    getAttention(),
    getMetrics(),
    getDiagnostics().catch(() => null),
  ]).then(([hybrid, eeg, fnirs, shap, attention, metrics, diagnostics]) => ({
    hybrid,
    eeg,
    fnirs,
    shap,
    attention,
    metrics,
    diagnostics,
  }));

const LOSO_ROWS = [
  { key: "fnirs_only", label: "fNIRS-only" },
  { key: "hybrid", label: "Hybrid (EEG + fNIRS)" },
  { key: "hybrid_non_normalized", label: "Hybrid, no per-subject norm" },
  { key: "ensemble_hybrid_rf", label: "Hybrid + RF ensemble - served live" },
];

export default function ModelInsights() {
  const { data, loading, error } = useApi(loadAll);
  if (loading) return <div className="container"><Loading /></div>;
  if (error) return <div className="container"><ErrorBox error={error} /></div>;

  const { hybrid, eeg, fnirs, shap, attention, metrics, diagnostics } = data;
  const loso = metrics?.loso || {};
  const bestF1 = Math.max(...LOSO_ROWS.map((r) => loso[r.key]?.f1_mean ?? 0));
  const chartRows = LOSO_ROWS.map((r) => ({
    ...r,
    served: r.key === "ensemble_hybrid_rf",
  }));

  return (
    <div className="container">
      <div className="page-header">
        <div className="eyebrow eyebrow--accent" style={{ marginBottom: 8 }}>Quantitative</div>
        <h1>Model Insights</h1>
        <p>
          Where the model gets things right and wrong, which feature families
          drive its decisions, and how it weighs each modality per class.
        </p>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <h2>LOSO configuration comparison (26 folds)</h2>
        <p className="card-sub">
          Cross-subject (unseen-person) means. The Hybrid + RF ensemble is what
          the live demo serves - the strongest reported configuration. The
          Hybrid network alone and fNIRS-only are kept as comparison points.
        </p>
        <div className="grid grid-2" style={{ marginBottom: 18, alignItems: "start" }}>
          <Figure num="1" title="Macro F1 by configuration, with ±1 fold std." source="metrics.json">
            <LosoConfigChart loso={loso} rows={chartRows} />
          </Figure>
          <Figure num="2" title="Per-class recall - High load is the hardest across every config." source="metrics.json">
            <PerClassRecall loso={loso} rows={LOSO_ROWS} />
          </Figure>
        </div>
        <div className="chart-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>Configuration</th>
                <th>Macro F1</th>
                <th>Accuracy</th>
                <th>AUC</th>
                <th>Recall [L, M, H]</th>
              </tr>
            </thead>
            <tbody>
              {LOSO_ROWS.map((r) => {
                const m = loso[r.key];
                if (!m) return null;
                const best = m.f1_mean === bestF1;
                return (
                  <tr key={r.key} className={best ? "is-best" : undefined}>
                    <td>
                      {r.label} {best && <span className="badge" style={{ background: "#e0e7ff", color: "var(--accent-ink)" }}>best F1</span>}
                    </td>
                    <td>{fmt(m.f1_mean, 4)}</td>
                    <td>{fmt(m.acc_mean, 3)}</td>
                    <td>{fmt(m.auc_mean, 3)}</td>
                    <td>[{m.recall_mean.map((x) => x.toFixed(2)).join(", ")}]</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="note info" style={{ marginTop: 14 }}>
          Against the original "multimodality wins" hypothesis, the Hybrid
          network alone does <strong>not</strong> beat fNIRS-only on unseen
          subjects. Only the Hybrid + Random Forest ensemble pulls clearly ahead.
        </div>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <h2>Confusion matrices (fixed-split test set)</h2>
        <p className="card-sub">
          Hybrid vs. each single-modality baseline. The Hybrid keeps a usable
          diagonal across all three classes; EEG-only collapses toward the
          extremes.
        </p>
        <div className="grid grid-3">
          <ConfusionMatrixChart matrix={hybrid.matrix} title="Hybrid (EEG + fNIRS)" />
          <ConfusionMatrixChart matrix={eeg.matrix} title="EEG only" />
          <ConfusionMatrixChart matrix={fnirs.matrix} title="fNIRS only" />
        </div>
      </div>

      <div className="grid grid-2" style={{ marginBottom: 20 }}>
        <div className="card">
          <h2>SHAP feature-category importance</h2>
          <p className="card-sub">
            Mean absolute SHAP value aggregated by feature family - larger =
            more influence on the model's output.
          </p>
          <ShapChart categories={shap.categories} />
          {shap.per_class && (
            <div style={{ marginTop: 18 }}>
              <div className="stat-sub" style={{ fontWeight: 600, color: "var(--ink-2)", marginBottom: 8 }}>
                Top drivers per class
              </div>
              <ShapPerClass perClass={shap.per_class} topN={5} />
            </div>
          )}
        </div>

        <div className="card">
          <h2>Modality trust by class</h2>
          <p className="card-sub">
            Attention-gate weighting of each stream. This reflects the model's
            learned reliance, not each sensor's standalone accuracy.
          </p>
          <div style={{ marginBottom: 18 }}>
            <div style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--ink-2)", marginBottom: 6 }}>
              Overall
            </div>
            <ModalityContribution
              eegShare={attention.overall?.eeg_share}
              fnirsShare={attention.overall?.fnirs_share}
              compact
            />
          </div>
          {(attention.per_class || []).map((pc) => (
            <div key={pc.class} style={{ marginBottom: 14 }}>
              <div style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--ink-2)", marginBottom: 6 }}>
                {pc.class} load
              </div>
              <ModalityContribution eegShare={pc.eeg_share} fnirsShare={pc.fnirs_share} compact />
            </div>
          ))}
          <div style={{ display: "flex", gap: 16, marginTop: 12 }}>
            <span style={{ fontSize: 12, color: "var(--ink-2)", display: "inline-flex", gap: 6, alignItems: "center" }}>
              <span className="swatch" style={{ background: MODALITY_COLORS.eeg }} /> EEG
            </span>
            <span style={{ fontSize: 12, color: "var(--ink-2)", display: "inline-flex", gap: 6, alignItems: "center" }}>
              <span className="swatch" style={{ background: MODALITY_COLORS.fnirs }} /> fNIRS
            </span>
          </div>
        </div>
      </div>

      {diagnostics && (
        <div className="card" style={{ marginBottom: 20 }}>
          <h2>On the held-out demo windows</h2>
          <p className="card-sub">
            The frozen served ensemble run over the {diagnostics.n_windows} demo
            windows ({diagnostics.per_subject?.length} subjects). Read-only
            diagnostics - the same numbers behind the Live demo, aggregated.
          </p>
          <div className="grid grid-2" style={{ alignItems: "start" }}>
            <Figure num="3" title="Accuracy varies widely by subject - the variability ceiling, made concrete." source="diagnostics.json">
              <PerSubjectBars perSubject={diagnostics.per_subject} />
            </Figure>
            <div>
              <div className="stat-sub" style={{ fontWeight: 600, color: "var(--ink-2)", marginBottom: 10 }}>
                Aggregate confusion · demo windows
              </div>
              <ConfusionMatrixChart matrix={diagnostics.demo_confusion?.matrix} />
              <div className="note info" style={{ marginTop: 14, fontSize: "0.82rem" }}>
                Overall demo accuracy {(diagnostics.overall?.accuracy * 100).toFixed(1)}% ·
                macro F1 {fmt(diagnostics.overall?.macro_f1, 3)}. This is the
                fixed-split demo subset, not a LOSO estimate.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
