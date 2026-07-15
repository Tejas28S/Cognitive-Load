import { getMetrics } from "../api/client";
import { useApi, Loading, ErrorBox } from "../useApi";
import { fmt } from "../constants";

const SIG_ORDER = [
  "hybrid_vs_fnirs_f1",
  "hybrid_vs_fnirs_auc",
  "normalized_vs_nonnormalized_f1",
  "normalized_vs_nonnormalized_auc",
];

export default function Methodology() {
  const { data, loading, error } = useApi(getMetrics);
  if (loading) return <div className="container"><Loading /></div>;
  if (error) return <div className="container"><ErrorBox error={error} /></div>;

  const sig = data?.significance || {};
  const accepted = data?.ensemble_accepted;
  const reason = data?.acceptance_reason;

  return (
    <div className="container">
      <div className="page-header">
        <div className="eyebrow eyebrow--accent" style={{ marginBottom: 8 }}>Reproducibility</div>
        <h1>Methodology &amp; Honest Limitations</h1>
        <p>
          This page is deliberately front-and-centre. The most important result
          in this project is a methodological one, and burying it would misrepresent
          what the model can actually do.
        </p>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <h2>The problem</h2>
        <p>
          Given short windows of simultaneously-recorded EEG (electrical) and
          fNIRS (hemodynamic) brain activity from the Tu Berlin open dataset (26
          subjects), classify the person's mental workload into Low, Medium, or
          High. Multimodality is the bet: EEG is fast but noisy, fNIRS is slow but
          stable, and a fusion model should lean on whichever is more informative.
        </p>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <h2>Fixed split vs. LOSO — the subject-leakage finding</h2>
        <p>
          A <strong>fixed split</strong> can place different trials from the same
          person in both training and test. The model can then latch onto
          person-specific signal quirks rather than workload, and the reported
          score is optimistic. <strong>Leave-one-subject-out (LOSO)</strong>{" "}
          holds out every subject in turn, so the test person is always unseen —
          the honest measure of generalization.
        </p>
        <p>
          The two numbers diverge substantially (see the Overview). That gap is
          the finding: cross-subject cognitive-load decoding from these features
          is hard, and any single fixed-split number overstates it. We always
          report both, and treat the LOSO figure as the real capability estimate.
        </p>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <h2>Statistical significance (paired across 26 LOSO folds)</h2>
        <p className="card-sub">
          Paired t-test and Wilcoxon signed-rank over per-fold scores — not
          eyeballed from mean ± std.
        </p>
        <div className="chart-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>Comparison</th>
                <th>Mean diff</th>
                <th>Paired t p</th>
                <th>Wilcoxon p</th>
              </tr>
            </thead>
            <tbody>
              {SIG_ORDER.filter((k) => sig[k]).map((k) => (
                <tr key={k}>
                  <td>{sig[k].comparison}</td>
                  <td>{sig[k].mean_diff >= 0 ? "+" : ""}{fmt(sig[k].mean_diff, 4)}</td>
                  <td>{fmt(sig[k].t_p, 4)}</td>
                  <td>{fmt(sig[k].wilcoxon_p, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p style={{ marginTop: 12 }}>
          Two honest takeaways. First, <strong>per-subject normalization is the
          clear, significant win</strong> (p &lt; 0.005 both tests). Second — and
          against the original hypothesis — the <strong>Hybrid network does not
          beat fNIRS-only on unseen subjects</strong>; fNIRS-only is actually a
          touch higher on macro F1 (the difference is significant by Wilcoxon).
          Multimodality only pulls ahead once the Hybrid is combined with a
          Random Forest in a probability-averaging ensemble (see the LOSO
          comparison on Model Insights). We report this rather than oversell the
          fusion story.
        </p>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <h2>What we tried that didn't win</h2>
        <ul style={{ lineHeight: 1.7, color: "var(--ink-2)" }}>
          <li>
            <strong>Class-weight tuning</strong> to rescue High-load recall —
            improved recall but dropped macro F1 and AUC; reverted after one
            targeted attempt.
          </li>
          <li>
            <strong>Normalization applied after log-compression</strong> —
            slightly more balanced recall but lower LOSO macro F1 than
            normalizing before; kept the before-log stage.
          </li>
          <li>
            <strong>ICA-based EEG rebuild</strong> — gated on a classical-ML AUC
            ≥ 0.55 sanity check that it never cleared, so deep retraining on the
            reworked EEG features stayed blocked and is documented as a finding,
            not adopted.
          </li>
          <li>
            <strong>Hybrid + Random Forest ensemble</strong> —{" "}
            {accepted ? "adopted as an auxiliary result" : "evaluated"}:{" "}
            <span className="mono">{reason}</span>.
          </li>
        </ul>
      </div>

      <div className="card">
        <h2>Limitations to keep in mind</h2>
        <div className="note" style={{ marginBottom: 12 }}>
          <strong>Per-subject normalization is a demo-mode simplification.</strong>{" "}
          The model normalizes each subject against that subject's own
          statistics. For a genuinely new person those stats don't exist yet, so
          the demo reuses each known subject's precomputed stats. This is honest
          demo scaffolding, not production behaviour.
        </div>
        <ul style={{ lineHeight: 1.7, color: "var(--ink-2)" }}>
          <li>Only 26 subjects — small for cross-subject generalization claims.</li>
          <li>
            The shaded "sustained-engagement" band is literature-informed
            context (Yerkes–Dodson), not a clinical or diagnostic threshold.
          </li>
          <li>
            All demo predictions come from real held-out test windows, but they
            are a curated subset chosen for coherent playback, not a fresh
            recording.
          </li>
        </ul>
      </div>
    </div>
  );
}
