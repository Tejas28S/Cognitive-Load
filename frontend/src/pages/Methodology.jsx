import { Link } from "react-router-dom";
import { getMetrics, getDiagnostics } from "../api/client";
import { useApi, Loading, ErrorBox } from "../useApi";
import { useModelVersion } from "../lib/modelVersion";
import { fmt } from "../constants";
import Reveal from "../components/research/Reveal";
import Figure from "../components/research/Figure";
import VerdictCallout from "../components/research/VerdictCallout";
import MetricStat from "../components/research/MetricStat";
import ForestPlot from "../components/research/ForestPlot";
import ReliabilityDiagram from "../components/research/ReliabilityDiagram";
import CoverageCurve from "../components/research/CoverageCurve";
import LeakageSchematic from "../components/research/LeakageSchematic";

const SIG_ORDER = [
  "hybrid_vs_fnirs_f1",
  "hybrid_vs_fnirs_auc",
  "normalized_vs_nonnormalized_f1",
  "normalized_vs_nonnormalized_auc",
];

const pct = (x) => (x == null ? "-" : `${(x * 100).toFixed(1)}%`);

// Load metrics (required) + diagnostics (optional - may 404 if the artifact
// hasn't been generated; the calibration section then hides gracefully).
const loadAll = (version) =>
  Promise.all([
    getMetrics(version),
    getDiagnostics(version).catch(() => null),
  ]).then(([metrics, diagnostics]) => ({ metrics, diagnostics }));

export default function Methodology() {
  const { version, classNames } = useModelVersion();
  const { data, loading, error } = useApi(() => loadAll(version), [version]);
  if (loading) return <div className="container"><Loading /></div>;
  if (error) return <div className="container"><ErrorBox error={error} /></div>;

  const { metrics, diagnostics } = data;
  const sig = metrics?.significance || {};
  const accepted = metrics?.ensemble_accepted;
  const reason = metrics?.acceptance_reason;
  const classes = classNames?.length ? classNames : version === "3class" ? ["Low", "Medium", "High"] : ["Low", "High"];
  const classPhrase = classes.join(", ").replace(/, ([^,]*)$/, " or $1");

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

      <Reveal className="card" style={{ marginBottom: 20 }}>
        <h2>The problem</h2>
        <p>
          Given short windows of simultaneously-recorded EEG (electrical) and
          fNIRS (hemodynamic) brain activity from the Tu Berlin open dataset (26
          subjects), classify the person's mental workload into {classPhrase}.
          Multimodality is the bet: EEG is fast but noisy, fNIRS is slow but
          stable, and a fusion model should lean on whichever is more informative.
        </p>
      </Reveal>

      <Reveal className="card" style={{ marginBottom: 20 }}>
        <h2>Fixed split vs. LOSO - the subject-leakage finding</h2>
        <p>
          A <strong>fixed split</strong> can place different trials from the same
          person in both training and test. The model can then latch onto
          person-specific signal quirks rather than workload, and the reported
          score is optimistic. <strong>Leave-one-subject-out (LOSO)</strong>{" "}
          holds out every subject in turn, so the test person is always unseen -
          the honest measure of generalization.
        </p>
        <Figure num="M1" title="Why a fixed split leaks subject identity, and how LOSO closes it." source="protocol_benchmark.py">
          <LeakageSchematic />
        </Figure>
        <p style={{ marginTop: 16 }}>
          The two numbers diverge substantially (see the Overview). That gap is
          the finding: cross-subject cognitive-load decoding from these features
          is hard, and any single fixed-split number overstates it. We always
          report both, and treat the LOSO figure as the real capability estimate.
        </p>
      </Reveal>

      {Object.keys(sig).length > 0 && (
      <Reveal className="card" style={{ marginBottom: 20 }}>
        <h2>Statistical significance (paired across 26 LOSO folds)</h2>
        <p className="card-sub">
          Paired t-test and Wilcoxon signed-rank over per-fold scores - not
          eyeballed from mean ± std. A comparison counts as significant only when
          BOTH tests clear p &lt; 0.05.
        </p>
        <Figure num="M2" title="Effect size of each paired comparison, with significance." source="significance_tests.json">
          <ForestPlot tests={sig} order={SIG_ORDER} />
        </Figure>
        <p style={{ marginTop: 16 }}>
          Two honest takeaways. First, <strong>per-subject normalization is the
          clear, significant win</strong> (p &lt; 0.005 both tests). Second - and
          against the original hypothesis - the <strong>Hybrid network does not
          beat fNIRS-only on unseen subjects</strong>; fNIRS-only is actually a
          touch higher on macro F1 (the difference is significant by Wilcoxon).
          Multimodality only pulls ahead once the Hybrid is combined with a
          Random Forest in a probability-averaging ensemble (see the LOSO
          comparison on Model Insights). We report this rather than oversell the
          fusion story.
        </p>
      </Reveal>
      )}
      {Object.keys(sig).length === 0 && (
      <Reveal className="card" style={{ marginBottom: 20 }}>
        <h2>Statistical significance</h2>
        <p>
          The paired-significance battery above belongs to the original 3-class
          study. The binary campaign's decision-gate results - 42 experiments
          selected by a one-standard-error rule against an 18-subject
          development pool, verified on an 8-subject lockbox - are on the{" "}
          <Link to="/evaluation">Evaluation</Link> and <Link to="/runs">Runs</Link>{" "}
          pages, sourced directly from the run archive.
        </p>
      </Reveal>
      )}

      {diagnostics && (
        <Reveal className="card" style={{ marginBottom: 20 }}>
          <h2>Is the model's confidence trustworthy? (calibration)</h2>
          <p className="card-sub">
            Read-only analysis of the frozen served ensemble over the{" "}
            {diagnostics.n_windows} held-out demo windows - the served model is
            unchanged. A well-calibrated model's confidence matches its accuracy.
          </p>
          <div className="grid grid-3" style={{ marginBottom: 18 }}>
            <div className="card card-flat">
              <MetricStat value={diagnostics.overall?.ece} format={(v) => v.toFixed(3)}
                label="Expected calibration error" sub="0 = perfectly calibrated" accent="var(--accent)" />
            </div>
            <div className="card card-flat">
              <MetricStat value={diagnostics.overall?.mean_confidence} format={pct}
                label="Mean confidence" sub={`vs accuracy ${pct(diagnostics.overall?.accuracy)}`} />
            </div>
            <div className="card card-flat">
              <MetricStat value={diagnostics.latency?.p95_ms} format={(v) => `${v.toFixed(0)} ms`}
                label="Latency · p95" sub={`mean ${diagnostics.latency?.mean_ms?.toFixed(0)} ms`} />
            </div>
          </div>
          <div className="grid grid-2">
            <Figure num="M3" title="Reliability diagram - accuracy vs confidence per bin." source="diagnostics.json">
              <ReliabilityDiagram reliability={diagnostics.reliability} />
            </Figure>
            <Figure num="M4" title="Confidence–coverage - accuracy rises as low-confidence calls are abstained." source="diagnostics.json">
              <CoverageCurve curve={diagnostics.coverage_curve} fullAccuracy={diagnostics.overall?.accuracy} />
            </Figure>
          </div>
          <div className="note info" style={{ marginTop: 16, fontSize: "0.85rem" }}>
            The coverage curve is the evidence behind the demo's <strong>abstain</strong>{" "}
            behaviour: because accuracy climbs as the least-confident windows are
            dropped, refusing low-confidence calls is a principled trade, not a
            cosmetic one. The model is mildly over-confident (ECE ≈{" "}
            {fmt(diagnostics.overall?.ece, 3)}) - reported, not hidden.
          </div>
        </Reveal>
      )}

      <Reveal className="card" style={{ marginBottom: 20 }}>
        <h2>What we tried that didn't win</h2>
        <div className="grid grid-2">
          <VerdictCallout tone="warn" stamp="Reverted" title="Class-weight tuning">
            Tuned class weights to rescue High-load recall - improved recall but
            dropped macro F1 and AUC; reverted after one targeted attempt.
          </VerdictCallout>
          <VerdictCallout tone="warn" stamp="Reverted" title="Normalize after log">
            Normalization applied after log-compression gave slightly more balanced
            recall but lower LOSO macro F1 than normalizing before; kept the
            before-log stage.
          </VerdictCallout>
          <VerdictCallout tone="stop" stamp="Blocked" title="ICA-based EEG rebuild">
            Gated on a classical-ML AUC ≥ 0.55 sanity check it never cleared, so
            deep retraining on the reworked EEG features stayed blocked - documented
            as a finding, not adopted.
          </VerdictCallout>
          <VerdictCallout tone={accepted ? "ok" : "warn"} stamp={accepted ? "Adopted" : "Evaluated"} title="Hybrid + Random Forest ensemble">
            <span>{reason}</span>
          </VerdictCallout>
        </div>
      </Reveal>

      <Reveal className="card">
        <h2>Limitations to keep in mind</h2>
        <div className="note" style={{ marginBottom: 12 }}>
          <strong>Per-subject normalization is a demo-mode simplification.</strong>{" "}
          The model normalizes each subject against that subject's own
          statistics. For a genuinely new person those stats don't exist yet, so
          the demo reuses each known subject's precomputed stats. This is honest
          demo scaffolding, not production behaviour.
        </div>
        <ul style={{ lineHeight: 1.7, color: "var(--ink-2)" }}>
          <li>Only 26 subjects - small for cross-subject generalization claims.</li>
          <li>
            The shaded "sustained-engagement" band is literature-informed
            context (Yerkes–Dodson), not a clinical or diagnostic threshold.
          </li>
          <li>
            All demo predictions come from real held-out test windows, but they
            are a curated subset chosen for coherent playback, not a fresh
            recording.
          </li>
          {diagnostics && (
            <li>
              The calibration figures above are on the fixed-split demo windows,
              not a fresh LOSO run - a diagnostic of the served bundle, read
              honestly as such.
            </li>
          )}
        </ul>
      </Reveal>
    </div>
  );
}
