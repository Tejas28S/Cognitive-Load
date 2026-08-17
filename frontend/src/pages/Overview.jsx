import { Link } from "react-router-dom";
import MetricsSummaryCard from "../components/MetricsSummaryCard";
import { getMetrics } from "../api/client";
import { useApi, Loading, ErrorBox } from "../useApi";
import { fmt, CLASS_COLORS, CLASS_NAMES } from "../constants";
import SignalTrace from "../components/SignalTrace";
import MetricStat from "../components/research/MetricStat";
import GapDumbbell from "../components/GapDumbbell";
import ArchitectureDiagram from "../components/ArchitectureDiagram";
import Figure from "../components/research/Figure";
import Reveal from "../components/research/Reveal";
import AnimatedBand from "../components/AnimatedBand";

const pct = (x) => (x == null ? "-" : `${(x * 100).toFixed(1)}%`);

const TEASERS = [
  {
    stamp: "Scholarship",
    title: "The honest protocol",
    body: "Literature's 60–90% is usually a binary or within-subject task. This reports 3-class, leave-one-subject-out - and shows exactly why the gap exists.",
    to: "/research#protocol",
    go: "Protocol taxonomy →",
  },
  {
    stamp: "Novel · declined",
    title: "Teaching the model to forget who",
    body: "A domain-adversarial network attacks subject variability directly. Applied with a paired-significance gate, it is correctly not adopted - a mechanistically-explained negative result.",
    to: "/research#dann",
    go: "The DANN contribution →",
  },
  {
    stamp: "Discipline",
    title: "One decision gate, every candidate",
    body: "HbR features, XGBoost, SVM, DANN - each measured against the served ensemble and adopted only on a clear, significant win. Most were declined, on the record.",
    to: "/research#gates",
    go: "Decision gates →",
  },
  {
    stamp: "Deployable",
    title: "Explainable and served live",
    body: "SHAP, attention, and RF importance all point to fNIRS. The frozen model runs behind a real API - every number traces to a committed artifact.",
    to: "/insights",
    go: "Model insights →",
  },
];

export default function Overview() {
  const { data: dataBin, loading: lBin, error: errBin } = useApi(() => getMetrics("binary"), []);
  const { data: data3c, loading: l3c, error: err3c } = useApi(() => getMetrics("3class"), []);

  const loading = lBin || l3c;
  const error = errBin || err3c;

  const fx = data3c?.fixed_split?.hybrid || {};
  const loso = data3c?.loso?.hybrid || {};
  const ens = data3c?.loso?.ensemble_hybrid_rf || {};
  
  const losoBin = dataBin?.loso?.ensemble_hybrid_rf || dataBin?.loso?.hybrid || {};

  return (
    <>
      <header className="band-dark page-hero">
        <AnimatedBand />
        <div className="hero-inner">
          <div className="eyebrow">EEG + fNIRS · mental workload · capstone</div>
          <h1 className="hero-title" style={{ marginTop: 16 }}>
            Reading mental workload from the brain - and reporting it honestly
          </h1>
          <p className="hero-lede" style={{ marginTop: 14 }}>
            A hybrid model fuses electrical (EEG) and hemodynamic (fNIRS) brain
            signals to estimate cognitive load as Low, Medium, or High. The result
            that matters most is methodological: how the model performs on people
            it has never seen.
          </p>

          <div className="hero-actions">
            <Link to="/research" className="btn-lg btn-solid">Read the research</Link>
            <Link to="/live" className="btn-lg btn-on-dark">Try the live demo</Link>
          </div>

          {!loading && !error && (
            <div className="hero-metrics">
              <MetricStat onDark value={losoBin.f1_mean} format={pct}
                label="Current binary macro F1 · LOSO" sub="Hybrid, 26 subjects held out" />
              <MetricStat onDark value={losoBin.acc_mean} format={pct}
                label="Accuracy · LOSO" sub="Binary · chance ≈ 50%" />
              <MetricStat onDark value={26} format={(n) => Math.round(n).toString()}
                label="Subjects" sub="TU Berlin / Shin n-back" />
            </div>
          )}
          <SignalTrace className="hero-signal" height={48} strokeWidth={2.2} animate glow />
        </div>
      </header>

      <div className="container">
        {loading && <Loading />}
        {error && <ErrorBox error={error} />}
        {!loading && !error && (
          <div className="stack-lg">
            {/* Two numbers, read together */}
            <Reveal as="section">
              <div className="eyebrow eyebrow--accent" style={{ marginBottom: 8 }}>
                Performance Comparison
              </div>
              <h2>Binary vs 3-Class Models (LOSO Evaluation)</h2>
              <div className="grid grid-2" style={{ marginTop: 16 }}>
                <div className="card">
                  <h3>Legacy 3-Class (Baseline)</h3>
                  <p className="card-sub">
                    Predicting Low, Medium, or High load. Evaluated on completely unseen subjects (Leave-One-Subject-Out).
                  </p>
                  <div className="grid grid-3">
                    <MetricsSummaryCard label="Accuracy" value={fmt(loso.acc_mean, 3)} sub="26-fold mean" />
                    <MetricsSummaryCard label="Macro F1" value={fmt(loso.f1_mean, 4)} sub="26-fold mean" />
                    <MetricsSummaryCard label="AUC" value={fmt(loso.auc_mean, 3)} sub="26-fold mean" />
                  </div>
                </div>
                <div className="card" style={{ borderColor: "var(--accent)" }}>
                  <h3 style={{ color: "var(--accent)" }}>Binary Classification (Improved)</h3>
                  <p className="card-sub">
                    Predicting High vs Low load. Evaluated on completely unseen subjects (Leave-One-Subject-Out).
                  </p>
                  <div className="grid grid-3">
                    <MetricsSummaryCard label="Accuracy" value={fmt(losoBin.acc_mean, 3)} sub="26-fold mean" accent="var(--accent)" />
                    <MetricsSummaryCard label="Macro F1" value={fmt(losoBin.f1_mean, 4)}
                      sub={`± ${fmt(losoBin.f1_std, 3)} (26 folds)`} accent="var(--accent)" />
                    <MetricsSummaryCard label="AUC" value={fmt(losoBin.auc_mean, 3)} sub="26-fold mean" accent="var(--accent)" />
                  </div>
                </div>
              </div>
              <div className="card" style={{ marginTop: 18 }}>
                <h3 style={{ marginTop: 0 }}>Accuracy Jump: 3-Class vs Binary</h3>
                <p className="card-sub">
                  The original 3-class baseline failed to generalize. By moving to a <strong>binary</strong> classification (Low vs High) and introducing a 1-second sliding window augmentation, performance surged dramatically.
                </p>
                <GapDumbbell
                  domain={[0.3, 0.7]}
                  label1="Legacy 3-Class (Baseline)"
                  label2="Binary Classification (Improved)"
                  gapLabel="performance gain"
                  gapDirection="loso-fixed"
                  rows={[
                    { metric: "Macro F1", fixed: ens.f1_mean, loso: losoBin.f1_mean },
                    { metric: "Accuracy", fixed: ens.acc_mean, loso: losoBin.acc_mean },
                    { metric: "AUC", fixed: ens.auc_mean, loso: losoBin.auc_mean },
                  ]}
                />
              </div>
              <div className="note info" style={{ marginTop: 16 }}>
                The gap between the panels shows the dramatic improvement. The grey dot represents the failing <strong>legacy 3-class model</strong> (~40-44%), while the <span style={{ color: "var(--accent)" }}>blue dot</span> represents the new <strong>binary classification model</strong> (~65%) operating on the augmented dataset.
              </div>
              <div style={{ display: "flex", gap: 12, marginTop: 14, flexWrap: "wrap" }}>
                {CLASS_NAMES.map((c) => (
                  <span key={c} className="pill">
                    <span className="swatch" style={{ background: CLASS_COLORS[c] }} />
                    {c} load
                  </span>
                ))}
              </div>
            </Reveal>

            {/* What's different */}
            <Reveal as="section">
              <div className="eyebrow eyebrow--accent" style={{ marginBottom: 8 }}>
                What makes this different
              </div>
              <h2>Not a leaderboard entry - a decisively-answered study</h2>
              <div className="teaser-grid" style={{ marginTop: 16 }}>
                {TEASERS.map((t) => (
                  <Link key={t.title} to={t.to} className="teaser">
                    <div className="teaser-stamp">{t.stamp}</div>
                    <h3>{t.title}</h3>
                    <p>{t.body}</p>
                    <span className="teaser-go">{t.go}</span>
                  </Link>
                ))}
              </div>
            </Reveal>

            {/* What the model does */}
            <Reveal as="section" className="card">
              <h2>What the model does</h2>
              <p>
                Each input is a per-trial feature window: 180 EEG band-power
                features and 468 fNIRS (HbO) response-shape descriptors. Two
                parallel encoders compress each modality, an attention-gated fusion
                layer weighs them, and a small classifier outputs a Low / Medium /
                High probability.
              </p>
              <Figure num="0" title="The served Hybrid + Random Forest ensemble." source="model_config.json">
                <ArchitectureDiagram />
              </Figure>
              <p style={{ marginTop: 16 }}>
                Explore a <Link to="/live">single prediction</Link>, watch a{" "}
                <Link to="/session">session over time</Link>, or read the full{" "}
                <Link to="/research">research dossier</Link>.
              </p>
            </Reveal>
          </div>
        )}
      </div>
    </>
  );
}
