import { Link } from "react-router-dom";
import MetricsSummaryCard from "../components/MetricsSummaryCard";
import { getMetrics } from "../api/client";
import { useApi, Loading, ErrorBox } from "../useApi";
import { fmt, CLASS_COLORS, CLASS_NAMES } from "../constants";
import SignalTrace from "../components/SignalTrace";
import MetricStat from "../components/research/MetricStat";

const pct = (x) => (x == null ? "—" : `${(x * 100).toFixed(1)}%`);

const TEASERS = [
  {
    stamp: "Scholarship",
    title: "The honest protocol",
    body: "Literature's 60–90% is usually a binary or within-subject task. This reports 3-class, leave-one-subject-out — and shows exactly why the gap exists.",
    to: "/research#protocol",
    go: "Protocol taxonomy →",
  },
  {
    stamp: "Novel · declined",
    title: "Teaching the model to forget who",
    body: "A domain-adversarial network attacks subject variability directly. Applied with a paired-significance gate, it is correctly not adopted — a mechanistically-explained negative result.",
    to: "/research#dann",
    go: "The DANN contribution →",
  },
  {
    stamp: "Discipline",
    title: "One decision gate, every candidate",
    body: "HbR features, XGBoost, SVM, DANN — each measured against the served ensemble and adopted only on a clear, significant win. Most were declined, on the record.",
    to: "/research#gates",
    go: "Decision gates →",
  },
  {
    stamp: "Deployable",
    title: "Explainable and served live",
    body: "SHAP, attention, and RF importance all point to fNIRS. The frozen model runs behind a real API — every number traces to a committed artifact.",
    to: "/insights",
    go: "Model insights →",
  },
];

export default function Overview() {
  const { data, loading, error } = useApi(getMetrics);

  const fx = data?.fixed_split?.hybrid || {};
  const loso = data?.loso?.hybrid || {};
  const ens = data?.loso?.ensemble_hybrid_rf || {};

  return (
    <>
      <header className="band-dark page-hero">
        <div className="hero-inner">
          <div className="eyebrow">EEG + fNIRS · mental workload · capstone</div>
          <h1 className="hero-title" style={{ marginTop: 16 }}>
            Reading mental workload from the brain — and reporting it honestly
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
              <MetricStat onDark value={ens.f1_mean} format={pct}
                label="Served macro F1 · LOSO" sub="Hybrid + RF, 26 subjects held out" />
              <MetricStat onDark value={ens.acc_mean} format={pct}
                label="Accuracy · LOSO" sub="3-class · chance ≈ 33%" />
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
            <section>
              <div className="eyebrow eyebrow--accent" style={{ marginBottom: 8 }}>
                Read these together
              </div>
              <h2>Two evaluations, two different questions</h2>
              <div className="grid grid-2" style={{ marginTop: 16 }}>
                <div className="card">
                  <h3>Fixed split</h3>
                  <p className="card-sub">
                    Subjects may appear in both train and test trials — a
                    "known people" setting.
                  </p>
                  <div className="grid grid-3">
                    <MetricsSummaryCard label="Accuracy" value={fmt(fx.acc, 3)} sub="fixed-split" />
                    <MetricsSummaryCard label="Macro F1" value={fmt(fx.f1_macro, 4)} sub="fixed-split" />
                    <MetricsSummaryCard label="AUC" value={fmt(fx.auc, 3)} sub="fixed-split" />
                  </div>
                </div>
                <div className="card">
                  <h3>LOSO · leave-one-subject-out</h3>
                  <p className="card-sub">
                    Every test subject is entirely unseen — the honest estimate
                    for a new person.
                  </p>
                  <div className="grid grid-3">
                    <MetricsSummaryCard label="Accuracy" value={fmt(loso.acc_mean, 3)} sub="26-fold mean" />
                    <MetricsSummaryCard label="Macro F1" value={fmt(loso.f1_mean, 4)}
                      sub={`± ${fmt(loso.f1_std, 3)} (26 folds)`} accent="var(--accent)" />
                    <MetricsSummaryCard label="AUC" value={fmt(loso.auc_mean, 3)} sub="26-fold mean" />
                  </div>
                </div>
              </div>
              <div className="note info" style={{ marginTop: 16 }}>
                The gap between the panels is the headline finding, not a bug: when
                subjects leak across the split, the model looks stronger than it is
                on genuinely new people. These LOSO figures are the Hybrid net
                alone; the live demo serves the stronger{" "}
                <strong>Hybrid + Random Forest ensemble</strong>. See{" "}
                <Link to="/research#protocol">why the gap exists</Link>.
              </div>
              <div style={{ display: "flex", gap: 12, marginTop: 14, flexWrap: "wrap" }}>
                {CLASS_NAMES.map((c) => (
                  <span key={c} className="pill">
                    <span className="swatch" style={{ background: CLASS_COLORS[c] }} />
                    {c} load
                  </span>
                ))}
              </div>
            </section>

            {/* What's different */}
            <section>
              <div className="eyebrow eyebrow--accent" style={{ marginBottom: 8 }}>
                What makes this different
              </div>
              <h2>Not a leaderboard entry — a decisively-answered study</h2>
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
            </section>

            {/* What the model does */}
            <section className="card">
              <h2>What the model does</h2>
              <p>
                Each input is a per-trial feature window: 180 EEG band-power
                features and 468 fNIRS (HbO) response-shape descriptors. Two
                parallel encoders compress each modality, an attention-gated fusion
                layer weighs them, and a small classifier outputs a Low / Medium /
                High probability — averaged with a Random Forest for the served
                prediction. Explore a{" "}
                <Link to="/live">single prediction</Link>, watch a{" "}
                <Link to="/session">session over time</Link>, or read the full{" "}
                <Link to="/research">research dossier</Link>.
              </p>
            </section>
          </div>
        )}
      </div>
    </>
  );
}
