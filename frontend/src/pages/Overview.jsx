import { Link } from "react-router-dom";
import MetricsSummaryCard from "../components/MetricsSummaryCard";
import { getAttention, getMetrics, getShap } from "../api/client";
import { useModelVersion } from "../lib/modelVersion";
import ModelSwitch from "../components/ModelSwitch";
import ModelProvenanceNote from "../components/ModelProvenanceNote";
import { useApi, Loading, ErrorBox } from "../useApi";
import { fmt, CLASS_COLORS } from "../constants";
import SignalTrace from "../components/SignalTrace";
import MetricStat from "../components/research/MetricStat";
import ShapChart from "../components/ShapChart";
import GapDumbbell from "../components/GapDumbbell";
import ArchitectureDiagram from "../components/ArchitectureDiagram";
import Figure from "../components/research/Figure";
import Reveal from "../components/research/Reveal";
import AnimatedBand from "../components/AnimatedBand";

const pct = (x) => (x == null ? "-" : `${(x * 100).toFixed(1)}%`);

const TEASERS = [
  {
    stamp: "Current work",
    title: "The leakage-audited campaign behind the binary model",
    body: "42 controlled experiments on an 18-subject dev pool, one 8-subject lockbox touched once, and a leakage audit that aborts a run rather than let it produce a flattered score. Every run is archived and independently re-verifiable.",
    to: "/evaluation",
    go: "Evaluation →",
  },
  {
    stamp: "Scholarship",
    title: "The honest protocol",
    body: "Literature's 60-90% is usually a binary or within-subject task. This reports leave-one-subject-out, where every test person is genuinely unseen - and shows exactly why the gap exists.",
    to: "/methodology",
    go: "Methodology →",
  },
  {
    stamp: "On the record",
    title: "Every run kept, including the failures",
    body: "58 configurations were trained and scored; none beat the baseline by enough to replace it. The losing arms are archived alongside the winner rather than quietly dropped.",
    to: "/runs",
    go: "Run archive →",
  },
  {
    stamp: "Discipline",
    title: "One decision gate, every candidate",
    body: "HbR features, XGBoost, SVM, DANN - each measured against the served model and adopted only on a clear, significant win. Most were declined, on the record.",
    to: "/dashboard#decisions",
    go: "Decision ledger →",
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
  const { version, active } = useModelVersion();

  // Both bundles are fetched so the comparison below can stand on its own; the
  // switch decides which one the headline numbers report.
  const { data: dataBin, loading: lBin, error: errBin } = useApi(() => getMetrics("binary"), []);
  const { data: data3c, loading: l3c, error: err3c } = useApi(() => getMetrics("3class"), []);
  // Attribution follows the switch, like the headline numbers. It is allowed to
  // be missing: the section below hides itself rather than blanking the page.
  const { data: shapData } = useApi(
    () => getShap(version).catch(() => null),
    [version]
  );
  // The independent second reading the attribution section cites. Optional for
  // the same reason: a missing artifact drops the aside, never the section.
  const { data: attentionData } = useApi(
    () => getAttention(version).catch(() => null),
    [version]
  );

  const loading = lBin || l3c;
  // Only the *selected* model failing should blank the page.
  const error = version === "3class" ? err3c : errBin;
  // The legacy 3-class bundle is optional: its weights are not always present.
  // Without it the page drops the comparison rather than failing outright.
  const has3c = Boolean(data3c) && !err3c;

  const loso = data3c?.loso?.hybrid || {};
  const ens = data3c?.loso?.ensemble_hybrid_rf || {};

  const losoBin = dataBin?.loso?.ensemble_hybrid_rf || dataBin?.loso?.hybrid || {};

  // Headline stats follow the switch.
  const heroData = version === "3class" ? data3c : dataBin;
  const heroLoso = heroData?.loso?.ensemble_hybrid_rf || heroData?.loso?.hybrid || {};
  const heroClasses = active?.class_names?.length
    ? active.class_names
    : version === "3class"
      ? ["Low", "Medium", "High"]
      : ["Low", "High"];
  // 32 families would leave the axis too crowded for Recharts to draw every
  // label; the tail is all near-zero anyway and the full list lives on Insights.
  const topFamilies = (shapData?.categories || []).slice(0, 12);
  const fnirsShare = attentionData?.overall?.fnirs_share;
  const attentionAside =
    fnirsShare == null ? "" : ` (${pct(fnirsShare)} of the fusion weight on fNIRS)`;

  const heroLabel = `${heroClasses.length}-class`;
  const heroChance = `${Math.round(100 / heroClasses.length)}%`;

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
            signals to estimate cognitive load as{" "}
            {heroClasses.join(", ").replace(/, ([^,]*)$/, " or $1")}. The result
            that matters most is methodological: how the model performs on people
            it has never seen.
          </p>

          <div style={{ marginTop: 20 }}>
            <ModelSwitch variant="hero" />
          </div>

          <div className="hero-actions">
            <Link to="/dashboard" className="btn-lg btn-solid">Open the dashboard</Link>
            <Link to="/live" className="btn-lg btn-on-dark">Try the live demo</Link>
          </div>

          {!loading && !error && (
            <div className="hero-metrics">
              <MetricStat value={heroLoso.f1_mean} format={pct}
                label={`Macro F1 · LOSO · ${heroLabel}`} sub="Hybrid, 26 subjects held out" />
              <MetricStat value={heroLoso.acc_mean} format={pct}
                label="Accuracy · LOSO" sub={`${heroLabel} · chance ≈ ${heroChance}`} />
              <MetricStat value={26} format={(n) => Math.round(n).toString()}
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
            <ModelProvenanceNote />
            {/* Two numbers, read together */}
            <Reveal as="section">
              <div className="eyebrow eyebrow--accent" style={{ marginBottom: 8 }}>
                Performance Comparison
              </div>
              <h2>{has3c ? "Binary vs 3-Class Models (LOSO Evaluation)" : "Binary Model (LOSO Evaluation)"}</h2>
              <div className={has3c ? "grid grid-2" : "grid"} style={{ marginTop: 16 }}>
                {has3c && (
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
                )}
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
              {has3c && (
              <div className="card" style={{ marginTop: 18 }}>
                <h3 style={{ marginTop: 0 }}>3-class vs binary, same LOSO protocol</h3>
                <p className="card-sub">
                  Each row is one metric measured twice: on the legacy 3-class model,
                  then on the binary (Low vs High) model trained with 1-second sliding
                  window augmentation. The dot moves right as the score improves.
                </p>
                <GapDumbbell
                  domain={[0.3, 0.7]}
                  label1="Legacy 3-class"
                  label2="Binary (served)"
                  gapLabel="gain"
                  gapDirection="loso-fixed"
                  gapTone="gain"
                  rows={[
                    { metric: "Macro F1", fixed: ens.f1_mean, loso: losoBin.f1_mean },
                    { metric: "Accuracy", fixed: ens.acc_mean, loso: losoBin.acc_mean },
                    { metric: "AUC", fixed: ens.auc_mean, loso: losoBin.auc_mean },
                  ]}
                />
              </div>
              )}
              {!has3c && (
                <div className="note" style={{ marginTop: 16 }}>
                  The legacy 3-class comparison is unavailable, <code>serving_3class/</code>
                  {" "}is missing <code>model_weights.pt</code> and <code>feature_scaler.pkl</code>.
                  Add them and this section returns automatically.
                </div>
              )}
              {has3c && (
              <div className="note info" style={{ marginTop: 16 }}>
                Read the accuracy row carefully: chance is {pct(1 / 3)} with three
                classes and {pct(1 / 2)} with two, so the raw move from{" "}
                {pct(ens.acc_mean)} to {pct(losoBin.acc_mean)} is not a like-for-like
                gain. Measured <em>above chance</em> the two are much closer -{" "}
                {pct(ens.acc_mean - 1 / 3)} against {pct(losoBin.acc_mean - 0.5)}.
                AUC is the comparison that does not depend on the number of classes,
                and it moves {fmt(ens.auc_mean, 3)} → {fmt(losoBin.auc_mean, 3)}. The
                binary task is genuinely the more learnable one, but the headline
                accuracy overstates by how much.
              </div>
              )}
              <div style={{ display: "flex", gap: 12, marginTop: 14, flexWrap: "wrap" }}>
                {heroClasses.map((c) => (
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
                    <h3>{t.title}</h3>
                    <div className="teaser-stamp">{t.stamp}</div>
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
                Each input is a per-trial feature window: 540 EEG band-power
                features and 936 fNIRS (HbO + HbR) response-shape descriptors. Two
                parallel encoders compress each modality, an attention-gated fusion
                layer weighs them, and a small classifier outputs a {heroClasses.join(" / ")}{" "}
                probability.
              </p>
              <Figure num="0" title="The served model: Hybrid network only (the Random Forest ensemble is retired)." source="model_config.json">
                <ArchitectureDiagram />
              </Figure>
              <div className="grid grid-2" style={{ marginTop: 18 }}>
                <div className="note plain">
                  <strong>The encoders.</strong> Each modality gets its own small
                  MLP with a residual connection. It compresses a wide, noisy
                  feature vector into a short embedding that keeps what separates
                  load levels and discards what only describes a particular
                  person or session. Keeping them separate - rather than one
                  encoder over the concatenated 1,476 features - lets EEG and
                  fNIRS be scaled and regularized on their own terms, so the
                  slower, lower-dimensional fNIRS signal is not drowned out.
                </div>
                <div className="note plain">
                  <strong>The attention fusion.</strong> The two embeddings rarely
                  deserve equal weight: on a trial where the EEG is clean but the
                  fNIRS is motion-corrupted, EEG should carry the decision. A small
                  gate reads both embeddings together and emits a 0-1 weight for
                  every dimension of each, scaling them before they are joined. The
                  vector the classifier sees is therefore a per-trial mixture rather
                  than a flat concatenation - and those gate values come back with
                  each prediction, as a record of which modality the model leaned on.
                </div>
              </div>
            </Reveal>

            {/* Which signal it leans on */}
            {shapData?.categories?.length ? (
              <Reveal as="section" className="card">
                <h2>Which signal the model leans on</h2>
                <p>
                  Feeding a network two very different signals raises an obvious
                  question: is it using both, or is one of them along for the
                  ride? This is the answer, measured rather than assumed.
                </p>
                <Figure
                  num="1"
                  title={`Mean |attribution| by feature family, over ${
                    shapData.n_windows?.toLocaleString() || "the"
                  } held-out windows.`}
                  source="shap.json"
                >
                  <ShapChart categories={topFamilies} height={340} />
                </Figure>
                <p>
                  Each bar is one <em>family</em> of inputs - a single kind of
                  measurement, pooled across every channel that produces it.
                  &ldquo;fNIRS HbO max&rdquo; is the highest oxygenated-blood level
                  reached during a window; &ldquo;EEG alpha&rdquo; is power in the
                  alpha band. The length is how much that family moved the decision on
                  average, in either direction, so a long bar means the model
                  relied on it, not that it pushed toward High load.
                </p>
                <p>
                  The ordering is the finding. Response-<em>shape</em> descriptors
                  from fNIRS take the top places, while EEG band powers sit far
                  down the list. Two things make that credible rather than
                  convenient: the fusion layer&apos;s own attention weights reach
                  the same conclusion by a completely separate route
                  {attentionAside}, and every family near the top summarises the
                  size or timing of the blood-oxygenation swing across a window -
                  exactly what a haemodynamic response to sustained mental effort
                  should move. The model is reading physiology, not latching onto
                  one arbitrary column.
                </p>
                <p className="stat-sub">
                  {shapData.method === "integrated_gradients"
                    ? "Attribution method: integrated gradients computed on the served weights. True SHAP is unavailable in this environment, so a closely related, axiomatically-grounded method is used and named rather than substituted silently."
                    : "Attribution method: mean absolute SHAP value per feature family."}{" "}
                  Showing the {topFamilies.length} strongest of{" "}
                  {shapData.categories.length} families; the full list is on the{" "}
                  <Link to="/insights">Model Insights</Link> page.
                </p>
                <p style={{ marginTop: 16 }}>
                  Explore a <Link to="/live">single prediction</Link>, watch a{" "}
                  <Link to="/session">session over time</Link>, or open the full{" "}
                  <Link to="/dashboard">model dashboard</Link>.
                </p>
              </Reveal>
            ) : null}
          </div>
        )}
      </div>
    </>
  );
}
