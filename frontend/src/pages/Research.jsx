import { Link } from "react-router-dom";
import { useApi, Loading, ErrorBox } from "../useApi";
import { useModelVersion } from "../lib/modelVersion";
import { getResearch, getMetrics, getShap, getAttention } from "../api/client";
import { COMPARE_COLORS } from "../constants";
import SignalTrace from "../components/SignalTrace";
import SectionNav from "../components/research/SectionNav";
import Reveal from "../components/research/Reveal";
import Figure from "../components/research/Figure";
import VerdictCallout from "../components/research/VerdictCallout";
import MetricStat from "../components/research/MetricStat";
import ProtocolBenchmarkChart from "../components/research/ProtocolBenchmarkChart";
import LambdaSweepChart from "../components/research/LambdaSweepChart";
import SubjectProbeBar from "../components/research/SubjectProbeBar";
import SubjectProbeScatter from "../components/research/SubjectProbeScatter";
import CalibrationLatencyPanel from "../components/research/CalibrationLatencyPanel";
import AblationCompareChart from "../components/research/AblationCompareChart";
import ArchitectureDiagram from "../components/ArchitectureDiagram";
import DannDiagram from "../components/DannDiagram";
import ShapChart from "../components/ShapChart";
import ModalityContribution from "../components/ModalityContribution";
import AnimatedBand from "../components/AnimatedBand";

const SECTIONS = [
  { id: "problem", label: "Problem & thesis" },
  { id: "data", label: "Data & task" },
  { id: "model", label: "The model" },
  { id: "protocol", label: "Protocol taxonomy" },
  { id: "dann", label: "Novel: DANN" },
  { id: "gates", label: "Decision gates" },
  { id: "explain", label: "Explainability" },
  { id: "limits", label: "Limitations" },
];

const pct = (x) => (x == null ? "-" : `${(x * 100).toFixed(1)}%`);
const f3 = (x) => (x == null ? "-" : Number(x).toFixed(3));
const signed = (x, d = 4) => (x == null ? "-" : `${x >= 0 ? "+" : ""}${Number(x).toFixed(d)}`);

export default function Research() {
  const { version } = useModelVersion();
  const { data, loading, error } = useApi(async () => {
    const [research, metrics, shap, attention] = await Promise.all([
      // research.json is a project-level dossier, only packaged in the binary
      // bundle; fall back so selecting 3-class does not blank this page.
      getResearch(version).catch(() => getResearch("binary")),
      getMetrics(version),
      getShap(version),
      getAttention(version),
    ]);
    return { research, metrics, shap, attention };
  }, [version]);

  return (
    <>
      <header className="band-dark page-hero">
        <AnimatedBand />
        <div className="hero-inner">
          <div className="eyebrow">Capstone · research dossier</div>
          <h1 className="hero-title" style={{ marginTop: 14 }}>
            An honest reading of cognitive load from two brain signals
          </h1>
          <p className="hero-lede" style={{ marginTop: 12 }}>
            Every claim here is judged by the same rule the project applied to its
            own model: adopt a change only when it clears a paired-significance
            decision gate. That discipline - and the rigorous negative results it
            produced - is the contribution.
          </p>
          <SignalTrace className="hero-signal" height={54} strokeWidth={2.4} animate glow />
        </div>
      </header>

      <div className="container container-wide research-grid">
        <aside>
          <SectionNav sections={SECTIONS} />
        </aside>

        <main className="research-main">
          {loading && <Loading />}
          {error && <ErrorBox error={error} />}
          {data && <Body {...data} />}
        </main>
      </div>
    </>
  );
}

function Body({ research, metrics, shap, attention }) {
  const src = (key) => research?.sources?.[key]?.artifact;

  const loso = metrics?.loso || {};
  const ens = loso.ensemble_hybrid_rf || {};
  const hyb = loso.hybrid || {};
  const fx = metrics?.fixed_split?.hybrid || {};

  return (
    <div className="stack-lg">
      {/* 1 - Problem & thesis */}
      <section className="section" id="problem">
        <SecHead
          n="01"
          title="The performance ceiling is the person, not the model"
          lede="Across every experiment, the same wall appears: a model that reads one person's brain well often fails on the next. The project's thesis is to measure that honestly rather than around it."
        />
        <Reveal className="stack">
          <p>
            Published EEG/fNIRS workload accuracies of 60–90% usually answer an{" "}
            <em>easier</em> question - a binary task, or a within-subject split
            where the same person appears in training and test, letting the model
            lean on that person's physiological baseline. This project reports the
            harder setup end to end: <strong>3-class</strong> load
            (Low / Medium / High) under <strong>leave-one-subject-out</strong>{" "}
            (LOSO) evaluation, where every test subject is unseen.
          </p>
          <div className="grid grid-3">
            <div className="card card-flat">
              <MetricStat value={ens.f1_mean} format={pct} label="Served macro F1 · LOSO"
                sub="Hybrid + RF ensemble, 26 subjects held out" accent="var(--accent)" />
            </div>
            <div className="card card-flat">
              <MetricStat value={ens.acc_mean} format={pct} label="Accuracy · LOSO"
                sub="3-class, chance ≈ 33%" />
            </div>
            <div className="card card-flat">
              <MetricStat value={fx.f1_macro} format={pct} label="Macro F1 · fixed split"
                sub="reported alongside LOSO, never instead of it" />
            </div>
          </div>
          <p className="muted" style={{ fontSize: "0.9rem" }}>
            The two numbers answer different questions and are always shown
            together - fixed-split measures performance when a subject's other
            trials can appear in training; LOSO measures generalization to a new
            person. The gap between them is a finding, not a footnote.
          </p>
        </Reveal>
      </section>

      {/* 2 - Data & task */}
      <section className="section" id="data">
        <SecHead
          n="02"
          title="TU Berlin / Shin n-back - 26 people, two synchronized signals"
          lede="Each trial pairs electrical (EEG) and hemodynamic (fNIRS) recordings while a participant holds an n-back working-memory load."
        />
        <Reveal>
          <div className="grid grid-2">
            <div className="card">
              <h3>What a sample is</h3>
              <p>
                654 full-trial samples from 26 subjects. Cognitive load is the
                n-back level: <strong>0-back → Low</strong>,{" "}
                <strong>2-back → Medium</strong>, <strong>3-back → High</strong>.
              </p>
              <ul className="feat-list">
                <li><span className="swatch" style={{ background: "var(--eeg)" }} /> <strong>180 EEG features</strong> - band powers (α, β, δ, θ, γ) + engagement across 30 channels</li>
                <li><span className="swatch" style={{ background: "var(--fnirs)" }} /> <strong>468 fNIRS (HbO) features</strong> - hemodynamic response-shape descriptors</li>
              </ul>
            </div>
            <div className="card">
              <h3>Why the signals differ</h3>
              <p>
                EEG is fast and electrical - millisecond spikes. fNIRS is slow and
                hemodynamic - a blood-oxygenation curve that rises and settles over
                seconds. Fusing them is the modelling premise, and the trace in the
                header is exactly that: spikes on the left relaxing into a smooth
                response on the right.
              </p>
              <SignalTrace height={44} strokeWidth={2} />
            </div>
          </div>
        </Reveal>
      </section>

      {/* 3 - The model */}
      <section className="section" id="model">
        <SecHead
          n="03"
          title="A deliberately small fusion net, ensembled with a Random Forest"
          lede="Two modality encoders fuse under a learned attention gate; the served head averages that network with a Random Forest over the same features."
        />
        <Reveal className="stack">
          <Figure num="1" title="The served Hybrid + Random Forest ensemble." source="model_config.json">
            <ArchitectureDiagram />
          </Figure>
          <div className="grid grid-2">
            <p>
              With only 654 samples, a large network memorizes and collapses
              (train ≈ 85%, validation near chance). The Hybrid net is kept
              small (~200K parameters) and heavily regularized. On LOSO the
              probability-average <strong>ensemble</strong> ({f3(ens.f1_mean)}{" "}
              macro F1) clearly beats the Hybrid net alone ({f3(hyb.f1_mean)}),
              which is why it is the served head.
            </p>
            <VerdictCallout tone="ok" stamp="Adopted" title="Hybrid + RF ensemble">
              Ensemble LOSO macro F1 {f3(ens.f1_mean)} &gt; Hybrid {f3(hyb.f1_mean)};
              minimum mean class recall {f3(ens.recall_mean?.[2])} on High load. The
              frozen Hybrid weights are unchanged - the Random Forest is layered on
              at inference.
            </VerdictCallout>
          </div>
        </Reveal>
      </section>

      {/* 4 - Protocol taxonomy */}
      <section className="section" id="protocol">
        <SecHead
          n="04"
          title="Why 60–90% in the literature is a different, easier problem"
          lede="Run the same Random Forest across two protocol axes and the apparent gap resolves into two causes - one that this project already neutralizes, and one it deliberately keeps."
        />
        <Reveal className="stack">
          <Figure num="2" title="Same estimator, same features - only the protocol changes."
            source={src("protocol")}>
            <ProtocolBenchmarkChart protocol={research?.protocol} />
          </Figure>
          <p>
            <strong>Subject leakage is minor here.</strong> Once per-subject
            normalization is applied, within-subject and LOSO macro F1 are close -
            so the honest LOSO figure is <em>not</em> propped up by cross-subject
            leakage. <strong>Task difficulty is the real gap:</strong> collapsing
            to a binary Low-vs-High task lifts LOSO macro F1 by roughly +0.20. Much
            of the headline literature is binary and/or within-subject; this
            project reports the harder 3-class, cross-subject number on purpose.
          </p>
        </Reveal>
      </section>

      {/* 5 - DANN (novel) */}
      <section className="section" id="dann">
        <SecHead
          n="05"
          title="Novel contribution - training the representation to forget who"
          lede="A domain-adversarial network attacks the subject-variability ceiling directly: make the fused embedding predict load but not identity. Applied rigorously, it is correctly declined - and the reason is itself the finding."
        />
        <Reveal className="stack">
          <Figure num="3" title="Gradient reversal makes the encoder erase subject identity."
            source={src("dann")}>
            <DannDiagram />
          </Figure>
          <p>
            A subject-discriminator head tries to recover <em>which subject</em> a
            sample came from. A gradient-reversal layer flips its gradient, so the
            same signal pushes the encoder to discard subject-specific structure
            while keeping load-relevant structure (Ganin &amp; Lempitsky, 2015).
          </p>

          <div className="grid grid-2">
            <Figure num="4" title="Baseline vs DANN across 26 LOSO folds (macro F1)."
              source={src("dann")}>
              <AblationCompareChart
                data={[
                  { config: "Hybrid alone", base: research?.dann?.baseline?.hybrid?.f1_mean, dann: research?.dann?.dann?.hybrid?.f1_mean },
                  { config: "Ensemble (served)", base: research?.dann?.baseline?.ensemble?.f1_mean, dann: research?.dann?.dann?.ensemble?.f1_mean },
                ]}
                categoryKey="config"
                series={[
                  { key: "base", name: "Baseline (locked)", color: COMPARE_COLORS[0] },
                  { key: "dann", name: "DANN", color: COMPARE_COLORS[1] },
                ]}
                domain={[0, 0.5]}
              />
            </Figure>
            <Figure num="5" title="Adversary strength vs task performance (controlled on/off)."
              source={src("dann_lambda_sweep")}>
              <LambdaSweepChart sweep={research?.dann_lambda_sweep} />
            </Figure>
          </div>

          <SigLine
            label="Ensemble (served)"
            s={research?.dann?.significance_dann_vs_baseline?.ensemble_dann_vs_base}
          />

          <div className="grid grid-2">
            <Figure num="6" title="Subject-invariance probe - can identity still be recovered from the embedding?"
              source={src("dann")}>
              <SubjectProbeBar probe={research?.dann?.subject_invariance_probe} />
            </Figure>
            <Figure num="6b" title="Per-subject probe across all 26 LOSO folds - both hug chance."
              source={src("dann")}>
              <SubjectProbeScatter probe={research?.dann?.subject_invariance_probe} />
            </Figure>
          </div>

          <VerdictCallout tone="stop" stamp="Not adopted" title="DANN - an honest, explained negative result">
            No significant ensemble-level lift (Δ{" "}
            {signed(research?.dann?.significance_dann_vs_baseline?.ensemble_dann_vs_base?.mean_diff)},
            Wilcoxon p≈{f3(research?.dann?.significance_dann_vs_baseline?.ensemble_dann_vs_base?.wilcoxon_p)}),
            and task performance falls monotonically as the adversary strengthens.
            Three lines of evidence agree on <em>why</em>: subject leakage is minor
            here (§04), the baseline embedding is <em>already</em> near chance for
            subject-id recovery ({f3(research?.dann?.subject_invariance_probe?.baseline_probe_acc_mean)}{" "}
            vs {f3(research?.dann?.subject_invariance_probe?.chance_mean)} chance), so
            there is nothing harmful for the adversary to strip - it only injects
            noise. The cross-subject difficulty is genuine task difficulty, not a
            decodable identity confound.
          </VerdictCallout>
        </Reveal>
      </section>

      {/* 6 - Decision gates */}
      <section className="section" id="gates">
        <SecHead
          n="06"
          title="One gate, applied to every candidate - including the ones that lost"
          lede="A change enters serving only on a clear, significant, both-tests improvement on the ensemble. Two more candidates were measured and declined."
        />
        <Reveal className="stack">
          <div className="grid grid-2">
            <Figure num="7" title="Adding HbR: significant for the net alone, not robustly for the served ensemble."
              source={src("hbr")}>
              <AblationCompareChart
                data={[
                  { config: "Hybrid alone", hbo: research?.hbr?.hbo_468?.hybrid?.f1_mean, hbr: research?.hbr?.hbo_hbr_936?.hybrid?.f1_mean },
                  { config: "Ensemble (served)", hbo: research?.hbr?.hbo_468?.ensemble?.f1_mean, hbr: research?.hbr?.hbo_hbr_936?.ensemble?.f1_mean },
                ]}
                categoryKey="config"
                series={[
                  { key: "hbo", name: "EEG + HbO (468)", color: COMPARE_COLORS[0] },
                  { key: "hbr", name: "EEG + HbO + HbR (936)", color: COMPARE_COLORS[1] },
                ]}
                domain={[0, 0.55]}
              />
            </Figure>
            <Figure num="8" title="Classical sweep: XGBoost ties RF, SVM is worse."
              source={src("classical")}>
              <AblationCompareChart
                data={[
                  { model: "Random Forest", f1: research?.classical?.standalone_loso?.rf?.f1_mean },
                  { model: "XGBoost", f1: research?.classical?.standalone_loso?.xgb?.f1_mean },
                  { model: "SVM (RBF)", f1: research?.classical?.standalone_loso?.svm?.f1_mean },
                ]}
                categoryKey="model"
                series={[{ key: "f1", name: "Standalone LOSO macro F1", color: COMPARE_COLORS[0] }]}
                domain={[0, 0.5]}
              />
            </Figure>
          </div>

          <Figure num="8b" title="Inside the ensemble: RF, XGB, and RF+XGB heads on the Hybrid - nothing clears the gate."
            source={src("classical")}>
            <AblationCompareChart
              data={[
                { config: "Hybrid + RF (served)", f1: research?.classical?.task2?.ensembles?.["hybrid+rf"]?.f1_mean, recall: research?.classical?.task2?.ensembles?.["hybrid+rf"]?.min_recall },
                { config: "Hybrid + XGB", f1: research?.classical?.task2?.ensembles?.["hybrid+xgb"]?.f1_mean, recall: research?.classical?.task2?.ensembles?.["hybrid+xgb"]?.min_recall },
                { config: "Hybrid + RF + XGB", f1: research?.classical?.task2?.ensembles?.["hybrid+rf+xgb"]?.f1_mean, recall: research?.classical?.task2?.ensembles?.["hybrid+rf+xgb"]?.min_recall },
              ]}
              categoryKey="config"
              series={[
                { key: "f1", name: "Macro F1 (LOSO)", color: COMPARE_COLORS[0] },
                { key: "recall", name: "Min class recall", color: COMPARE_COLORS[1] },
              ]}
              domain={[0, 0.5]}
            />
          </Figure>

          <div className="grid grid-2">
            <VerdictCallout tone="warn" stamp="Retained: HbO-only" title="HbR ablation">
              HbR clearly helps the Hybrid net alone (Δ
              {signed(research?.hbr?.significance_936_vs_468?.hybrid_936_vs_468?.mean_diff)},
              both tests p≈0.002), but the <em>ensemble</em> - the served config the
              gate turns on - is only borderline (t p=
              {f3(research?.hbr?.significance_936_vs_468?.ensemble_936_vs_468?.t_p)},
              Wilcoxon p=
              {f3(research?.hbr?.significance_936_vs_468?.ensemble_936_vs_468?.wilcoxon_p)}),
              and on the real served split the gain is only{" "}
              {signed(research?.hbr_served_split?.delta_936_minus_468?.macro_f1, 3)} macro
              F1. Default to HbO-only for parsimony - confirmed by ablation, not
              assumption.
            </VerdictCallout>
            <VerdictCallout tone="warn" stamp="Ruled out" title="XGBoost / SVM">
              XGBoost is statistically indistinguishable from Random Forest
              (Δ{signed(research?.classical?.significance_vs_rf?.xgb?.mean_diff)},
              Wilcoxon p={f3(research?.classical?.significance_vs_rf?.xgb?.wilcoxon_p)});
              SVM is significantly worse and nearly collapses on High load. Nothing
              clearly beats the locked Hybrid + RF, so nothing in serving changed.
              That RF is this hard to beat is further evidence of a difficulty
              ceiling.
            </VerdictCallout>
          </div>

          <Figure num="8c" title="Served end-to-end: the leaner 468 model matches the 936 prototype on calibration and latency."
            source={src("hbr")}>
            <CalibrationLatencyPanel served={research?.hbr_served_split} />
          </Figure>
        </Reveal>
      </section>

      {/* 7 - Explainability */}
      <section className="section" id="explain">
        <SecHead
          n="07"
          title="Three independent lenses agree: fNIRS carries the signal"
          lede="SHAP attributions, the learned attention gate, and Random-Forest importance all point the same way - the model leans on hemodynamic response shape."
        />
        <Reveal className="stack">
          <Figure num="9" title="Mean |SHAP| by feature category - fNIRS HbO response-shape dominates."
            source="shap.json">
            <ShapChart categories={shap?.categories} height={320} />
          </Figure>
          <div className="grid grid-2">
            <div className="card">
              <h3>Learned modality trust</h3>
              <ModalityContribution
                eegShare={attention?.overall?.eeg_share}
                fnirsShare={attention?.overall?.fnirs_share}
              />
              <p className="stat-sub" style={{ marginTop: 10 }}>
                The attention gate leans slightly toward fNIRS
                ({pct(attention?.overall?.fnirs_share)} vs{" "}
                {pct(attention?.overall?.eeg_share)}) - consistent with SHAP and
                with EEG-only models repeatedly failing an AUC ≥ 0.55 gate.
              </p>
            </div>
            <div className="card">
              <h3>Physiologically plausible</h3>
              <p>
                For an n-back task the model relies on the <em>shape</em> of the
                hemodynamic response - skew, early-window rise, slope, peak - while
                retaining weaker EEG band-power cues. It is not keying on a single
                arbitrary feature. See the{" "}
                <Link to="/insights">Model Insights</Link> page for per-class
                confusion and attention.
              </p>
            </div>
          </div>
        </Reveal>
      </section>

      {/* 8 - Limitations */}
      <section className="section" id="limits">
        <SecHead
          n="08"
          title="What this does not claim"
          lede="The honest framing cuts both ways - here is where the result is bounded."
        />
        <Reveal className="stack">
          <div className="grid grid-2">
            <div className="note plain">
              <strong>Only 26 subjects.</strong> Small for strong cross-subject
              generalization claims; the ±0.08 fold-to-fold std reflects that.
            </div>
            <div className="note plain">
              <strong>High load is hardest.</strong> Fixed-split High recall is{" "}
              {pct(fx.recall?.[2])}; LOSO High recall reaches about{" "}
              {pct(ens.recall_mean?.[2])}.
            </div>
            <div className="note plain">
              <strong>Per-subject normalization is a demo simplification.</strong>{" "}
              A genuinely new person has no prior stats; the live demo reuses each
              known subject's precomputed statistics.
            </div>
            <div className="note plain">
              <strong>Cross-dataset transfer is future work.</strong> Running on
              another EEG+fNIRS dataset needs a dataset-specific feature-extraction
              pipeline first; the app rejects mismatched schemas rather than
              guessing.
            </div>
          </div>
          <p className="muted" style={{ fontSize: "0.86rem" }}>
            Every figure on this page reads from a committed artifact bundled into{" "}
            <span className="mono">serving/frontend_data/research.json</span>. See{" "}
            <Link to="/methodology">Methodology &amp; Reproducibility</Link> for the
            significance tests and run commands.
          </p>
        </Reveal>
      </section>
    </div>
  );
}

function SecHead({ n, title, lede }) {
  return (
    <div className="section-head">
      <span className="eyebrow eyebrow--accent">{n}</span>
      <h2>{title}</h2>
      {lede && <p className="lede">{lede}</p>}
    </div>
  );
}

// One-line significance readout under a comparison chart.
function SigLine({ label, s }) {
  if (!s) return null;
  return (
    <p className="mono" style={{ fontSize: "0.82rem", color: "var(--ink-2)", margin: 0 }}>
      {label}: Δ {signed(s.mean_diff)} macro F1 · paired t p={f3(s.t_p)} · Wilcoxon
      p={f3(s.wilcoxon_p)} · n={s.n} → not a significant lift
    </p>
  );
}
