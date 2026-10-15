import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import PipelineStepper, {
  type Stage,
} from "../components/pipeline/PipelineStepper";
import SectionHeader from "../components/ui/SectionHeader";
import Figure from "../components/ui/Figure";
import Callout from "../components/ui/Callout";
import StatCard from "../components/ui/StatCard";
import Reveal from "../components/ui/Reveal";
import CohortSplit from "../components/pipeline/figures/CohortSplit";
import WindowingDiagram from "../components/pipeline/figures/WindowingDiagram";
import FeatureComposition from "../components/pipeline/figures/FeatureComposition";
import ArchitectureFlow from "../components/pipeline/figures/ArchitectureFlow";
import FoldAccuracy from "../components/pipeline/figures/FoldAccuracy";
import LeakageSchematic from "../components/research/LeakageSchematic";
import ConfusionMatrixChart from "../components/ConfusionMatrixChart";
import { getMetrics, getConfusion } from "../api/client";
import { useApi } from "../useApi";
import {
  DATASET,
  FEATURES,
  LEAKAGE_CHECKS,
  MODEL,
  PROTOCOL,
  SEARCH,
} from "../lib/projectFacts";

const STAGES: Stage[] = [
  { id: "collection", label: "Data collection", hint: "26 subjects, two modalities" },
  { id: "preprocessing", label: "Preprocessing", hint: "Windowing & normalisation" },
  { id: "features", label: "Feature extraction", hint: "1,476 per window" },
  { id: "architecture", label: "Architecture", hint: "Dual encoders + fusion" },
  { id: "evaluation", label: "Training & LOSO", hint: "Leakage-audited folds" },
  { id: "findings", label: "Results", hint: "What it can actually do" },
];

const pct = (x: number | null | undefined) =>
  x == null ? "-" : `${(x * 100).toFixed(1)}%`;

type LosoBlock = {
  acc_mean?: number;
  acc_std?: number;
  f1_mean?: number;
  f1_std?: number;
  auc_mean?: number;
};
type ConfusionResponse = {
  model?: string;
  labels?: string[];
  matrix?: number[][];
};

type MetricsResponse = {
  loso?: Record<string, LosoBlock>;
  lockbox?: Record<string, LosoBlock>;
  per_class?: Record<string, { recall?: number; precision?: number; f1?: number }>;
  per_subject_summary?: {
    best?: { subject_id: number; accuracy: number };
    worst?: { subject_id: number; accuracy: number };
  };
  acceptance_reason?: string;
};

/** One pipeline stage: an anchored section with generous separation. */
function StageSection({
  id,
  index,
  overline,
  title,
  lede,
  children,
}: {
  id: string;
  index: number;
  overline: string;
  title: string;
  lede: ReactNode;
  children: ReactNode;
}) {
  return (
    <section
      id={id}
      // Clears the sticky nav + the pinned stepper when jumped to by anchor.
      className="scroll-mt-[150px] border-b border-line py-20 last:border-b-0 lg:py-28"
    >
      <div className="mx-auto max-w-(--container-wrap) px-6">
        <Reveal>
          <SectionHeader
            index={index}
            overline={overline}
            title={title}
            lede={lede}
          />
        </Reveal>
        <div className="mt-12 space-y-10">{children}</div>
      </div>
    </section>
  );
}

/** Body copy at a comfortable measure. */
function Prose({ children }: { children: ReactNode }) {
  return (
    <div className="max-w-[68ch] space-y-5 text-base text-ink-2 text-pretty">
      {children}
    </div>
  );
}

export default function Workflow() {
  const { data } = useApi<MetricsResponse>(() => getMetrics("binary"), []);
  // The API returns { model, labels, matrix }. `labels` is authoritative: the
  // binary bundle once shipped the 3-class study's 3x3 matrix, so the render
  // below refuses anything whose label count doesn't match the served model
  // rather than mislabelling someone else's errors as this model's.
  const { data: confusion } = useApi<ConfusionResponse | null>(
    () => getConfusion<ConfusionResponse>("hybrid", "binary").catch(() => null),
    []
  );

  const loso = data?.loso?.ensemble_hybrid_rf || data?.loso?.hybrid;
  const lockbox = data?.lockbox?.hybrid;
  const perClass = data?.per_class;
  const spread = data?.per_subject_summary;

  // Render the matrix only when it is square and its labels match the served
  // model's class count, otherwise it belongs to a different model.
  const confusionOk = Boolean(
    confusion?.matrix?.length &&
      confusion.labels?.length === confusion.matrix.length &&
      confusion.matrix.every((row) => row.length === confusion.labels!.length) &&
      confusion.labels.length === MODEL.numClasses
  );

  return (
    <>
      {/* ── Page header ──────────────────────────────────────────────────── */}
      <header className="border-b border-line bg-paper-edge">
        <div className="mx-auto max-w-(--container-wrap) px-6 py-16 lg:py-20">
          <p className="eyebrow-mono text-accent">Research pipeline</p>
          <h1 className="mt-4 max-w-[22ch] font-display text-3xl font-bold tracking-tight text-ink text-balance">
            From two brain signals to one honest number
          </h1>
          <p className="mt-5 max-w-[68ch] text-lg text-ink-2 text-pretty">
            Six stages, in the order they were actually done. Each one carries
            the decisions made, the figures that justify them, and, where it
            applies, what went wrong. The rail below tracks your position; the
            numbers throughout are read live from committed artifacts.
          </p>
        </div>
      </header>

      <PipelineStepper stages={STAGES} />

      {/* ── 1. Data collection ───────────────────────────────────────────── */}
      <StageSection
        id="collection"
        index={1}
        overline="Stage one"
        title="Data collection"
        lede={
          <>
            Simultaneous EEG and fNIRS recordings from {DATASET.nSubjects}{" "}
            participants performing an n-back working-memory task, from the{" "}
            {DATASET.name} dataset.
          </>
        }
      >
        <Prose>
          <p>
            Cognitive load is induced rather than observed. Participants perform
            an <strong className="text-ink">n-back</strong> task: a stream of
            stimuli arrives, and they must report whether the current one
            matches the item <em>n</em> steps back. At 0-back this is a simple
            match against a fixed target; at 3-back it demands continuous
            updating of a mental buffer, and is reliably experienced as hard.
          </p>
          <p>
            Those two conditions become the labels.{" "}
            <strong className="text-ink">0-back is Low load</strong> and{" "}
            <strong className="text-ink">3-back is High load</strong>, an
            experimental manipulation with a known ground truth, not a
            subjective rating. Both modalities are recorded at the same time
            from the same person, which is what makes fusion possible at all.
          </p>
          <p>
            Before a single model was trained, the cohort was split. This is the
            most consequential decision on the page, and it was made first
            precisely so it could not be revisited once results started arriving.
          </p>
        </Prose>

        <Figure
          num="1"
          title="Cohort composition and the development / lockbox split."
          source="RESULTS_BINARY_IMPROVEMENT.md"
        >
          <CohortSplit />
        </Figure>

        <Callout tone="caution" title="Twenty-six people is a small cohort">
          Every generalization claim on this site rests on {DATASET.nSubjects}{" "}
          participants. That is normal for simultaneous EEG–fNIRS work, where
          each session is expensive, but it is small for the claim "this works
          on people it has never seen". The wide per-subject spread later in
          this pipeline is the visible consequence.
        </Callout>
      </StageSection>

      {/* ── 2. Preprocessing ─────────────────────────────────────────────── */}
      <StageSection
        id="preprocessing"
        index={2}
        overline="Stage two"
        title="Signal preprocessing"
        lede="Continuous recordings become fixed-length windows, then are normalised per person before any model sees them."
      >
        <Prose>
          <p>
            A continuous recording is not a training example. The signal is cut
            into overlapping{" "}
            <strong className="text-ink">
              {DATASET.windowing.lengthSeconds}-second windows
            </strong>{" "}
            advanced in {DATASET.windowing.strideSeconds}-second steps, yielding{" "}
            {DATASET.nWindows.toLocaleString()} windows,{" "}
            {DATASET.windowsPerSubject} per subject, evenly split between the
            two classes.
          </p>
        </Prose>

        <Figure
          num="2"
          title="Sliding-window segmentation. Consecutive windows share half their samples."
          source="research/datamod.py"
        >
          <WindowingDiagram />
        </Figure>

        <Callout tone="finding" title="The overlap creates a leakage route">
          Because consecutive windows share five seconds of raw signal, two
          neighbours are near-duplicates. Allowing one into training and the
          other into test would inflate the score even with the subject split
          done correctly. This is why the audit checks{" "}
          <strong className="text-ink">window identity</strong>, not just
          subject identity.
        </Callout>

        <Prose>
          <p>
            The processing chain is then fixed and applied identically
            everywhere:{" "}
            <code className="rounded-xs bg-surface-2 px-1.5 py-0.5 font-mono text-sm text-ink">
              {FEATURES.preprocessing.join(" → ")}
            </code>
            .
          </p>
          <p>
            Order matters more than it looks. Per-subject normalisation runs{" "}
            <strong className="text-ink">before</strong> log compression, the
            alternative was tried, and normalising after the log gave slightly
            more balanced recall but lower LOSO macro F1, so it was reverted.
            The scalers in the final step are refitted inside every fold, on
            training rows only.
          </p>
        </Prose>

        <Callout tone="caution" title="Per-subject normalisation is demo scaffolding">
          Normalising each person against their own statistics is what makes the
          features comparable across very different heads and optode placements.
          But a genuinely new person has no statistics yet. The live demo reuses
          each known subject's precomputed values, honest scaffolding for a
          demonstration, not production behaviour, and the single largest gap
          between this system and a deployable one.
        </Callout>
      </StageSection>

      {/* ── 3. Feature extraction ────────────────────────────────────────── */}
      <StageSection
        id="features"
        index={3}
        overline="Stage three"
        title="Feature extraction"
        lede={
          <>
            Each window is reduced to{" "}
            {FEATURES.totalPerWindow.toLocaleString()} numbers, band power for
            the electrical signal, response shape for the hemodynamic one.
          </>
        }
      >
        <Prose>
          <p>
            The two modalities are informative in completely different ways, so
            they are described in completely different terms.
          </p>
          <p>
            <strong className="text-eeg">EEG</strong> is summarised as{" "}
            <strong className="text-ink">band power</strong>. Workload has
            well-documented spectral signatures, frontal theta tends to rise
            with effort while parietal alpha tends to fall, so each of the{" "}
            {FEATURES.eeg.channels} channels contributes power in{" "}
            {FEATURES.eeg.bands.length} bands, each summarised by its mean,
            spread and trend within the window.
          </p>
          <p>
            <strong className="text-fnirs">fNIRS</strong> is summarised as{" "}
            <strong className="text-ink">response shape</strong>. The
            hemodynamic response is slow and its <em>trajectory</em> carries the
            information, so each channel is described by{" "}
            {FEATURES.fnirs.descriptors.length} descriptors, area under the
            curve, early versus late means, peak-to-peak, skew, for both
            oxygenated and deoxygenated haemoglobin.
          </p>
        </Prose>

        <Figure
          num="3"
          title="Composition of one model input."
          source="serving/feature_schema.json"
        >
          <FeatureComposition />
        </Figure>

        <Callout tone="finding" title="fNIRS carries more than its share">
          fNIRS supplies roughly two-thirds of the features by count, but
          three independent explainability methods (SHAP, the fusion layer's own
          attention weights, and random-forest importance) all concentrate on
          fNIRS well beyond that proportion. The hemodynamic signal is doing
          more of the work than the raw count would suggest.{" "}
          <Link
            to="/insights"
            className="font-medium text-accent underline decoration-accent/30 underline-offset-2 hover:decoration-accent"
          >
            See the attributions →
          </Link>
        </Callout>

        <Callout tone="finding" title="Feature selection: 50 features preserve cross-subject accuracy">
          Selecting the top 50 discriminative physiological features via ANOVA F-score (SelectKBest)
          cuts cross-subject overfitting drastically — lowering the generalization gap from ~17% to 4.6% in fast linear ensembles
          while matching 62.25% test accuracy with 58× faster training in the Hybrid neural network.
          Prefrontal HbO hemodynamics carry the primary transferable signal across subjects.{" "}
          <Link
            to="/compare"
            className="font-medium text-accent underline decoration-accent/30 underline-offset-2 hover:decoration-accent"
          >
            Compare the models & feature sets →
          </Link>
        </Callout>
      </StageSection>

      {/* ── 4. Architecture ──────────────────────────────────────────────── */}
      <StageSection
        id="architecture"
        index={4}
        overline="Stage four"
        title="Model architecture"
        lede="Two parallel encoders, an attention-gated fusion layer, and a deliberately small classifier head."
      >
        <Prose>
          <p>
            Concatenating {FEATURES.totalPerWindow.toLocaleString()} raw features
            and feeding them to one network lets the larger, noisier modality
            dominate the gradient. Instead each stream gets its own projection
            and encoder, compressing to {MODEL.encDim} dimensions before the two
            ever meet.
          </p>
          <p>
            They meet at an{" "}
            <strong className="text-ink">attention-gated fusion</strong> layer,
            which learns how much to weight each modality rather than fixing the
            ratio in advance, the mechanism that was supposed to let the model
            lean on fNIRS when EEG is noisy.
          </p>
        </Prose>

        <Figure
          num="4"
          title="The served architecture. Widths are the actual configured dimensions."
          source="serving/model_config.json"
        >
          <ArchitectureFlow />
        </Figure>

        <Prose>
          <p>
            The head is small and the regularisation is heavy: dropout of{" "}
            {MODEL.encoderDropout} in the encoders and {MODEL.fcDropout} in the
            classifier, on a network of roughly {MODEL.approxParams} parameters.
            With {DATASET.nWindows.toLocaleString()} windows from only{" "}
            {DATASET.nSubjects} people, the binding constraint is overfitting to
            individuals, not model capacity, a reading the search history in
            stage six went on to confirm the hard way.
          </p>
        </Prose>

        <Callout tone="negative" stamp="Against the hypothesis" title="Fusion did not beat fNIRS alone">
          On unseen subjects the full Hybrid network does{" "}
          <strong className="text-ink">not</strong> outperform an fNIRS-only
          model; fNIRS-only is marginally ahead on macro F1, significantly so by
          Wilcoxon. Multimodality only pulled ahead once the Hybrid was combined
          with a random forest in the 3-class study. This contradicted the
          project's founding premise and is reported rather than buried.
        </Callout>
      </StageSection>

      {/* ── 5. Training & evaluation ─────────────────────────────────────── */}
      <StageSection
        id="evaluation"
        index={5}
        overline="Stage five"
        title="Training & evaluation"
        lede="Leave-one-subject-out, with a leakage audit that aborts the run rather than let it produce a flattering score."
      >
        <Prose>
          <p>
            This stage is the project's actual contribution, so it is worth
            being precise about what goes wrong without it.
          </p>
          <p>
            A conventional random train/test split puts different windows{" "}
            <em>from the same person</em> on both sides. The model can then
            identify the individual, their skull thickness, optode placement,
            resting rhythm, and recover the label from that, without learning
            anything about workload. The score looks strong and means nothing
            for a new user.
          </p>
          <p>
            <strong className="text-ink">
              Leave-one-subject-out (LOSO)
            </strong>{" "}
            closes it: every subject is held out in turn, trained on the other{" "}
            {DATASET.nSubjects - 1}, and never appears in their own training
            data. It is {DATASET.nSubjects} times the compute for a lower
            number, and it is the only number worth quoting.
          </p>
        </Prose>

        <Figure
          num="5"
          title="Why a fixed split leaks subject identity, and how LOSO closes it."
          source="protocol_benchmark.py"
        >
          <LeakageSchematic />
        </Figure>

        <Prose>
          <p>
            Protocol discipline is not left to intent. Before any fold is
            allowed to train, an audit asserts every one of these, and a
            failure raises <code className="font-mono text-sm text-ink">LeakageError</code> and
            aborts the run, so a leaked score is never produced in the first
            place:
          </p>
        </Prose>

        <ul className="grid max-w-4xl gap-x-8 gap-y-2.5 sm:grid-cols-2">
          {LEAKAGE_CHECKS.map((check) => (
            <li key={check} className="flex gap-3 text-sm text-ink-2">
              <span aria-hidden="true" className="mt-1.5 text-ok">
                ✓
              </span>
              {check}
            </li>
          ))}
        </ul>

        <Callout tone="finding" title="The audit is itself tested">
          An audit that never fires proves nothing. A dedicated test suite
          injects each kind of leakage deliberately, a duplicated row, a
          scaler fitted on the test split, a subject present on both sides, and
          confirms the audit catches every one.
        </Callout>

        <Prose>
          <p>
            Selection discipline follows the same logic. All tuning happens on{" "}
            {PROTOCOL.devFolds}-fold LOSO over the development pool. The{" "}
            {PROTOCOL.lockboxSubjects.length}-subject lockbox is opened once,
            after the configuration is frozen. And a change is adopted only if
            it clears the baseline by at least one standard error, a bar
            deliberately set above the noise floor.
          </p>
        </Prose>

        <Figure
          num="6"
          title="Accuracy on every held-out subject, 26-fold LOSO. Bars below chance are shown in red."
          source="outputs/runs/ · BASELINE_loso26"
        >
          <FoldAccuracy />
        </Figure>

        <Callout tone="caution" title="The mean hides the person">
          The headline figure is an average over people who differ enormously.
          {spread?.best && spread?.worst ? (
            <>
              {" "}The same frozen model reaches{" "}
              <strong className="text-ink">
                {pct(spread.best.accuracy)}
              </strong>{" "}
              on subject {spread.best.subject_id} and falls to{" "}
              <strong className="text-ink">
                {pct(spread.worst.accuracy)}
              </strong>{" "}
              on subject {spread.worst.subject_id}.
            </>
          ) : null}{" "}
          It estimates population-level reliability. It is not a guarantee for
          any individual, and should never be read as one.
        </Callout>
      </StageSection>

      {/* ── 6. Findings ──────────────────────────────────────────────────── */}
      <StageSection
        id="findings"
        index={6}
        overline="Stage six"
        title="Results & findings"
        lede="What the model does on people it has never seen, what fifty-eight attempts to improve it produced, and what remains open."
      >
        <div className="grid gap-px overflow-hidden rounded-md border border-line bg-line sm:grid-cols-2 lg:grid-cols-4">
          <div className="bg-surface">
            <StatCard
              value={pct(loso?.acc_mean)}
              label="Accuracy · 26-fold LOSO"
              sub={
                loso?.acc_std ? `± ${(loso.acc_std * 100).toFixed(1)} pts` : "chance 50%"
              }
              accent
            />
          </div>
          <div className="bg-surface">
            <StatCard
              value={loso?.f1_mean != null ? loso.f1_mean.toFixed(4) : "-"}
              label="Macro F1 · LOSO"
              sub={loso?.f1_std ? `± ${loso.f1_std.toFixed(3)}` : undefined}
            />
          </div>
          <div className="bg-surface">
            <StatCard
              value={loso?.auc_mean != null ? loso.auc_mean.toFixed(3) : "-"}
              label="ROC-AUC · LOSO"
              sub="0.5 = uninformative"
            />
          </div>
          <div className="bg-surface">
            <StatCard
              value={pct(lockbox?.acc_mean)}
              label="Accuracy · lockbox"
              sub={`${PROTOCOL.lockboxSubjects.length} subjects, opened once`}
            />
          </div>
        </div>

        <Prose>
          <p>
            The lockbox agreeing with the LOSO figure is the single most
            reassuring result here. The development pool drove{" "}
            {SEARCH.totalConfigs} configurations' worth of decisions and could
            easily have been overfitted by the selection process itself; that
            the sealed subjects land in the same place says the number is real.
          </p>
        </Prose>

        {confusionOk && (
          <Figure
            num="7"
            title="Confusion matrix, pooled across all 4,576 held-out windows. Rows are the true class."
            source="serving/frontend_data/confusion_matrices.json"
          >
            <ConfusionMatrixChart
              matrix={confusion!.matrix}
              classNames={confusion!.labels}
              title={null}
            />
          </Figure>
        )}

        {perClass && (
          <Prose>
            <p>
              The errors are not symmetric. Recall on Low load is{" "}
              {pct(perClass.Low?.recall)} against {pct(perClass.High?.recall)}{" "}
              on High, the model is readier to call a window Low. For a
              workload monitor that is the less useful direction to be biased
              in, since missing high load is the failure that matters.
            </p>
          </Prose>
        )}

        <Prose>
          <p>
            Read the campaign in full on the{" "}
            <Link
              to="/evaluation"
              className="font-medium text-accent underline decoration-accent/30 underline-offset-2 hover:decoration-accent"
            >
              evaluation page
            </Link>
            , compare all evaluated models on the{" "}
            <Link
              to="/compare"
              className="font-medium text-accent underline decoration-accent/30 underline-offset-2 hover:decoration-accent"
            >
              model comparison page
            </Link>
            , or see how the honest protocol compares to the literature's
            headline numbers in the{" "}
            <Link
              to="/methodology"
              className="font-medium text-accent underline decoration-accent/30 underline-offset-2 hover:decoration-accent"
            >
              methodology page
            </Link>
            .
          </p>
        </Prose>

        <Reveal>
          <div className="mt-6 flex flex-wrap items-center justify-between gap-6 rounded-md border border-line bg-surface p-8">
            <div>
              <h3 className="font-display text-xl font-semibold text-ink">
                See it run on a held-out window
              </h3>
              <p className="mt-2 max-w-[52ch] text-base text-ink-2">
                The demo predicts on real held-out windows from subjects the
                model never trained on, and reports its confidence honestly,
                including when it should abstain.
              </p>
            </div>
            <Link
              to="/live"
              className="group inline-flex shrink-0 items-center gap-2.5 rounded-sm bg-accent px-6 py-3.5 text-sm font-semibold text-white transition-colors duration-200 hover:bg-accent-ink focus-visible:outline-2 focus-visible:outline-offset-3 focus-visible:outline-accent"
            >
              Open the live demo
              <span
                aria-hidden="true"
                className="transition-transform duration-200 ease-signal group-hover:translate-x-1"
              >
                →
              </span>
            </Link>
          </div>
        </Reveal>
      </StageSection>
    </>
  );
}
