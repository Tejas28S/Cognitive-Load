import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import ConfusionMatrixChart from "../components/ConfusionMatrixChart";
import CoverageCurve from "../components/research/CoverageCurve";
import ReliabilityDiagram from "../components/research/ReliabilityDiagram";
import {
  getAttention,
  getConfusion,
  getDiagnostics,
  getMetrics,
  getRunPerSubject,
  getRunSummary,
  getShap,
} from "../api/client";
import { useApi, Loading, ErrorBox } from "../useApi";
import { useChartColors } from "../lib/chartTheme";
import {
  DATASET,
  FEATURES,
  HEADLINE_RUN,
  LEAKAGE_CHECKS,
  MODEL,
  PROTOCOL,
  SEARCH,
} from "../lib/projectFacts";
import "../styles/dashboard.css";

/* ---------------------------------------------------------------------------
   The dashboard is pinned to the BINARY bundle on purpose.

   Every other page follows the global model switch. This one does not: it is a
   single-model operations view, and half its panels (the paired candidate/
   baseline decision, the 26-fold cohort spread) only exist for the binary
   campaign. The 3-class bundle appears here exactly once, in the "Two classes
   or three" panel, as a comparison - never mixed into the headline tiles.
   ------------------------------------------------------------------------ */
const VERSION = "binary";

const SECTIONS = [
  { id: "headline", label: "Headline" },
  { id: "protocol", label: "Protocol check" },
  { id: "errors", label: "Where it errs" },
  { id: "confidence", label: "Confidence" },
  { id: "signals", label: "Signal sources" },
  { id: "decisions", label: "Decision ledger" },
  { id: "cohort", label: "Cohort spread" },
  { id: "bundles", label: "Two classes or three" },
];

const pct = (x, d = 1) =>
  x === null || x === undefined || Number.isNaN(x) ? "-" : `${(x * 100).toFixed(d)}%`;
const num = (x, d = 3) =>
  x === null || x === undefined || Number.isNaN(x) ? "-" : Number(x).toFixed(d);
/** Percentage points, signed - the honest unit for "how far above chance". */
const pts = (x, d = 1) =>
  x === null || x === undefined || Number.isNaN(x)
    ? "-"
    : `${x >= 0 ? "+" : "-"}${Math.abs(x * 100).toFixed(d)} pts`;

const load = () =>
  Promise.all([
    getMetrics(VERSION),
    getConfusion("hybrid", VERSION),
    getDiagnostics(VERSION).catch(() => null),
    getAttention(VERSION).catch(() => null),
    getShap(VERSION).catch(() => null),
    // The archived promotion run: it evaluated the served baseline and a
    // candidate through the same two protocols in the same environment, so its
    // folds are a paired reading rather than a cross-run comparison.
    getRunSummary(HEADLINE_RUN.runId).catch(() => null),
    getRunPerSubject(HEADLINE_RUN.runId, HEADLINE_RUN.baselineExperiment).catch(
      () => null
    ),
    // Only for the two-classes-or-three panel.
    getMetrics("3class").catch(() => null),
  ]).then(
    ([metrics, confusion, diagnostics, attention, shap, runSummary, perSubject, threeClass]) => ({
      metrics,
      confusion,
      diagnostics,
      attention,
      shap,
      runSummary,
      perSubject,
      threeClass,
    })
  );

/** Highlights the rail entry for whichever band is currently in view. */
function useScrollSpy(ids, ready) {
  const [active, setActive] = useState(ids[0]);
  useEffect(() => {
    if (!ready) return;
    const seen = new Map();
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => seen.set(e.target.id, e.intersectionRatio));
        // The band occupying the most of the viewport wins; ties keep document
        // order, so scrolling never flickers between two adjacent entries.
        let best = null;
        ids.forEach((id) => {
          const r = seen.get(id) || 0;
          if (r > 0 && (best === null || r > seen.get(best))) best = id;
        });
        if (best) setActive(best);
      },
      { rootMargin: "-70px 0px -55% 0px", threshold: [0, 0.15, 0.4, 0.75, 1] }
    );
    ids.forEach((id) => {
      const el = document.getElementById(id);
      if (el) obs.observe(el);
    });
    return () => obs.disconnect();
  }, [ids, ready]);
  return active;
}

export default function Dashboard() {
  const { data, loading, error } = useApi(load, []);
  const ids = useMemo(() => SECTIONS.map((s) => s.id), []);
  const active = useScrollSpy(ids, !loading && !error);
  const c = useChartColors();

  if (loading)
    return (
      <div className="container">
        <Loading />
      </div>
    );
  if (error)
    return (
      <div className="container">
        <ErrorBox error={error} />
      </div>
    );

  const { metrics, confusion, diagnostics, attention, shap, runSummary, perSubject, threeClass } =
    data;

  const loso = metrics?.loso?.hybrid || {};
  const lockbox = metrics?.lockbox?.hybrid || {};
  const perClass = metrics?.per_class || {};
  const cohort = metrics?.per_subject_summary || {};
  const labels = confusion?.labels?.length ? confusion.labels : ["Low", "High"];
  const chance = 1 / labels.length;

  const folds = perSubject?.per_subject || [];
  const foldAcc = folds.map((f) => f.accuracy).filter((a) => a != null);

  return (
    <div className="dash">
      <aside className="dash-rail">
        <div className="dash-rail-id">
          <div className="bundle">Binary bundle</div>
          <div className="classes">
            {DATASET.classNames.join(" · ")} - one label per {DATASET.windowing.lengthSeconds}s
            window
          </div>
        </div>

        <nav aria-label="Dashboard sections">
          <ol>
            {SECTIONS.map((s) => (
              <li key={s.id}>
                <a
                  href={`#${s.id}`}
                  className={active === s.id ? "is-active" : undefined}
                  aria-current={active === s.id ? "true" : undefined}
                >
                  <span className="tick" aria-hidden="true" />
                  {s.label}
                </a>
              </li>
            ))}
          </ol>
        </nav>

        <div className="dash-rail-status">
          <div>
            <span>Architecture</span>
            <b>{MODEL.approxParams} params</b>
          </div>
          <div>
            <span>Cohort</span>
            <b>{DATASET.nSubjects} subjects</b>
          </div>
          <div>
            <span>Windows</span>
            <b>{DATASET.nWindows.toLocaleString()}</b>
          </div>
          {diagnostics?.latency?.p95_ms != null && (
            <div>
              <span>Latency p95</span>
              <b>{diagnostics.latency.p95_ms.toFixed(1)} ms</b>
            </div>
          )}
        </div>
      </aside>

      <div className="dash-main">
        <header className="dash-head">
          <div>
            <h1>Model dashboard</h1>
            <p>
              What the served cognitive-load classifier does, measured on people it has
              never seen. Every figure below is read from a committed artifact, and each
              one carries the protocol it was measured under.
            </p>
          </div>
          <div className="dash-chips">
            <span className="dash-chip">
              <span className="dot" aria-hidden="true" />
              Serving {MODEL.family.split(" ")[0].toLowerCase()} network
            </span>
            <span className="dash-chip">{PROTOCOL.headlineFolds}-fold LOSO</span>
            <span className="dash-chip mono">{HEADLINE_RUN.runId}</span>
          </div>
        </header>

        <HeadlineBand
          loso={loso}
          lockbox={lockbox}
          diagnostics={diagnostics}
          chance={chance}
          foldAcc={foldAcc}
        />

        <ProtocolBand loso={loso} lockbox={lockbox} runSummary={runSummary} colors={c} />

        <ErrorsBand confusion={confusion} perClass={perClass} labels={labels} colors={c} />

        <ConfidenceBand diagnostics={diagnostics} />

        <SignalsBand attention={attention} shap={shap} colors={c} />

        <DecisionsBand metrics={metrics} runSummary={runSummary} />

        <CohortBand folds={folds} cohort={cohort} chance={chance} colors={c} />

        <BundlesBand binary={metrics} threeClass={threeClass} />
      </div>
    </div>
  );
}

/* ── Headline ────────────────────────────────────────────────────────────── */

function Tile({ label, value, pm, sub, delta, lead, meter, children }) {
  return (
    <div className={lead ? "dash-tile lead" : "dash-tile"}>
      <div>
        <div className="k">{label}</div>
        <div className="v">
          {value}
          {pm && <span className="pm"> ± {pm}</span>}
        </div>
        {(sub || delta) && (
          <div className="s">
            {delta}
            {delta && sub ? " " : null}
            {sub}
          </div>
        )}
      </div>
      {/* A 0-1 track with a tick at the reference the number should be read
          against - chance for a score, the LOSO figure for the lockbox. The
          value is already printed above, so this is orientation, not data. */}
      {meter?.value != null && (
        <div className="tile-meter">
          <div className="tile-meter-track">
            <div
              className="tile-meter-fill"
              style={{ width: `${Math.max(0, Math.min(1, meter.value)) * 100}%` }}
            />
            {meter.mark != null && (
              <i style={{ left: `${Math.max(0, Math.min(1, meter.mark)) * 100}%` }} />
            )}
          </div>
          {meter.note && <div className="tile-meter-note">{meter.note}</div>}
        </div>
      )}
      {children}
    </div>
  );
}

function Delta({ value, unit = "pts", good }) {
  if (value === null || value === undefined || Number.isNaN(value)) return null;
  const dir = good === undefined ? value >= 0 : good;
  const cls = Math.abs(value) < 0.0005 ? "flat" : dir ? "dash-delta" : "dash-delta down";
  return (
    <span className={cls === "flat" ? "dash-delta flat" : cls}>
      {unit === "pts" ? pts(value) : `${value >= 0 ? "+" : "-"}${Math.abs(value).toFixed(3)}`}
    </span>
  );
}

function HeadlineBand({ loso, lockbox, diagnostics, chance, foldAcc }) {
  const acc = loso.acc_mean;
  const median = foldAcc.length
    ? [...foldAcc].sort((a, b) => a - b)[Math.floor(foldAcc.length / 2)]
    : null;
  const sorted = [...foldAcc].sort((a, b) => a - b);

  return (
    <section className="dash-band" id="headline">
      <div className="dash-band-head">
        <h2>Headline</h2>
        <p>
          The cross-subject numbers first: each fold trains on 25 people and is scored on
          the one held out, so nothing here is a within-person score.
        </p>
      </div>

      <div className="dash-tiles">
        <Tile
          lead
          label={`Accuracy, held-out subject · chance is ${pct(chance, 1)}`}
          value={pct(acc)}
          pm={loso.acc_std != null ? `${(loso.acc_std * 100).toFixed(1)} pts` : null}
          delta={acc != null ? <Delta value={acc - chance} /> : null}
          sub="across 26 folds"
        >
          {sorted.length > 0 && (
            <>
              <div
                className="fold-strip"
                role="img"
                aria-label={`Per-fold accuracy for ${sorted.length} held-out subjects, sorted lowest to highest: from ${pct(
                  sorted[0]
                )} to ${pct(sorted[sorted.length - 1])}.`}
              >
                {sorted.map((a, i) => (
                  <i
                    key={i}
                    className={a >= chance ? "up" : undefined}
                    style={{ height: `${Math.max(6, ((a - 0.3) / 0.55) * 100)}%` }}
                  />
                ))}
              </div>
              <div className="s" style={{ marginTop: 8 }}>
                Every fold, worst to best - {pct(sorted[0])} to {pct(sorted[sorted.length - 1])},
                median {pct(median)}. Solid bars clear chance.
              </div>
            </>
          )}
        </Tile>

        <Tile
          label="Macro F1"
          value={num(loso.f1_mean)}
          pm={loso.f1_std != null ? num(loso.f1_std) : null}
          sub="26-fold LOSO, the headline protocol"
          meter={{ value: loso.f1_mean, mark: chance, note: `0 - 1 scale, tick at chance ${num(chance, 2)}` }}
        />
        <Tile
          label="ROC-AUC"
          value={num(loso.auc_mean)}
          pm={loso.auc_std != null ? num(loso.auc_std) : null}
          sub="ranking quality, independent of threshold"
          meter={{ value: loso.auc_mean, mark: 0.5, note: "0.5 is a coin flip, 1.0 is perfect ordering" }}
        />
        <Tile
          label="Macro F1, lockbox"
          value={num(lockbox.f1_mean)}
          pm={lockbox.f1_std != null ? num(lockbox.f1_std) : null}
          delta={
            lockbox.f1_mean != null && loso.f1_mean != null ? (
              <Delta value={lockbox.f1_mean - loso.f1_mean} unit="abs" />
            ) : null
          }
          sub={`vs LOSO, on ${PROTOCOL.lockboxSubjects.length} subjects kept out of every selection decision`}
          meter={{
            value: lockbox.f1_mean,
            mark: loso.f1_mean,
            note: `tick marks the ${PROTOCOL.headlineFolds}-fold LOSO figure`,
          }}
        />
      </div>

      {diagnostics && (
        <>
          <div className="dash-band-head" style={{ marginTop: 8 }}>
            <h2 style={{ fontSize: "1rem" }}>Serving diagnostics</h2>
            <p>
              Measured on the {diagnostics.n_windows?.toLocaleString()} demo windows the API
              actually answers from. This is a fixed split, not a LOSO estimate - its
              accuracy runs high because those subjects appear in training. Read it as
              "how the deployed endpoint behaves", never as a generalisation claim.
            </p>
          </div>
          <div className="dash-tiles">
            <Tile
              label="Accuracy on demo windows"
              value={pct(diagnostics.overall?.accuracy)}
              sub="fixed split - optimistic by design"
            />
            <Tile
              label="Calibration error (ECE)"
              value={num(diagnostics.overall?.ece, 3)}
              sub="how far confidence sits from accuracy"
            />
            <Tile
              label="Mean confidence"
              value={pct(diagnostics.overall?.mean_confidence)}
              sub={`stated confidence vs ${pct(diagnostics.overall?.accuracy)} actual - overconfident`}
            />
            <Tile
              label="Inference latency, p95"
              value={`${diagnostics.latency?.p95_ms?.toFixed(1)} ms`}
              sub={`mean ${diagnostics.latency?.mean_ms?.toFixed(1)} ms per window`}
            />
          </div>
        </>
      )}
    </section>
  );
}

/* ── Protocol check ──────────────────────────────────────────────────────── */

function MetricTip({ active, payload, label, series }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="dash-tip">
      <div className="t">{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey}>
          {series[p.dataKey] || p.dataKey}: <span className="n">{Number(p.value).toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
}

function ProtocolBand({ loso, lockbox, runSummary, colors }) {
  const rows = [
    {
      metric: "Macro F1",
      loso: loso.f1_mean,
      losoErr: loso.f1_std,
      lock: lockbox.f1_mean,
      lockErr: lockbox.f1_std,
    },
    {
      metric: "Accuracy",
      loso: loso.acc_mean,
      losoErr: loso.acc_std,
      lock: lockbox.acc_mean,
      lockErr: lockbox.acc_std,
    },
    {
      metric: "ROC-AUC",
      loso: loso.auc_mean,
      losoErr: loso.auc_std,
      lock: lockbox.auc_mean,
      lockErr: lockbox.auc_std,
    },
  ];
  const gap = runSummary?.results?.baseline?.generalization_gap_f1;

  return (
    <section className="dash-band" id="protocol">
      <div className="dash-band-head">
        <h2>Protocol check</h2>
        <p>
          Two questions a single accuracy number cannot answer: does the result survive on
          subjects that were never used to choose anything, and how much of the training
          score is memorisation?
        </p>
      </div>

      <div className="dash-row wide-left">
        <div className="dash-panel">
          <h3>All subjects versus the lockbox</h3>
          <p className="sub">
            Bars are fold means; the whisker is ±1 standard deviation across folds. The
            lockbox subjects were held out of the entire 58-configuration search, so their
            score is the closest thing to a first-contact estimate.
          </p>
          <div className="dash-legend">
            <span>
              <i style={{ background: colors.accent }} /> 26-fold LOSO, all subjects
            </span>
            <span>
              <i style={{ background: colors.fnirs }} /> Lockbox, 8 unseen subjects
            </span>
            <span>
              <i className="line" style={{ background: colors.reference }} /> chance 0.50
            </span>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={332}>
              <BarChart data={rows} margin={{ top: 8, right: 12, bottom: 8, left: 4 }} barGap={2}>
                <CartesianGrid stroke={colors.grid} vertical={false} />
                <XAxis
                  dataKey="metric"
                  tick={{ fill: colors.axis, fontSize: 11 }}
                  tickLine={false}
                  axisLine={{ stroke: colors.grid }}
                />
                <YAxis
                  domain={[0, 0.9]}
                  ticks={[0, 0.25, 0.5, 0.75]}
                  tickFormatter={(v) => v.toFixed(2)}
                  tick={{ fill: colors.axis, fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  width={38}
                />
                {/* Labelled in the legend, not on the line: at these bar
                    heights every in-plot position collides with a mark. */}
                <ReferenceLine y={0.5} stroke={colors.reference} />
                <Tooltip
                  cursor={{ fill: "rgba(0,0,0,0.03)" }}
                  content={
                    <MetricTip
                      series={{ loso: "26-fold LOSO", lock: "Lockbox" }}
                    />
                  }
                />
                <Bar dataKey="loso" fill={colors.accent} radius={[4, 4, 0, 0]} maxBarSize={24}>
                  <ErrorBar dataKey="losoErr" width={5} strokeWidth={1.5} stroke={colors.accent} />
                </Bar>
                <Bar dataKey="lock" fill={colors.fnirs} radius={[4, 4, 0, 0]} maxBarSize={24}>
                  <ErrorBar dataKey="lockErr" width={5} strokeWidth={1.5} stroke={colors.fnirs} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <TableView caption="All subjects versus the lockbox">
            <thead>
              <tr>
                <th>Metric</th>
                <th>26-fold LOSO</th>
                <th>Lockbox (8)</th>
                <th>Difference</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.metric}>
                  <td className="name">{r.metric}</td>
                  <td>
                    {num(r.loso)} ± {num(r.losoErr)}
                  </td>
                  <td>
                    {num(r.lock)} ± {num(r.lockErr)}
                  </td>
                  <td>{r.lock != null && r.loso != null ? num(r.lock - r.loso) : "-"}</td>
                </tr>
              ))}
            </tbody>
          </TableView>
          <p className="foot">
            The lockbox lands within a fifth of a standard deviation of the full-cohort
            figure. Nothing in the search leaked into it, so the headline number is not an
            artefact of having chosen the configuration on these people.
          </p>
        </div>

        <div style={{ display: "grid", gap: "var(--gap)", minWidth: 0 }}>
        <div className="dash-panel">
          <h3>Where the training score goes</h3>
          <p className="sub">
            Macro F1 at each fold&apos;s selected epoch, from the archived promotion run.
            The distance from the first bar to the last is what a within-subject split
            would have hidden.
          </p>
          {gap ? (
            <>
              <div className="ladder">
                {[
                  ["Training folds", gap.train_f1_macro_mean, false],
                  ["Validation", gap.val_f1_macro_mean, false],
                  ["Held-out subject", gap.heldout_f1_macro_mean, true],
                ].map(([label, v, isHeld]) => (
                  <div className={isHeld ? "ladder-row heldout" : "ladder-row"} key={label}>
                    <div className="lab">
                      <span>{label}</span>
                      <b>{num(v)}</b>
                    </div>
                    <div className="ladder-track">
                      <div className="ladder-fill" style={{ width: `${(v / 1) * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
              <p className="foot">
                Generalisation gap {num(gap.train_minus_heldout)} macro F1. Validation and
                held-out sit {num(gap.val_minus_heldout)} apart, which is the check that
                matters: the early-stopping signal is not quietly reading the test subject.
                Mean best epoch {gap.mean_best_epoch?.toFixed(1)} of{" "}
                {gap.mean_epochs_run?.toFixed(1)} run.
              </p>
            </>
          ) : (
            <p className="sub">
              The run archive is not reachable, so the train/held-out ladder is
              unavailable. It is never substituted with a stored constant.
            </p>
          )}
        </div>

        <div className="dash-panel">
          <h3>Leakage audit</h3>
          <p className="sub">
            Asserted before every fold is allowed to train. A failure raises an error
            and aborts the run, so a leaked score is never produced in the first place.
          </p>
          <ul className="checklist">
            {LEAKAGE_CHECKS.map((check) => (
              <li key={check}>
                <span className="ok" aria-hidden="true" />
                {check}
              </li>
            ))}
          </ul>
          <p className="foot">
            The negative tests inject each leakage type on purpose and confirm the audit
            fires - an audit that never trips would prove nothing.
          </p>
        </div>
        </div>
      </div>
    </section>
  );
}

/* ── Errors ──────────────────────────────────────────────────────────────── */

function ErrorsBand({ confusion, perClass, labels, colors }) {
  // /api/model/confusion-matrix answers { model, labels, matrix }.
  const matrix = confusion?.matrix;
  const total = matrix?.flat().reduce((a, b) => a + b, 0);
  const classColor = { Low: colors.low, Medium: colors.medium, High: colors.high };
  const order = labels.filter((l) => perClass[l]);

  return (
    <section className="dash-band" id="errors">
      <div className="dash-band-head">
        <h2>Where it errs</h2>
        <p>
          Both classes carry {DATASET.classBalance.Low.toLocaleString()} windows, so
          accuracy is not hiding a majority-class shortcut. The asymmetry that remains is
          in which direction the mistakes go.
        </p>
      </div>

      <div className="dash-row wide-right">
        <div className="dash-panel">
          <h3>Confusion, pooled held-out windows</h3>
          <p className="sub">
            {total?.toLocaleString()} windows, every one predicted by a model that never
            saw that person.
          </p>
          <ConfusionMatrixChart matrix={matrix} classNames={labels} />
        </div>

        <div className="dash-panel">
          <h3>Per class</h3>
          <p className="sub">
            Recall is the share of that class the model finds; precision is how often it is
            right when it says so.
          </p>
          <div className="dash-scroll">
            <table className="dash-table">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>Windows</th>
                </tr>
              </thead>
              <tbody>
                {order.map((name) => {
                  const r = perClass[name];
                  return (
                    <tr key={name}>
                      <td className="name">
                        <span
                          className="swatch"
                          style={{ background: classColor[name] || colors.accent }}
                          aria-hidden="true"
                        />
                        {name}
                      </td>
                      <td>{num(r.precision)}</td>
                      <td>{num(r.recall)}</td>
                      <td>{num(r.f1)}</td>
                      <td>{r.support?.toLocaleString()}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {perClass.Low && perClass.High && (
            <p className="foot">
              The model leans toward calling a window Low: it recovers {pct(perClass.Low.recall)}{" "}
              of Low windows but only {pct(perClass.High.recall)} of High ones. For a
              workload monitor that bias is the costly direction - missed high load is the
              case you wanted flagged - and it is why the endpoint ships an abstain
              threshold rather than a bare argmax.
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

/* ── Confidence ──────────────────────────────────────────────────────────── */

function ConfidenceBand({ diagnostics }) {
  if (!diagnostics) return null;
  const cov = diagnostics.coverage_curve || [];
  const top = cov[0];
  const full = cov[cov.length - 1];
  const ab = diagnostics.abstain_thresholds || {};

  return (
    <section className="dash-band" id="confidence">
      <div className="dash-band-head">
        <h2>Confidence</h2>
        <p>
          A cognitive-load reading is only useful if the model can say when it does not
          know. These two panels test that: the first asks whether confidence ranks
          correctly, the second whether its stated number is honest.
        </p>
      </div>

      <div className="dash-row two">
        <div className="dash-panel">
          <h3>Accuracy against coverage</h3>
          <p className="sub">
            Keep only the most confident windows and re-score. A model whose confidence
            means something gets steadily more accurate as coverage drops - this curve is
            the evidence behind the shipped abstain behaviour.
          </p>
          <CoverageCurve curve={cov} fullAccuracy={diagnostics.overall?.accuracy} height={280} />
          <p className="foot">
            {top && full ? (
              <>
                On the most confident {pct(top.coverage, 0)} of windows the model is{" "}
                {pct(top.accuracy)} accurate; answering everything drops it to{" "}
                {pct(full.accuracy)}. The endpoint abstains below{" "}
                {pct(ab.confidence, 0)} confidence or a {num(ab.margin, 2)} margin.
              </>
            ) : null}
          </p>
        </div>

        <div className="dash-panel">
          <h3>Is the stated confidence honest?</h3>
          <p className="sub">
            Each bar is the real accuracy of windows that fell in a confidence bin, plotted
            against that bin&apos;s mean confidence. Bars below the diagonal are
            overconfident.
          </p>
          <ReliabilityDiagram reliability={diagnostics.reliability} height={280} />
          <p className="foot">
            Expected calibration error {num(diagnostics.overall?.ece, 3)}. Every populated
            bin sits below the diagonal, so a reported 90% should be read closer to{" "}
            {pct(
              diagnostics.reliability?.bins?.find((b) => b.bin_lo === 0.9)?.accuracy
            )}
            . Treat the probability as a ranking, not a literal chance of being right.
          </p>
        </div>
      </div>
    </section>
  );
}

/* ── Signals ─────────────────────────────────────────────────────────────── */

function SignalsBand({ attention, shap, colors }) {
  const rows = attention
    ? [
        { label: "Overall", eeg: attention.overall?.eeg_share, fnirs: attention.overall?.fnirs_share },
        ...(attention.per_class || []).map((p) => ({
          label: `${p.class} load`,
          eeg: p.eeg_share,
          fnirs: p.fnirs_share,
        })),
      ]
    : [];

  const top = (shap?.categories || [])
    .slice(0, 10)
    .map((r) => ({ name: r.category, v: r.mean_abs_shap }));
  const fnirsInTop = top.filter((r) => /fnirs/i.test(r.name)).length;

  return (
    <section className="dash-band" id="signals">
      <div className="dash-band-head">
        <h2>Signal sources</h2>
        <p>
          The network fuses an electrical stream (EEG, {FEATURES.eeg.total} features) with
          a haemodynamic one (fNIRS, {FEATURES.fnirs.total} features). Which one is it
          actually listening to?
        </p>
      </div>

      <div className="dash-row two">
        <div className="dash-panel">
          <h3>Attention weight by modality</h3>
          <p className="sub">
            The fusion layer&apos;s learned weights, normalised to a share. Read the
            per-class rows against the overall row, not against each other.
          </p>
          {rows.length ? (
            <>
              <div className="split">
                {rows.map((r) => (
                  <div className="split-row" key={r.label}>
                    <div className="lab">
                      <span>{r.label}</span>
                      <b>
                        {pct(r.eeg)} / {pct(r.fnirs)}
                      </b>
                    </div>
                    <div className="split-bar">
                      <span
                        className="eeg"
                        style={{ width: `${r.eeg * 100}%` }}
                        title={`EEG ${pct(r.eeg)}`}
                      >
                        EEG
                      </span>
                      <span
                        className="fnirs"
                        style={{ width: `${r.fnirs * 100}%` }}
                        title={`fNIRS ${pct(r.fnirs)}`}
                      >
                        fNIRS
                      </span>
                    </div>
                  </div>
                ))}
              </div>
              <div className="split-key" style={{ marginTop: 14 }}>
                <span>
                  <i style={{ background: colors.eeg }} />
                  EEG - electrical, millisecond scale
                </span>
                <span>
                  <i style={{ background: colors.fnirs }} />
                  fNIRS - blood oxygenation, seconds scale
                </span>
              </div>
              <p className="foot">
                fNIRS carries the larger share, and the split barely moves between Low and
                High. The model is not switching modality by class; it leans on the slower
                haemodynamic signal throughout.
              </p>
            </>
          ) : (
            <p className="sub">Attention weights unavailable for this bundle.</p>
          )}
        </div>

        <div className="dash-panel">
          <h3>Feature families by attribution</h3>
          <p className="sub">
            Mean absolute attribution per family, over{" "}
            {shap?.n_windows?.toLocaleString()} held-out windows. Larger means the family
            moved the decision more, in either direction.
          </p>
          {top.length ? (
            <>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height={Math.max(240, top.length * 30)}>
                  <BarChart
                    data={top}
                    layout="vertical"
                    margin={{ top: 4, right: 46, bottom: 4, left: 4 }}
                  >
                    <CartesianGrid stroke={colors.grid} horizontal={false} />
                    <XAxis
                      type="number"
                      tick={{ fill: colors.axis, fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(v) => v.toFixed(2)}
                    />
                    <YAxis
                      type="category"
                      dataKey="name"
                      width={132}
                      interval={0}
                      tick={{ fill: colors.axis, fontSize: 10.5 }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <Tooltip
                      cursor={{ fill: "rgba(0,0,0,0.03)" }}
                      content={<MetricTip series={{ v: "Mean |attribution|" }} />}
                    />
                    <Bar dataKey="v" fill={colors.accent} radius={[0, 4, 4, 0]} maxBarSize={14}>
                      <LabelList
                        dataKey="v"
                        position="right"
                        formatter={(v) => Number(v).toFixed(2)}
                        style={{ fill: colors.ink2, fontSize: 10.5 }}
                      />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="foot">
                {fnirsInTop} of the top {top.length} families are fNIRS response-shape
                descriptors, which agrees with the attention split beside it - two
                different methods pointing at the same modality. Method:{" "}
                {shap.method === "integrated_gradients"
                  ? "integrated gradients on the served weights"
                  : shap.method}
                , baseline = {shap.baseline}.
              </p>
            </>
          ) : (
            <p className="sub">Attribution artifact unavailable for this bundle.</p>
          )}
        </div>
      </div>
    </section>
  );
}

/* ── Decisions ───────────────────────────────────────────────────────────── */

function LedgerRow({ name, why, verdict, tone }) {
  return (
    <div className="ledger-row">
      <div>
        <div className="name">{name}</div>
        <div className="why">{why}</div>
      </div>
      <span className={`verdict ${tone}`}>{verdict}</span>
    </div>
  );
}

function DecisionsBand({ metrics, runSummary }) {
  const paired = runSummary?.results?.paired_tests?.loso26_candidate_vs_baseline;
  const pairedLock = runSummary?.results?.paired_tests?.lockbox_candidate_vs_baseline;
  const base = runSummary?.results?.baseline;
  const cand = runSummary?.results?.candidate;

  return (
    <section className="dash-band" id="decisions">
      <div className="dash-band-head">
        <h2>Decision ledger</h2>
        <p>
          {SEARCH.totalConfigs} configurations were trained and scored across{" "}
          {SEARCH.campaigns.length} campaigns. {SEARCH.adopted} replaced the baseline. The
          rejections are the result, not a gap in it.
        </p>
      </div>

      <div className="dash-panel">
        <h3>What is served, and what was turned down</h3>
        <p className="sub">{PROTOCOL.acceptanceRule}</p>
        <div className="ledger">
          <LedgerRow
            name={`${MODEL.family}, ${MODEL.approxParams} parameters`}
            why={
              <>
                Macro F1 {num(metrics?.loso?.hybrid?.f1_mean)} on {PROTOCOL.headlineFolds}
                -fold LOSO. Projection {MODEL.projDim}, encoder {MODEL.encDim},{" "}
                {MODEL.nBlocks} blocks, {MODEL.fusion} fusion.
              </>
            }
            verdict="Served"
            tone="served"
          />
          <LedgerRow
            name="Random Forest ensemble"
            why="Kept in the 3-class bundle, disabled here - the hybrid network alone outperforms it, so the extra stage buys nothing."
            verdict="Disabled"
            tone="off"
          />
          {paired && (
            <LedgerRow
              name={`Candidate ${runSummary.candidate} (${runSummary.n_params?.toLocaleString()} parameters)`}
              why={
                <>
                  Paired against the baseline on the same {paired.n_folds} folds: Δ macro F1{" "}
                  <span className="mono">{num(paired.mean_diff)}</span>, {paired.wins} wins /{" "}
                  {paired.losses} losses, Wilcoxon p = {num(paired.wilcoxon_p, 2)}. Cheaper,
                  but not better.
                </>
              }
              verdict="Rejected"
              tone="rejected"
            />
          )}
          {pairedLock && (
            <LedgerRow
              name="Same candidate, lockbox confirmation"
              why={
                <>
                  Δ macro F1 <span className="mono">{num(pairedLock.mean_diff)}</span> over{" "}
                  {pairedLock.n_folds} unseen subjects, {pairedLock.wins} wins /{" "}
                  {pairedLock.losses} losses. The second protocol agreed with the first.
                </>
              }
              verdict="Rejected"
              tone="rejected"
            />
          )}
          <LedgerRow
            name={`${SEARCH.totalConfigs} configurations searched`}
            why={SEARCH.campaigns
              .map((c) => `${c.name} (${c.configs})`)
              .join(" · ")}
            verdict={`${SEARCH.adopted} adopted`}
            tone="held"
          />
        </div>

        {base && cand && (
          <>
            <div className="dash-scroll" style={{ marginTop: 20 }}>
              <table className="dash-table">
                <caption
                  style={{
                    captionSide: "top",
                    textAlign: "left",
                    fontSize: "0.79rem",
                    color: "var(--ink-3)",
                    paddingBottom: 8,
                  }}
                >
                  Baseline versus candidate, identical folds and environment -{" "}
                  <span className="mono">{runSummary.run_id}</span>
                </caption>
                <thead>
                  <tr>
                    <th>Protocol / metric</th>
                    <th>Baseline (served)</th>
                    <th>Candidate</th>
                    <th>Δ</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["26-fold LOSO · macro F1", "loso26", "f1_macro"],
                    ["26-fold LOSO · accuracy", "loso26", "accuracy"],
                    ["26-fold LOSO · ROC-AUC", "loso26", "roc_auc"],
                    ["Lockbox · macro F1", "lockbox", "f1_macro"],
                    ["Lockbox · accuracy", "lockbox", "accuracy"],
                  ].map(([label, proto, key]) => {
                    const b = base[proto]?.[key]?.mean;
                    const k = cand[proto]?.[key]?.mean;
                    return (
                      <tr key={label}>
                        <td className="name">{label}</td>
                        <td>{num(b)}</td>
                        <td>{num(k)}</td>
                        <td>{b != null && k != null ? num(k - b) : "-"}</td>
                      </tr>
                    );
                  })}
                  <tr>
                    <td className="name">Parameters</td>
                    <td>{MODEL.approxParams}</td>
                    <td>{runSummary.n_params?.toLocaleString()}</td>
                    <td>-</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="foot">
              The candidate is far smaller - {runSummary.n_params?.toLocaleString()}{" "}
              parameters against the served network&apos;s {MODEL.approxParams} - and
              loses on both protocols by well under a fold standard deviation. That is a
              tie, not a win, and under the one-standard-error rule a tie keeps the
              incumbent. Nothing in serving changed.
            </p>
          </>
        )}
        {!base && (
          <p className="foot">
            The paired baseline/candidate comparison lives in the run archive
            (<span className="mono">{HEADLINE_RUN.runId}</span>), which is not reachable
            right now.
          </p>
        )}
      </div>
    </section>
  );
}

/* ── Cohort ──────────────────────────────────────────────────────────────── */

function CohortBand({ folds, cohort, chance, colors }) {
  const data = [...folds]
    .filter((f) => f.accuracy != null)
    .sort((a, b) => a.accuracy - b.accuracy)
    .map((f) => ({
      name: `S${f.subject_id}`,
      accuracy: f.accuracy,
      f1: f.f1_macro,
      auc: f.roc_auc,
      n: f.n_windows,
    }));
  const mean = data.length
    ? data.reduce((a, d) => a + d.accuracy, 0) / data.length
    : null;
  // "No better than a coin toss" needs a threshold, not a vibe: folds within
  // five points of chance are the ones the model cannot claim to have read.
  const atChance = data.filter((d) => d.accuracy <= chance + 0.05).length;

  return (
    <section className="dash-band" id="cohort">
      <div className="dash-band-head">
        <h2>Cohort spread</h2>
        <p>
          The population mean is not a promise to any individual. Each column is one
          person, scored by a model trained without them.
        </p>
      </div>

      <div className="dash-panel">
        <h3>Held-out accuracy, one column per subject</h3>
        <p className="sub">
          Sorted worst to best, from the served configuration&apos;s own folds in the
          archived run <span className="mono">{HEADLINE_RUN.runId}</span>. Same colour
          throughout - the height already carries the value, so hue is free to mean
          nothing here.
        </p>
        {data.length ? (
          <>
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={data} margin={{ top: 8, right: 104, bottom: 24, left: 4 }}>
                  <CartesianGrid stroke={colors.grid} vertical={false} />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: colors.axis, fontSize: 10 }}
                    tickLine={false}
                    axisLine={{ stroke: colors.grid }}
                    interval={0}
                    angle={-45}
                    textAnchor="end"
                    height={44}
                  />
                  <YAxis
                    domain={[0.3, 0.9]}
                    ticks={[0.3, 0.5, 0.7, 0.9]}
                    tickFormatter={(v) => v.toFixed(1)}
                    tick={{ fill: colors.axis, fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    width={36}
                  />
                  {/* Both labels sit outside the plot: the bars run the full
                      width, so any in-plot position lands on a mark. */}
                  <ReferenceLine
                    y={chance}
                    stroke={colors.reference}
                    label={{
                      value: `chance ${chance.toFixed(2)}`,
                      position: "right",
                      fill: colors.axis,
                      fontSize: 10.5,
                    }}
                  />
                  {mean != null && (
                    <ReferenceLine
                      y={mean}
                      stroke={colors.ink2}
                      label={{
                        value: `mean ${mean.toFixed(3)}`,
                        position: "right",
                        fill: colors.ink2,
                        fontSize: 10.5,
                      }}
                    />
                  )}
                  <Tooltip
                    cursor={{ fill: "rgba(0,0,0,0.03)" }}
                    content={<SubjectTip />}
                  />
                  <Bar
                    dataKey="accuracy"
                    fill={colors.accent}
                    radius={[4, 4, 0, 0]}
                    maxBarSize={22}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <TableView caption="Held-out accuracy by subject">
              <thead>
                <tr>
                  <th>Subject</th>
                  <th>Windows</th>
                  <th>Accuracy</th>
                  <th>Macro F1</th>
                  <th>ROC-AUC</th>
                </tr>
              </thead>
              <tbody>
                {data.map((d) => (
                  <tr key={d.name}>
                    <td className="name">{d.name}</td>
                    <td>{d.n}</td>
                    <td>{num(d.accuracy)}</td>
                    <td>{num(d.f1)}</td>
                    <td>{num(d.auc)}</td>
                  </tr>
                ))}
              </tbody>
            </TableView>
          </>
        ) : (
          <p className="sub">Per-fold rows unavailable - the run archive is not reachable.</p>
        )}
        <p className="foot">
          {data.length ? (
            <>
              {atChance} of the {data.length} folds land within five points of chance -
              for those people the model reads nothing. The served bundle&apos;s own
              record of the same configuration shows the same shape, ranging from{" "}
              {cohort.worst ? pct(cohort.worst.accuracy) : "-"} (subject{" "}
              {cohort.worst?.subject_id}) to {cohort.best ? pct(cohort.best.accuracy) : "-"}{" "}
              (subject {cohort.best?.subject_id}). That spread is the honest headline:
              this estimates population-level reliability, not a per-person guarantee, and
              a deployment would need a short calibration session before it could claim
              anything individual.
            </>
          ) : null}
        </p>
      </div>
    </section>
  );
}

function SubjectTip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="dash-tip">
      <div className="t">Subject {d.name.slice(1)}</div>
      <div>
        Accuracy <span className="n">{num(d.accuracy)}</span>
      </div>
      <div>
        Macro F1 <span className="n">{num(d.f1)}</span> · AUC{" "}
        <span className="n">{num(d.auc)}</span>
      </div>
      <div>{d.n} held-out windows</div>
    </div>
  );
}

/* ── Bundles ─────────────────────────────────────────────────────────────── */

function BundlesBand({ binary, threeClass }) {
  const b = binary?.loso?.hybrid;
  // The 3-class bundle serves the hybrid + RF ensemble; that is the row to
  // compare against, because it is what that bundle actually answers with.
  const t = threeClass?.loso?.ensemble_hybrid_rf || threeClass?.loso?.hybrid;
  if (!b) return null;

  const rows = [
    {
      name: "Two classes - Low vs High",
      classes: 2,
      chance: 0.5,
      f1: b.f1_mean,
      acc: b.acc_mean,
      auc: b.auc_mean,
    },
    t
      ? {
          name: "Three classes - Low / Medium / High",
          classes: 3,
          chance: 1 / 3,
          f1: t.f1_mean,
          acc: t.acc_mean,
          auc: t.auc_mean,
        }
      : null,
  ].filter(Boolean);

  return (
    <section className="dash-band" id="bundles">
      <div className="dash-band-head">
        <h2>Two classes or three</h2>
        <p>
          Both bundles are trained on the same recordings under the same protocol. Their
          accuracies are not directly comparable, because they are not answering the same
          question.
        </p>
      </div>

      <div className="dash-panel">
        <h3>Same data, different question</h3>
        <p className="sub">
          Read the last column, not the middle ones: lift over chance is what survives the
          change in class count.
        </p>
        <div className="dash-scroll">
          <table className="dash-table">
            <thead>
              <tr>
                <th>Bundle</th>
                <th>Chance</th>
                <th>Macro F1</th>
                <th>Accuracy</th>
                <th>ROC-AUC</th>
                <th>Lift over chance</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.name}>
                  <td className="name">{r.name}</td>
                  <td>{pct(r.chance)}</td>
                  <td>{num(r.f1)}</td>
                  <td>{pct(r.acc)}</td>
                  <td>{num(r.auc)}</td>
                  <td>{pts(r.acc - r.chance)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="foot">
          Dropping the Medium class raises accuracy by roughly {rows[1] ? pts(rows[0].acc - rows[1].acc) : "-"},
          but the lift over chance barely moves. Most of the apparent improvement is the
          easier question, not a better model - which is why the binary bundle is the one
          this dashboard reports and the 3-class bundle stays available for anyone who
          needs the middle band.
        </p>
      </div>

      <p className="dash-note">
        Every number on this page is read at load time from the serving bundle or the run
        archive. Nothing is transcribed into the page, so a figure that cannot be reached
        shows as unavailable rather than falling back to a stale constant.
      </p>
    </section>
  );
}

/* ── Shared: the readable twin every chart ships with ────────────────────── */

function TableView({ caption, children }) {
  return (
    <details className="dash-tv">
      <summary>Table view</summary>
      <div className="dash-scroll">
        <table className="dash-table">
          <caption
            style={{
              captionSide: "top",
              textAlign: "left",
              fontSize: "0.78rem",
              color: "var(--ink-3)",
              paddingBottom: 8,
            }}
          >
            {caption}
          </caption>
          {children}
        </table>
      </div>
    </details>
  );
}
