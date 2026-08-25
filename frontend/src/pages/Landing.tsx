import { Link } from "react-router-dom";
import { motion, useReducedMotion } from "motion/react";
import SignalField from "../components/landing/SignalField";
import Reveal, { RevealGroup, RevealItem } from "../components/ui/Reveal";
import StatCard from "../components/ui/StatCard";
import { getMetrics } from "../api/client";
import { useApi } from "../useApi";
import { DATASET, FEATURES, SEARCH } from "../lib/projectFacts";

const pct = (x: number | null | undefined) =>
  x == null ? "-" : `${(x * 100).toFixed(1)}%`;

type LosoBlock = { acc_mean?: number; f1_mean?: number; auc_mean?: number };
type MetricsResponse = {
  loso?: Record<string, LosoBlock>;
  lockbox?: Record<string, LosoBlock>;
};

/**
 * First paint. A full-viewport opening that states what the project is, then
 * — on scroll — the abstract and the headline numbers.
 *
 * The hero deliberately does not lead with an accuracy figure. The project's
 * central claim is methodological (how the model is evaluated, not how high it
 * scores), and putting 63% above the fold without the protocol beside it would
 * misrepresent the work in exactly the way the research argues against.
 */
export default function Landing() {
  const reduced = useReducedMotion();
  const { data } = useApi<MetricsResponse>(() => getMetrics("binary"), []);

  const loso = data?.loso?.ensemble_hybrid_rf || data?.loso?.hybrid;
  const lockbox = data?.lockbox?.hybrid;

  return (
    <>
      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <section
        className="relative flex min-h-[calc(100svh-63px)] items-center overflow-hidden border-b border-line bg-paper-edge"
        style={{
          backgroundImage:
            "radial-gradient(120% 90% at 82% -10%, color-mix(in srgb, var(--fnirs) 10%, transparent), transparent 62%), radial-gradient(90% 80% at 8% 110%, color-mix(in srgb, var(--eeg) 11%, transparent), transparent 58%)",
        }}
      >
        <SignalField />

        {/* Scrim: lifts the paper back toward opaque directly under the type,
            so the headline keeps its contrast wherever the traces drift. */}
        <div
          aria-hidden="true"
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(ellipse 70% 65% at 32% 50%, color-mix(in srgb, var(--paper-edge) 88%, transparent) 0%, transparent 70%)",
          }}
        />

        <div className="relative mx-auto w-full max-w-(--container-wrap) px-6 py-24">
          <motion.p
            className="eyebrow-mono text-accent"
            initial={reduced ? false : { opacity: 0, y: 12 }}
            animate={reduced ? undefined : { opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.2, 0.65, 0.2, 1] }}
          >
            Electroencephalography &nbsp;+&nbsp; Functional near-infrared
            spectroscopy
          </motion.p>

          <motion.h1
            className="mt-6 max-w-[19ch] font-display text-4xl font-bold tracking-tight text-ink text-balance"
            initial={reduced ? false : { opacity: 0, y: 18 }}
            animate={reduced ? undefined : { opacity: 1, y: 0 }}
            transition={{
              duration: 0.7,
              delay: 0.08,
              ease: [0.2, 0.65, 0.2, 1],
            }}
          >
            Reading mental workload from the brain
          </motion.h1>

          <motion.p
            className="mt-7 max-w-[58ch] text-lg text-ink-2 text-pretty"
            initial={reduced ? false : { opacity: 0, y: 18 }}
            animate={reduced ? undefined : { opacity: 1, y: 0 }}
            transition={{
              duration: 0.7,
              delay: 0.16,
              ease: [0.2, 0.65, 0.2, 1],
            }}
          >
            A hybrid deep-learning model fuses electrical and hemodynamic brain
            signals to estimate cognitive load, and reports what it can
            actually do on people it has never seen.
          </motion.p>

          <motion.div
            className="mt-10 flex flex-wrap items-center gap-4"
            initial={reduced ? false : { opacity: 0, y: 18 }}
            animate={reduced ? undefined : { opacity: 1, y: 0 }}
            transition={{
              duration: 0.7,
              delay: 0.24,
              ease: [0.2, 0.65, 0.2, 1],
            }}
          >
            <Link
              to="/workflow"
              className="group inline-flex items-center gap-2.5 rounded-sm bg-accent px-6 py-3.5 text-sm font-semibold text-white transition-colors duration-200 hover:bg-accent-ink focus-visible:outline-2 focus-visible:outline-offset-3 focus-visible:outline-accent"
            >
              Explore the research
              <span
                aria-hidden="true"
                className="transition-transform duration-200 ease-signal group-hover:translate-x-1"
              >
                →
              </span>
            </Link>
            <Link
              to="/live"
              className="inline-flex items-center gap-2 rounded-sm border border-line-strong bg-surface px-6 py-3.5 text-sm font-semibold text-ink transition-colors duration-200 hover:border-accent hover:text-accent-ink focus-visible:outline-2 focus-visible:outline-offset-3 focus-visible:outline-accent"
            >
              Try the live demo
            </Link>
          </motion.div>

          {/* Modality legend, establishes the colour coding used site-wide. */}
          <motion.dl
            className="mt-14 flex flex-wrap gap-x-10 gap-y-4"
            initial={reduced ? false : { opacity: 0 }}
            animate={reduced ? undefined : { opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            {[
              {
                c: "var(--eeg)",
                t: "EEG",
                d: "Electrical · fast, noisy",
              },
              {
                c: "var(--fnirs)",
                t: "fNIRS",
                d: "Hemodynamic · slow, stable",
              },
            ].map((m) => (
              <div key={m.t} className="flex items-center gap-3">
                <span
                  aria-hidden="true"
                  className="h-2 w-8 rounded-full"
                  style={{ background: m.c }}
                />
                <div>
                  <dt className="font-mono text-sm font-medium text-ink">
                    {m.t}
                  </dt>
                  <dd className="text-xs text-ink-3">{m.d}</dd>
                </div>
              </div>
            ))}
          </motion.dl>
        </div>

        {/* Scroll cue */}
        <motion.div
          className="absolute inset-x-0 bottom-8 flex justify-center"
          initial={reduced ? false : { opacity: 0 }}
          animate={reduced ? undefined : { opacity: 1 }}
          transition={{ duration: 0.6, delay: 1 }}
        >
          <a
            href="#abstract"
            className="group flex flex-col items-center gap-2 text-ink-3 transition-colors hover:text-accent"
          >
            <span className="eyebrow-mono text-2xs">Scroll</span>
            <motion.span
              aria-hidden="true"
              className="text-lg leading-none"
              animate={reduced ? undefined : { y: [0, 6, 0] }}
              transition={{
                duration: 1.8,
                repeat: Infinity,
                ease: [0.4, 0, 0.6, 1],
              }}
            >
              ↓
            </motion.span>
            <span className="sr-only">Skip to the project abstract</span>
          </a>
        </motion.div>
      </section>

      {/* ── Abstract ─────────────────────────────────────────────────────── */}
      <section
        id="abstract"
        className="scroll-mt-24 border-b border-line bg-paper py-24"
      >
        <div className="mx-auto max-w-(--container-wrap) px-6">
          <div className="grid gap-x-16 gap-y-10 lg:grid-cols-12">
            <Reveal className="lg:col-span-4">
              <p className="eyebrow-mono text-accent">Abstract</p>
              <h2 className="mt-4 font-display text-2xl font-semibold tracking-tight text-ink text-balance">
                The result that matters is methodological
              </h2>
            </Reveal>

            <div className="space-y-6 text-lg text-ink-2 text-pretty lg:col-span-8">
              <Reveal delay={0.05}>
                <p>
                  Mental workload has a measurable neural signature. EEG follows
                  the brain's electrical activity at millisecond resolution but
                  is noisy; fNIRS tracks the slower hemodynamic response and is
                  comparatively stable. The bet behind this project is that a
                  model given both can lean on whichever is more informative at
                  a given moment.
                </p>
              </Reveal>

              <Reveal delay={0.1}>
                <p>
                  Two parallel encoders compress {FEATURES.eeg.total} EEG
                  band-power features and {FEATURES.fnirs.total} fNIRS
                  response-shape descriptors, an attention-gated fusion layer
                  weighs the two streams, and a small classifier separates{" "}
                  <span className="font-medium text-ink">low</span> from{" "}
                  <span className="font-medium text-ink">high</span> cognitive
                  load, operationalised as a 0-back versus 3-back task.
                </p>
              </Reveal>

              <Reveal delay={0.15}>
                <p>
                  The central finding is not the score. It is that a
                  conventional train/test split{" "}
                  <span className="font-medium text-ink">
                    leaks subject identity
                  </span>
                 , different trials from the same person land on both sides of
                  the split, the model learns the person rather than the
                  workload, and the reported number flatters itself. Evaluated
                  the honest way, leaving out every subject in turn, performance
                  drops substantially. This site reports the lower number, and
                  shows the machinery built to keep it honest.
                </p>
              </Reveal>

              <Reveal delay={0.2}>
                <p className="border-l-2 border-accent pl-5 text-base text-ink-2">
                  Across {SEARCH.totalConfigs} controlled configurations in
                  three campaigns, <strong className="text-ink">none</strong>{" "}
                  cleared the acceptance bar against the baseline. That negative
                  result is reported here in the same detail a positive one
                  would be.
                </p>
              </Reveal>
            </div>
          </div>

          {/* Stat strip */}
          <RevealGroup className="mt-20 grid gap-px overflow-hidden rounded-md border border-line bg-line sm:grid-cols-2 lg:grid-cols-4">
            <RevealItem className="bg-surface">
              <StatCard
                value={DATASET.nSubjects}
                label="Subjects"
                sub="TU Berlin open n-back cohort"
              />
            </RevealItem>
            <RevealItem className="bg-surface">
              <StatCard
                value="2"
                label="Modalities fused"
                sub={`${FEATURES.totalPerWindow.toLocaleString()} features per window`}
              />
            </RevealItem>
            <RevealItem className="bg-surface">
              <StatCard
                value={loso ? pct(loso.acc_mean) : "-"}
                label="Accuracy · 26-fold LOSO"
                sub="Every subject held out in turn"
                accent
              />
            </RevealItem>
            <RevealItem className="bg-surface">
              <StatCard
                value={lockbox ? pct(lockbox.acc_mean) : "-"}
                label="Accuracy · lockbox"
                sub="8 subjects, evaluated once"
              />
            </RevealItem>
          </RevealGroup>

          <Reveal delay={0.1}>
            <p className="mt-5 font-mono text-xs text-ink-3">
              Performance figures are read live from the served model bundle.
              Chance is 50%. Where a value shows “-”, the API is unreachable,
              nothing here is a hard-coded placeholder.
            </p>
          </Reveal>

          <Reveal delay={0.15}>
            <div className="mt-16 flex flex-wrap items-center justify-between gap-6 border-t border-line pt-10">
              <p className="max-w-[48ch] text-base text-ink-2">
                The full pipeline, from raw recordings through the leakage
                audit to the decision gates, is laid out stage by stage.
              </p>
              <Link
                to="/workflow"
                className="group inline-flex items-center gap-2.5 rounded-sm bg-ink px-6 py-3.5 text-sm font-semibold text-paper transition-colors duration-200 hover:bg-accent focus-visible:outline-2 focus-visible:outline-offset-3 focus-visible:outline-accent"
              >
                Walk the pipeline
                <span
                  aria-hidden="true"
                  className="transition-transform duration-200 ease-signal group-hover:translate-x-1"
                >
                  →
                </span>
              </Link>
            </div>
          </Reveal>
        </div>
      </section>
    </>
  );
}
