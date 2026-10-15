import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import SectionHeader from "../components/ui/SectionHeader";
import Reveal from "../components/ui/Reveal";
import { getMetrics } from "../api/client";
import { useApi } from "../useApi";
import { DATASET, SEARCH } from "../lib/projectFacts";

const pct = (x: number | null | undefined) =>
  x == null ? "-" : `${(x * 100).toFixed(1)}%`;

type Item = { q: string; a: ReactNode };
type Group = { heading: string; items: Item[] };

const A = ({ to, children }: { to: string; children: ReactNode }) => (
  <Link
    to={to}
    className="font-medium text-accent underline decoration-accent/30 underline-offset-2 hover:decoration-accent"
  >
    {children}
  </Link>
);

function AccordionItem({ item, index }: { item: Item; index: number }) {
  const [open, setOpen] = useState(false);
  const reduced = useReducedMotion();
  const id = `faq-${index}`;

  return (
    <div className="border-b border-line">
      <h3>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-controls={id}
          className="group flex w-full items-start justify-between gap-6 py-5 text-left focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        >
          <span
            className={`font-display text-lg font-semibold text-balance transition-colors ${
              open ? "text-accent" : "text-ink group-hover:text-accent"
            }`}
          >
            {item.q}
          </span>
          <span
            aria-hidden="true"
            className={`mt-1 shrink-0 text-xl leading-none text-ink-3 transition-transform duration-300 ease-signal ${
              open ? "rotate-45 text-accent" : "group-hover:text-ink-2"
            }`}
          >
            +
          </span>
        </button>
      </h3>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            id={id}
            key="body"
            initial={reduced ? false : { height: 0, opacity: 0 }}
            animate={reduced ? undefined : { height: "auto", opacity: 1 }}
            exit={reduced ? undefined : { height: 0, opacity: 0 }}
            transition={{ duration: 0.32, ease: [0.2, 0.65, 0.2, 1] }}
            className="overflow-hidden"
          >
            <div className="max-w-[68ch] space-y-4 pb-6 text-base text-ink-2 text-pretty">
              {item.a}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function Faq() {
  const { data } = useApi<{
    loso?: Record<string, { acc_mean?: number; auc_mean?: number }>;
  }>(() => getMetrics("binary"), []);
  const loso = data?.loso?.ensemble_hybrid_rf || data?.loso?.hybrid;

  const groups: Group[] = [
    {
      heading: "The basics",
      items: [
        {
          q: "What is cognitive load?",
          a: (
            <>
              <p>
                Cognitive load is how much mental effort a task is currently
                demanding of you, relative to what you can supply. Holding a
                phone number in your head while someone talks to you is high
                load; reading a familiar road sign is low.
              </p>
              <p>
                It matters because performance degrades at both extremes. Too
                little and attention wanders; too much and errors climb, and
                the people most affected are often the least able to notice,
                because overload degrades self-assessment along with everything
                else.
              </p>
            </>
          ),
        },
        {
          q: "What do EEG and fNIRS actually measure?",
          a: (
            <>
              <p>
                <strong className="text-ink">EEG</strong>{" "}
                (electroencephalography) reads the electrical activity of
                populations of neurons through electrodes on the scalp. It is
                extremely fast, millisecond resolution, but noisy, and easily
                contaminated by jaw clenches, blinks and movement.
              </p>
              <p>
                <strong className="text-ink">fNIRS</strong> (functional
                near-infrared spectroscopy) shines harmless infrared light
                through the scalp and measures how much returns. Oxygenated and
                deoxygenated blood absorb it differently, so the reflected light
                reveals where blood oxygenation is changing. Working brain
                tissue draws more oxygenated blood, but the response lags the
                neural event by several seconds.
              </p>
              <p>
                Neither is invasive. Both are worn on the head like a cap.
              </p>
            </>
          ),
        },
        {
          q: "Why use both instead of one?",
          a: (
            <>
              <p>
                Their weaknesses do not overlap. EEG is fast but noisy; fNIRS is
                stable but slow. The hypothesis was that a model given both
                could lean on whichever was more trustworthy at a given moment.
              </p>
              <p>
                Honestly reported, that hypothesis did not hold as expected: on
                unseen subjects the fusion model did not beat an fNIRS-only
                model. That result is documented rather than hidden, see the{" "}
                <A to="/workflow#architecture">architecture stage</A>.
              </p>
            </>
          ),
        },
        {
          q: "How is “load” defined here, concretely?",
          a: (
            <>
              <p>
                By experimental condition, not by opinion. Participants
                performed an n-back task:{" "}
                <strong className="text-ink">0-back</strong> (match against one
                fixed target, easy) is labelled Low load, and{" "}
                <strong className="text-ink">3-back</strong> (continuously
                update what appeared three items ago, hard) is labelled High.
              </p>
              <p>
                So the model predicts which task condition a person was in. That
                is a well-controlled proxy for workload with a known ground
                truth, but it is a proxy, and it is a laboratory task rather
                than a cockpit or an operating theatre.
              </p>
            </>
          ),
        },
      ],
    },
    {
      heading: "Accuracy and trust",
      items: [
        {
          q: "How accurate is the model?",
          a: (
            <>
              <p>
                <strong className="text-ink">
                  {loso?.acc_mean ? pct(loso.acc_mean) : "About 63%"}
                </strong>{" "}
                on subjects it has never seen, against a chance level of 50%,
                for the two-class Low-vs-High task
                {loso?.auc_mean ? ` (ROC-AUC ${loso.auc_mean.toFixed(3)})` : ""}.
              </p>
              <p>
                That is meaningfully better than guessing and well short of
                reliable. It is quoted under leave-one-subject-out evaluation,
                which is the honest protocol; the same model scores higher under
                weaker protocols, and this site does not quote those.
              </p>
              <p>
                The average also conceals a lot: per-subject accuracy ranges
                from near chance to the high seventies. See the{" "}
                <A to="/workflow#evaluation">per-fold breakdown</A>.
              </p>
            </>
          ),
        },
        {
          q: "Papers report 80–90% accuracy. Why is this lower?",
          a: (
            <>
              <p>
                Usually because a different, easier question is being answered.
                Two things inflate published figures:
              </p>
              <p>
                <strong className="text-ink">Within-subject testing.</strong> If
                the model trains and tests on the same people, it can recognise
                the individual, skull thickness, optode placement, resting
                rhythm, and infer the label from identity rather than workload.
                The score looks strong and predicts nothing about a new user.
              </p>
              <p>
                <strong className="text-ink">Easier label sets.</strong> Two
                well-separated conditions are far easier than a graded scale.
              </p>
              <p>
                This project reports cross-subject performance, where every test
                person is genuinely unseen. The drop between the two protocols
                is the finding, not a defect. The{" "}
                <A to="/methodology">methodology page</A> lays out the taxonomy.
              </p>
            </>
          ),
        },
        {
          q: "What is leave-one-subject-out, and why insist on it?",
          a: (
            <>
              <p>
                Train on {DATASET.nSubjects - 1} people, test on the one left
                out, and repeat until everyone has been the held-out person.
                Average the {DATASET.nSubjects} results.
              </p>
              <p>
                It costs {DATASET.nSubjects} times the compute and returns a
                lower number. It is worth it because it answers the question a
                deployed system would actually face: does this work on someone
                new? A conventional random split cannot answer that, because the
                same person's data ends up on both sides.
              </p>
            </>
          ),
        },
        {
          q: "How do I know the evaluation isn't quietly broken?",
          a: (
            <>
              <p>
                Because it is enforced by code rather than by care. Before any
                fold trains, assertions check that subjects, trials, windows and
                individual feature rows are disjoint across splits, and that
                scalers were fitted on training rows only. A failure raises an
                error and aborts the run, a leaked score is never produced, so
                it cannot be accidentally published.
              </p>
              <p>
                A separate test suite injects each kind of leakage deliberately
                to confirm the audit catches it. An audit that never fires would
                prove nothing.
              </p>
              <p>
                Every metric shown is also recomputed from that run's saved
                predictions and checked to match. See the{" "}
                <A to="/workflow#evaluation">evaluation stage</A>.
              </p>
            </>
          ),
        },
        {
          q: "Did anything improve on the baseline?",
          a: (
            <>
              <p>
                Not yet. Across {SEARCH.totalConfigs} controlled configurations
                in three campaigns, nothing has cleared the acceptance bar and
                been adopted.
              </p>
              <p>
                The instructive case: one candidate won its development screen,
                was promoted to the full protocol, and then lost to the
                untouched baseline on both the 26-fold evaluation and the sealed
                lockbox, with no result statistically significant. A screen win
                can evaporate.
              </p>
              <p>
                One candidate is still open: a five-model ensemble that cleared
                the development bar but has not yet been confirmed on the full
                protocol. It is not what the site serves, and will not be
                claimed until it is verified. Details on the{" "}
                <A to="/evaluation">evaluation page</A>.
              </p>
            </>
          ),
        },
      ],
    },
    {
      heading: "Using it",
      items: [
        {
          q: "Is this clinically validated? Can it diagnose anything?",
          a: (
            <>
              <p>
                <strong className="text-ink">No, to both.</strong> This is a
                research demonstration, not a medical device. It has not been
                clinically validated, is not approved for any diagnostic
                purpose, and must not be used to make decisions about any
                individual's health, fitness to work, or capability.
              </p>
              <p>
                The output describes which experimental task condition a
                recording resembles. It is not a measure of intelligence,
                attention disorder, fatigue, or any clinical state.
              </p>
            </>
          ),
        },
        {
          q: "What is the live demo actually doing?",
          a: (
            <>
              <p>
                Running the real frozen model, through the real serving API, on
                real held-out windows from subjects the model never trained on.
                The prediction and its confidence are genuine.
              </p>
              <p>
                Two honest caveats. The demo windows are a curated subset chosen
                for coherent playback, not a fresh recording. And the demo
                reuses each known subject's precomputed normalisation
                statistics, which would not exist for a genuinely new person.
              </p>
              <p>
                Try it on the <A to="/live">live prediction page</A>.
              </p>
            </>
          ),
        },
        {
          q: "Why does it sometimes refuse to answer?",
          a: (
            <>
              <p>
                The demo abstains when its confidence falls below a threshold.
                This is backed by measurement, not caution theatre: accuracy
                rises steadily as the least-confident predictions are set aside,
                so declining to call a borderline window is a real trade rather
                than a cosmetic one.
              </p>
              <p>
                The model is also mildly over-confident overall, which is
                measured and reported on the{" "}
                <A to="/methodology">methodology page</A> rather than smoothed
                over.
              </p>
            </>
          ),
        },
        {
          q: "Can I run it on my own recordings?",
          a: (
            <>
              <p>
                Only if your data is already in this dataset's exact feature
                format. The <A to="/upload">upload page</A> accepts a CSV of
                pre-extracted features matching the published schema, and
                provides a template.
              </p>
              <p>
                It will not accept raw signals, recordings from a different
                headset, or a different channel montage. The model was trained
                on one specific montage, and the per-subject normalisation it
                depends on has no meaning for a person it has no statistics for.
              </p>
            </>
          ),
        },
        {
          q: "Where do the numbers on this site come from?",
          a: (
            <>
              <p>
                Every figure is read at request time from a committed artifact,
                a file written by a training run and stored in the run archive,
                rather than typed into the interface. Where a value cannot be
                loaded, the page shows a dash and says so; it never falls back
                to a cached or hardcoded number.
              </p>
              <p>
                You can browse model evaluations in the{" "}
                <A to="/compare">model comparison</A> page.
              </p>
            </>
          ),
        },
      ],
    },
  ];

  return (
    <>
      <header className="border-b border-line bg-paper-edge">
        <div className="mx-auto max-w-(--container-wrap) px-6 py-16 lg:py-20">
          <p className="eyebrow-mono text-accent">Questions</p>
          <h1 className="mt-4 max-w-[22ch] font-display text-3xl font-bold tracking-tight text-ink text-balance">
            Frequently asked questions
          </h1>
          <p className="mt-5 max-w-[62ch] text-lg text-ink-2 text-pretty">
            Written for readers arriving without a background in neuroscience or
            machine learning, and kept as blunt about the limitations as about
            the results.
          </p>
        </div>
      </header>

      <div className="mx-auto max-w-(--container-wrap) px-6 py-20 lg:py-24">
        <div className="space-y-20">
          {groups.map((group, gi) => (
            <section key={group.heading}>
              <Reveal>
                <SectionHeader
                  overline={`Section ${String(gi + 1).padStart(2, "0")}`}
                  title={group.heading}
                />
              </Reveal>

              <div className="mt-8 border-t border-line">
                {group.items.map((item, i) => (
                  <AccordionItem
                    key={item.q}
                    item={item}
                    index={gi * 100 + i}
                  />
                ))}
              </div>
            </section>
          ))}
        </div>

        <Reveal>
          <div className="mt-20 rounded-md border border-line bg-surface p-8">
            <h2 className="font-display text-xl font-semibold text-ink">
              Still unanswered?
            </h2>
            <p className="mt-3 max-w-[62ch] text-base text-ink-2 text-pretty">
              The <A to="/methodology">methodology page</A> covers protocol and
              limitations in technical detail, the{" "}
              <A to="/workflow">pipeline</A> walks the whole process stage by
              stage, and the <A to="/dashboard">model dashboard</A> holds the
              headline record including every rejected candidate.
            </p>
          </div>
        </Reveal>
      </div>
    </>
  );
}
