import { Link } from "react-router-dom";
import SectionHeader from "../components/ui/SectionHeader";
import Callout from "../components/ui/Callout";
import Reveal, { RevealGroup, RevealItem } from "../components/ui/Reveal";
import { DATASET, SEARCH } from "../lib/projectFacts";

const PRINCIPLES = [
  {
    n: "01",
    title: "Report the harder number",
    body: "Where two protocols disagree, the site quotes the one that generalises. Cross-subject performance is lower than a fixed split suggests, and it is the figure shown first.",
  },
  {
    n: "02",
    title: "Make leakage impossible, not unlikely",
    body: "Protocol correctness is enforced by assertions that abort a run, not by care and good intentions. A leaked score is never produced, so it can never be accidentally published.",
  },
  {
    n: "03",
    title: "Publish the negative results",
    body: `${SEARCH.totalConfigs} configurations were evaluated and none was adopted. Every one is on the record with its verdict, including the candidate that won its screen and then lost the full protocol.`,
  },
  {
    n: "04",
    title: "Trace every number to an artifact",
    body: "Nothing in this interface is a hardcoded figure. Each value is read from a committed file that can be independently recomputed from that run's saved predictions.",
  },
];

export default function About() {
  return (
    <>
      <header className="border-b border-line bg-paper-edge">
        <div className="mx-auto max-w-(--container-wrap) px-6 py-16 lg:py-20">
          <p className="eyebrow-mono text-accent">About the project</p>
          <h1 className="mt-4 max-w-[24ch] font-display text-3xl font-bold tracking-tight text-ink text-balance">
            Why measure cognitive load, and why measure it this carefully
          </h1>
        </div>
      </header>

      {/* Motivation */}
      <section className="border-b border-line py-20 lg:py-24">
        <div className="mx-auto max-w-(--container-wrap) px-6">
          <div className="grid gap-x-16 gap-y-10 lg:grid-cols-12">
            <Reveal className="lg:col-span-4">
              <SectionHeader overline="Motivation" title="The problem" />
            </Reveal>

            <div className="max-w-[68ch] space-y-5 text-base text-ink-2 text-pretty lg:col-span-8">
              <Reveal delay={0.05}>
                <p>
                  Mental workload is the gap between what a task demands and
                  what a person can currently supply. Too little and attention
                  drifts; too much and errors climb. The people for whom this
                  matters most, air-traffic controllers, surgeons, pilots,
                  drivers, students, are usually the least able to notice it
                  happening, because the same overload that degrades performance
                  also degrades self-assessment.
                </p>
              </Reveal>

              <Reveal delay={0.1}>
                <p>
                  A reliable objective measure would let interfaces adapt before
                  performance breaks down, rather than after. That is the long-run
                  motivation. This project addresses one prerequisite for it:
                  whether the measurement generalises to a person the system has
                  never encountered.
                </p>
              </Reveal>

              <Reveal delay={0.15}>
                <p>
                  Two non-invasive signals are informative and complementary.
                  EEG tracks electrical activity at millisecond resolution but
                  is noisy and easily contaminated by movement. fNIRS tracks
                  blood oxygenation, slower, lagging the neural event by
                  seconds, but comparatively stable. The hypothesis was that a
                  model given both could lean on whichever was more reliable at
                  a given moment.
                </p>
              </Reveal>

              <Reveal delay={0.2}>
                <p>
                  Testing that hypothesis honestly turned out to be the harder
                  problem, and became the project's real subject.
                </p>
              </Reveal>
            </div>
          </div>
        </div>
      </section>

      {/* The methodological turn */}
      <section className="border-b border-line bg-surface py-20 lg:py-24">
        <div className="mx-auto max-w-(--container-wrap) px-6">
          <Reveal>
            <SectionHeader
              overline="The turn"
              title="Where the project changed direction"
              lede="Published accuracies for this task frequently sit between 60% and 90%. Reproducing that range turned out to be easy, and meaningless."
            />
          </Reveal>

          <div className="mt-10 grid gap-x-16 gap-y-8 lg:grid-cols-12">
            <div className="max-w-[68ch] space-y-5 text-base text-ink-2 text-pretty lg:col-span-8">
              <Reveal delay={0.05}>
                <p>
                  Those headline figures are usually measured under one of two
                  easier protocols: a binary task rather than a graded one, or a{" "}
                  <em>within-subject</em> split, where the model is tested on
                  the same people it trained on. Under a within-subject split a
                  model can identify the individual, their skull thickness,
                  optode placement, resting rhythm, and infer the label from
                  that, having learned nothing transferable about workload.
                </p>
              </Reveal>

              <Reveal delay={0.1}>
                <p>
                  Once every subject was properly held out, performance dropped
                  substantially. The gap between those two numbers is the
                  finding this project reports. It is not a flaw in the model;
                  it is the difference between a measurement that would work on
                  a new user and one that would not.
                </p>
              </Reveal>
            </div>

            <Reveal delay={0.15} className="lg:col-span-4">
              <Callout tone="finding" title="What this site is for">
                Showing the honest number, and showing the machinery that keeps
                it honest, so a reader can check the claim rather than take it
                on trust.
              </Callout>
            </Reveal>
          </div>
        </div>
      </section>

      {/* Principles */}
      <section className="border-b border-line py-20 lg:py-24">
        <div className="mx-auto max-w-(--container-wrap) px-6">
          <Reveal>
            <SectionHeader
              overline="Working principles"
              title="Four rules the project holds to"
              lede="Each of these cost something. They are listed because the cost is the point, they are what makes the reported number worth reading."
            />
          </Reveal>

          <RevealGroup className="mt-12 grid gap-x-12 gap-y-10 sm:grid-cols-2">
            {PRINCIPLES.map((p) => (
              <RevealItem key={p.n}>
                <article className="border-t border-line pt-6">
                  <p className="eyebrow-mono text-2xs text-accent tnum">{p.n}</p>
                  <h3 className="mt-3 font-display text-xl font-semibold text-ink text-balance">
                    {p.title}
                  </h3>
                  <p className="mt-3 max-w-[52ch] text-base text-ink-2 text-pretty">
                    {p.body}
                  </p>
                </article>
              </RevealItem>
            ))}
          </RevealGroup>
        </div>
      </section>

      {/* Data provenance + scope */}
      <section className="py-20 lg:py-24">
        <div className="mx-auto max-w-(--container-wrap) px-6">
          <div className="grid gap-x-16 gap-y-10 lg:grid-cols-12">
            <Reveal className="lg:col-span-4">
              <SectionHeader overline="Provenance & scope" title="Data and limits" />
            </Reveal>

            <div className="space-y-6 lg:col-span-8">
              <Reveal delay={0.05}>
                <div className="max-w-[68ch] space-y-5 text-base text-ink-2 text-pretty">
                  <p>
                    All work uses the {DATASET.name} dataset,{" "}
                    {DATASET.citation}, an openly published set of simultaneous
                    EEG–NIRS recordings from {DATASET.nSubjects} participants
                    performing n-back tasks. No new human data was collected for
                    this project, and no participant is identifiable from
                    anything shown here.
                  </p>
                </div>
              </Reveal>

              <Reveal delay={0.1}>
                <Callout tone="caution" stamp="Scope" title="This is not a medical device">
                  Nothing here is diagnostic, clinically validated, or fit for
                  use in a safety-critical setting. It is a research
                  demonstration of a measurement technique, evaluated on a small
                  public dataset. The load estimates describe an experimental
                  task condition, not a clinical state, and should not be used
                  to make decisions about any individual.
                </Callout>
              </Reveal>

              <Reveal delay={0.15}>
                <p className="text-base text-ink-2">
                  The full set of documented limitations, cohort size,
                  demo-mode normalisation, the interpretive band in the trend
                  view, and the upload format's constraints, is kept on the{" "}
                  <Link
                    to="/methodology"
                    className="font-medium text-accent underline decoration-accent/30 underline-offset-2 hover:decoration-accent"
                  >
                    methodology page
                  </Link>
                  , and answered in plainer language in the{" "}
                  <Link
                    to="/faq"
                    className="font-medium text-accent underline decoration-accent/30 underline-offset-2 hover:decoration-accent"
                  >
                    FAQ
                  </Link>
                  .
                </p>
              </Reveal>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
