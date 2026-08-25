import SectionHeader from "../components/ui/SectionHeader";
import Reveal, { RevealGroup, RevealItem } from "../components/ui/Reveal";
import { ADVISORS, HAS_PLACEHOLDERS, TEAM, type TeamMember } from "../lib/team";
import { DATASET } from "../lib/projectFacts";

/** Initials for the avatar plate. Placeholder entries get a neutral glyph. */
function initials(name: string) {
  if (name.startsWith("[")) return "··";
  return name
    .split(/\s+/)
    .slice(0, 2)
    .map((w) => w[0]?.toUpperCase() ?? "")
    .join("");
}

function MemberCard({ member }: { member: TeamMember }) {
  const isPlaceholder = member.placeholder;

  return (
    <article
      className={`h-full rounded-md border bg-surface p-6 shadow-sm transition-[transform,box-shadow,border-color] duration-200 ease-signal hover:-translate-y-[3px] hover:shadow-lg motion-reduce:hover:translate-y-0 ${
        isPlaceholder
          ? "border-dashed border-warn/45 hover:border-warn/70"
          : "border-line hover:border-line-strong"
      }`}
    >
      <div className="flex items-start gap-4">
        <span
          aria-hidden="true"
          className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-sm font-display text-lg font-semibold ${
            isPlaceholder
              ? "bg-warn/10 text-warn"
              : "bg-accent/10 text-accent-ink"
          }`}
        >
          {initials(member.name)}
        </span>

        <div className="min-w-0">
          <h3
            className={`font-card text-lg leading-snug font-semibold text-balance ${
              isPlaceholder ? "text-ink-2" : "text-ink"
            }`}
          >
            {member.name}
          </h3>
          <p className="mt-1 text-sm font-medium text-accent">{member.role}</p>
          <p className="mt-0.5 text-sm text-ink-3">{member.affiliation}</p>
        </div>
      </div>

      <p className="mt-5 text-sm leading-relaxed text-ink-2 text-pretty">
        {member.focus}
      </p>

      {member.interests && (
        <p className="mt-3 text-sm text-ink-3">{member.interests}</p>
      )}

      {isPlaceholder && (
        <p className="mt-5 border-t border-warn/25 pt-3 font-mono text-2xs text-warn">
          Unconfirmed, edit src/lib/team.ts
        </p>
      )}
    </article>
  );
}

export default function Bios() {
  return (
    <>
      <header className="border-b border-line bg-paper-edge">
        <div className="mx-auto max-w-(--container-wrap) px-6 py-16 lg:py-20">
          <p className="eyebrow-mono text-accent">Team</p>
          <h1 className="mt-4 max-w-[22ch] font-display text-3xl font-bold tracking-tight text-ink text-balance">
            Who built this
          </h1>
          <p className="mt-5 max-w-[62ch] text-lg text-ink-2 text-pretty">
            A capstone research project on cognitive load estimation from
            simultaneous EEG and fNIRS recordings.
          </p>
        </div>
      </header>

      {HAS_PLACEHOLDERS && (
        <div className="border-b border-warn/30 bg-warn/[0.07]">
          <div className="mx-auto flex max-w-(--container-wrap) gap-4 px-6 py-5">
            <span aria-hidden="true" className="text-lg leading-none text-warn">
              ⚠
            </span>
            <div>
              <p className="text-sm font-semibold text-ink">
                This page contains unconfirmed placeholder entries.
              </p>
              <p className="mt-1 max-w-[70ch] text-sm text-ink-2">
                Names, roles and affiliations are the one thing on this site
                that cannot be derived from the repository, so none have been
                invented. Fill them in at{" "}
                <code className="font-mono text-xs text-ink">
                  src/lib/team.ts
                </code>{" "}
                and remove each entry's <code className="font-mono text-xs text-ink">placeholder</code>{" "}
                flag; this banner disappears once nothing is left unconfirmed.
              </p>
            </div>
          </div>
        </div>
      )}

      <section className="border-b border-line py-20 lg:py-24">
        <div className="mx-auto max-w-(--container-wrap) px-6">
          <Reveal>
            <SectionHeader overline="Contributors" title="Project team" />
          </Reveal>

          <RevealGroup className="mt-12 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {TEAM.map((m) => (
              <RevealItem key={m.name}>
                <MemberCard member={m} />
              </RevealItem>
            ))}
          </RevealGroup>
        </div>
      </section>

      {ADVISORS.length > 0 && (
        <section className="border-b border-line py-20 lg:py-24">
          <div className="mx-auto max-w-(--container-wrap) px-6">
            <Reveal>
              <SectionHeader
                overline="Supervision"
                title="Advisors"
                lede="Academic supervision and review."
              />
            </Reveal>

            <RevealGroup className="mt-12 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
              {ADVISORS.map((m) => (
                <RevealItem key={m.name}>
                  <MemberCard member={m} />
                </RevealItem>
              ))}
            </RevealGroup>
          </div>
        </section>
      )}

      {/* Acknowledgement, this one is real and belongs on the page. */}
      <section className="py-20 lg:py-24">
        <div className="mx-auto max-w-(--container-wrap) px-6">
          <Reveal>
            <SectionHeader
              overline="Acknowledgement"
              title="Data providers"
              lede={
                <>
                  This project collected no new human data. All results come
                  from the openly published {DATASET.name} dataset (
                  {DATASET.citation}), whose authors made simultaneous EEG–NIRS
                  recordings from {DATASET.nSubjects} participants freely
                  available for research use.
                </>
              }
            />
          </Reveal>
        </div>
      </section>
    </>
  );
}
