import type { ReactNode } from "react";

type Tone = "finding" | "caution" | "negative";

const TONES: Record<Tone, { stamp: string; label: string }> = {
  // A result the project stands behind.
  finding: { stamp: "text-accent", label: "Finding" },
  // A caveat the reader must carry forward.
  caution: { stamp: "text-warn", label: "Caveat" },
  // Something tried that did not work, reported as such.
  negative: { stamp: "text-stop", label: "Not adopted" },
};

/**
 * A pulled-out result. Used where a conclusion would otherwise be buried in a
 * paragraph, including the negative ones, which get the same visual weight as
 * the positive ones on purpose.
 *
 * The heading leads the card; the tone word sits under it as quiet tracked
 * text. Colour is carried by that word alone rather than by a bar down the
 * side, and the whole card lifts on hover like every other card on the site.
 */
export default function Callout({
  tone = "finding",
  stamp,
  title,
  children,
}: {
  tone?: Tone;
  /** Overrides the tone's default stamp text. */
  stamp?: string;
  title?: string;
  children: ReactNode;
}) {
  const t = TONES[tone];

  return (
    <aside className="rounded-md border border-line bg-surface p-6 shadow-sm transition-[transform,box-shadow,border-color] duration-200 ease-signal hover:-translate-y-[3px] hover:border-line-strong hover:shadow-lg motion-reduce:hover:translate-y-0">
      {title && (
        <h4 className="font-card text-xl leading-snug font-semibold text-ink text-balance">
          {title}
        </h4>
      )}
      <p
        className={`mt-1.5 text-[0.68rem] font-semibold tracking-[0.13em] uppercase ${t.stamp}`}
      >
        {stamp ?? t.label}
      </p>
      <div className="mt-3 max-w-[62ch] text-base text-ink-2 text-pretty">
        {children}
      </div>
    </aside>
  );
}
