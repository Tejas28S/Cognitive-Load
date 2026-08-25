import type { ReactNode } from "react";

type SectionHeaderProps = {
  /** Small tracked-out mono label above the title. */
  overline?: string;
  title: ReactNode;
  /** One or two sentences. Kept to a ~65ch measure for readability. */
  lede?: ReactNode;
  /** Stage number for the pipeline sections — rendered as a large ghosted numeral. */
  index?: number;
  align?: "left" | "center";
  className?: string;
};

export default function SectionHeader({
  overline,
  title,
  lede,
  index,
  align = "left",
  className = "",
}: SectionHeaderProps) {
  const centered = align === "center";

  return (
    <header
      className={`relative ${centered ? "text-center" : ""} ${className}`}
    >
      {index !== undefined && (
        <span
          aria-hidden="true"
          className="pointer-events-none absolute -top-6 -left-2 select-none font-display text-[5.5rem] leading-none font-bold text-ink/[0.045] tnum sm:-top-9 sm:-left-4 sm:text-[8rem]"
        >
          {String(index).padStart(2, "0")}
        </span>
      )}

      <div className="relative">
        {overline && (
          <p className="eyebrow-mono mb-3 text-accent">{overline}</p>
        )}

        <h2 className="font-display text-2xl font-semibold tracking-tight text-ink text-balance sm:text-3xl">
          {title}
        </h2>

        {lede && (
          <p
            className={`mt-4 max-w-[65ch] text-lg text-ink-2 text-pretty ${
              centered ? "mx-auto" : ""
            }`}
          >
            {lede}
          </p>
        )}
      </div>
    </header>
  );
}
