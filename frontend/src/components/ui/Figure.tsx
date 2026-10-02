import type { ReactNode } from "react";

type FigureProps = {
  /** Plate number, e.g. "3" -> "Fig. 3". */
  num?: string | number;
  title: string;
  /** The committed artifact this figure's numbers come from. */
  source?: string;
  children: ReactNode;
  className?: string;
};

/**
 * A figure "plate": the chart framed, numbered and captioned the way a paper
 * figure is, with the artifact it traces to named in the caption.
 *
 * The source line is not decoration: it is the project's core claim made
 * checkable. Anything shown here can be recomputed from the named file.
 *
 * The body is the horizontal scroll container, so wide charts scroll inside
 * the plate instead of forcing the page to scroll sideways.
 */
export default function Figure({
  num,
  title,
  source,
  children,
  className = "",
}: FigureProps) {
  return (
    <figure
      className={`m-0 overflow-hidden rounded-md border border-line bg-surface shadow-sm transition-[transform,box-shadow,border-color] duration-200 ease-signal hover:-translate-y-[3px] hover:border-line-strong hover:shadow-lg motion-reduce:hover:translate-y-0 ${className}`}
    >
      <div className="overflow-x-auto p-5 sm:p-6">{children}</div>

      <figcaption className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2 border-t border-line bg-surface-2 px-5 py-3 sm:px-6">
        <span className="text-sm text-ink-2">
          {num != null && (
            <span className="mr-2 font-mono text-xs font-medium tracking-wide text-accent uppercase">
              Fig. {num}
            </span>
          )}
          {title}
        </span>
        {source && (
          <span className="font-mono text-2xs text-ink-3">{source}</span>
        )}
      </figcaption>
    </figure>
  );
}
