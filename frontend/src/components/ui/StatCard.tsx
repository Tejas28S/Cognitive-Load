import type { ReactNode } from "react";

type StatCardProps = {
  value: ReactNode;
  label: string;
  /** Provenance or qualifier: the protocol a number was measured under. */
  sub?: ReactNode;
  accent?: boolean;
};

/**
 * A single measured quantity. The rule this enforces: a number never appears
 * without the protocol it was measured under, because "63.1%" means something
 * different under LOSO than under a fixed split.
 */
export default function StatCard({
  value,
  label,
  sub,
  accent = false,
}: StatCardProps) {
  return (
    <div className="h-full bg-surface p-5 transition-colors duration-200 hover:bg-surface-2">
      <div
        className={`font-display text-[2.1rem] leading-none font-bold tracking-tight tnum ${
          accent ? "text-accent" : "text-ink"
        }`}
      >
        {value}
      </div>

      <div className="eyebrow-mono mt-3 text-ink-3">{label}</div>

      {sub && <div className="mt-1.5 text-sm text-ink-2">{sub}</div>}
    </div>
  );
}
