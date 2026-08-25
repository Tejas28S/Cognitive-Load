import { motion, useReducedMotion } from "motion/react";
import { DATASET, PROTOCOL } from "../../../lib/projectFacts";

/**
 * The cohort, and the wall running through it.
 *
 * 18 subjects drive every experiment and every comparison. 8 are sealed away
 * and evaluated once, after the configuration is frozen. Showing this as
 * people rather than as a ratio is the point: the lockbox is not a random
 * held-out slice of rows, it is a set of individuals the selection process
 * never got to see.
 */
export default function CohortSplit() {
  const reduced = useReducedMotion();
  const dev = new Set<number>(PROTOCOL.developmentSubjects);
  const all = Array.from({ length: DATASET.nSubjects }, (_, i) => i + 1);

  return (
    <div className="min-w-[420px]">
      <div className="grid gap-8 sm:grid-cols-[1fr_auto]">
        <div>
          <p className="eyebrow-mono mb-4 text-ink-3">
            {DATASET.nSubjects} subjects · {DATASET.windowsPerSubject} windows each
          </p>

          <ul className="flex flex-wrap gap-1.5">
            {all.map((id, i) => {
              const isDev = dev.has(id);
              return (
                <motion.li
                  key={id}
                  className={`flex h-9 w-9 items-center justify-center rounded-sm font-mono text-2xs font-medium tnum ${
                    isDev
                      ? "bg-accent/12 text-accent-ink ring-1 ring-accent/30"
                      : "bg-ink/[0.04] text-ink-2 ring-1 ring-dashed ring-line-strong"
                  }`}
                  initial={reduced ? false : { opacity: 0, scale: 0.8 }}
                  whileInView={reduced ? undefined : { opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{
                    duration: 0.3,
                    delay: 0.015 * i,
                    ease: [0.2, 0.65, 0.2, 1],
                  }}
                  title={
                    isDev
                      ? `Subject ${id}, development pool`
                      : `Subject ${id}, lockbox, evaluated once`
                  }
                >
                  {id}
                </motion.li>
              );
            })}
          </ul>

          <dl className="mt-6 flex flex-wrap gap-x-8 gap-y-3 text-sm">
            <div className="flex items-center gap-2.5">
              <span className="h-3 w-3 rounded-xs bg-accent/25 ring-1 ring-accent/40" />
              <dt className="font-medium text-ink">
                Development · {PROTOCOL.devFolds}
              </dt>
              <dd className="text-ink-3">every selection decision</dd>
            </div>
            <div className="flex items-center gap-2.5">
              <span className="h-3 w-3 rounded-xs bg-ink/[0.06] ring-1 ring-line-strong" />
              <dt className="font-medium text-ink">
                Lockbox · {PROTOCOL.lockboxSubjects.length}
              </dt>
              <dd className="text-ink-3">opened once, at the end</dd>
            </div>
          </dl>
        </div>

        {/* Class balance, exactly even by construction of the n-back task. */}
        <div className="sm:w-44 sm:border-l sm:border-line sm:pl-8">
          <p className="eyebrow-mono mb-4 text-ink-3">Class balance</p>
          <ul className="space-y-3">
            {(
              [
                ["Low (0-back)", DATASET.classBalance.Low, "var(--low)"],
                ["High (3-back)", DATASET.classBalance.High, "var(--high)"],
              ] as const
            ).map(([label, n, color]) => (
              <li key={label}>
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-sm text-ink-2">{label}</span>
                  <span className="font-mono text-sm text-ink tnum">
                    {n.toLocaleString()}
                  </span>
                </div>
                <motion.div
                  className="mt-1.5 h-1.5 rounded-full"
                  style={{ background: color }}
                  initial={reduced ? false : { scaleX: 0 }}
                  whileInView={reduced ? undefined : { scaleX: 1 }}
                  viewport={{ once: true }}
                  transition={{
                    duration: 0.6,
                    delay: 0.2,
                    ease: [0.2, 0.65, 0.2, 1],
                  }}
                />
              </li>
            ))}
          </ul>
          <p className="mt-4 font-mono text-2xs leading-relaxed text-ink-3">
            Perfectly balanced by design, chance is exactly 50%, so accuracy is
            directly interpretable.
          </p>
        </div>
      </div>
    </div>
  );
}
