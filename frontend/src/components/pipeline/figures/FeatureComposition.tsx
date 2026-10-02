import { motion, useReducedMotion } from "motion/react";
import { FEATURES } from "../../../lib/projectFacts";

/**
 * What one model input actually contains: 1,476 numbers, and where they come
 * from. fNIRS contributes nearly twice what EEG does by count, worth showing,
 * because the explainability work (SHAP, attention, RF importance) independently
 * finds fNIRS carrying most of the signal, and a reader should be able to see
 * that the count alone doesn't explain that result.
 */
export default function FeatureComposition() {
  const reduced = useReducedMotion();
  const total = FEATURES.totalPerWindow;

  const groups = [
    {
      key: "eeg",
      label: "EEG",
      count: FEATURES.eeg.total,
      color: "var(--eeg)",
      breakdown: `${FEATURES.eeg.channels} channels × ${FEATURES.eeg.bands.length} bands × ${FEATURES.eeg.stats.length} statistics`,
      detail: FEATURES.eeg.bands.join(" · "),
      detailLabel: "Bands",
    },
    {
      key: "fnirs",
      label: "fNIRS",
      count: FEATURES.fnirs.total,
      color: "var(--fnirs)",
      breakdown: `${FEATURES.fnirs.channels} channels × ${FEATURES.fnirs.descriptors.length} descriptors × ${FEATURES.fnirs.chromophores.length} chromophores`,
      detail: FEATURES.fnirs.descriptors.join(" · "),
      detailLabel: "Descriptors",
    },
  ];

  return (
    <div className="min-w-[420px]">
      {/* Proportion bar */}
      <div className="flex h-11 w-full overflow-hidden rounded-sm">
        {groups.map((g, i) => (
          <motion.div
            key={g.key}
            className="relative flex items-center justify-center"
            style={{ background: g.color }}
            initial={reduced ? false : { width: 0 }}
            whileInView={
              reduced ? undefined : { width: `${(g.count / total) * 100}%` }
            }
            viewport={{ once: true }}
            transition={{
              duration: 0.7,
              delay: 0.1 + i * 0.12,
              ease: [0.2, 0.65, 0.2, 1],
            }}
          >
            <span className="font-mono text-xs font-medium text-white tnum">
              {g.count}
            </span>
          </motion.div>
        ))}
      </div>

      <div className="mt-2 flex justify-between font-mono text-2xs text-ink-3">
        <span>0</span>
        <span>{total.toLocaleString()} features per 10-second window</span>
      </div>

      {/* Composition detail */}
      <dl className="mt-7 grid gap-6 sm:grid-cols-2">
        {groups.map((g) => (
          <div key={g.key} className="border-t-2 pt-4" style={{ borderColor: g.color }}>
            <dt className="flex items-baseline justify-between gap-3">
              <span className="font-display text-xl font-semibold text-ink">
                {g.label}
              </span>
              <span className="font-mono text-sm text-ink-2 tnum">
                {((g.count / total) * 100).toFixed(0)}%
              </span>
            </dt>
            <dd className="mt-2 font-mono text-2xs text-ink-3">
              {g.breakdown}
            </dd>
            <dd className="mt-3">
              <span className="eyebrow-mono text-2xs text-ink-3">
                {g.detailLabel}
              </span>
              <p className="mt-1 font-mono text-2xs leading-relaxed text-ink-2">
                {g.detail}
              </p>
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
