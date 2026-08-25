import { motion, useReducedMotion } from "motion/react";
import { FEATURES, MODEL } from "../../../lib/projectFacts";

/**
 * The served architecture, drawn as signal flow.
 *
 * The connectors animate their pathLength on entry so the diagram assembles in
 * reading order — inputs, encoders, fusion, head. That is the one place motion
 * earns its keep here: it makes the direction of flow legible without arrows
 * cluttering every join.
 */

const W = 780;
const H = 300;

type BoxProps = {
  x: number;
  y: number;
  w: number;
  h: number;
  title: string;
  sub?: string;
  color?: string;
  delay: number;
  reduced: boolean | null;
};

function Box({ x, y, w, h, title, sub, color, delay, reduced }: BoxProps) {
  return (
    <motion.g
      initial={reduced ? false : { opacity: 0, y: 8 }}
      whileInView={reduced ? undefined : { opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4, delay, ease: [0.2, 0.65, 0.2, 1] }}
    >
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={7}
        fill="var(--surface)"
        stroke={color || "var(--border-strong)"}
        strokeWidth={color ? 1.5 : 1}
      />
      <text
        x={x + w / 2}
        y={sub ? y + h / 2 - 4 : y + h / 2 + 4}
        textAnchor="middle"
        className="fill-ink"
        style={{ fontSize: 12.5, fontWeight: 600 }}
      >
        {title}
      </text>
      {sub && (
        <text
          x={x + w / 2}
          y={y + h / 2 + 13}
          textAnchor="middle"
          className="fill-ink-3 font-mono"
          style={{ fontSize: 10 }}
        >
          {sub}
        </text>
      )}
    </motion.g>
  );
}

function Link({
  d,
  delay,
  reduced,
  color = "var(--border-strong)",
}: {
  d: string;
  delay: number;
  reduced: boolean | null;
  color?: string;
}) {
  return (
    <motion.path
      d={d}
      fill="none"
      stroke={color}
      strokeWidth={1.5}
      strokeLinecap="round"
      initial={reduced ? false : { pathLength: 0, opacity: 0 }}
      whileInView={reduced ? undefined : { pathLength: 1, opacity: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay, ease: [0.2, 0.65, 0.2, 1] }}
    />
  );
}

export default function ArchitectureFlow() {
  const reduced = useReducedMotion();

  const eegY = 52;
  const fnirsY = 186;
  const midY = 119;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-auto w-full min-w-[680px]"
      role="img"
      aria-label={`Two parallel encoders, one for ${FEATURES.eeg.total} EEG features and one for ${FEATURES.fnirs.total} fNIRS features, each projecting to ${MODEL.projDim} then ${MODEL.encDim} dimensions, combined by an attention fusion layer and passed to a classifier head that outputs two class probabilities.`}
    >
      {/* Input → projection */}
      <Link d={`M168 ${eegY + 26} H208`} delay={0.15} reduced={reduced} color="var(--eeg)" />
      <Link d={`M168 ${fnirsY + 26} H208`} delay={0.15} reduced={reduced} color="var(--fnirs)" />

      {/* Projection → encoder */}
      <Link d={`M318 ${eegY + 26} H358`} delay={0.35} reduced={reduced} color="var(--eeg)" />
      <Link d={`M318 ${fnirsY + 26} H358`} delay={0.35} reduced={reduced} color="var(--fnirs)" />

      {/* Encoders → fusion (curved, converging) */}
      <Link
        d={`M468 ${eegY + 26} H500 Q524 ${eegY + 26} 524 ${midY + 12} V${midY + 14}`}
        delay={0.55}
        reduced={reduced}
        color="var(--eeg)"
      />
      <Link
        d={`M468 ${fnirsY + 26} H500 Q524 ${fnirsY + 26} 524 ${midY + 40} V${midY + 38}`}
        delay={0.55}
        reduced={reduced}
        color="var(--fnirs)"
      />

      {/* Fusion → head → output */}
      <Link d={`M596 ${midY + 26} H636`} delay={0.75} reduced={reduced} color="var(--accent)" />

      {/* ── Inputs ── */}
      <Box
        x={28} y={eegY} w={140} h={52}
        title="EEG features"
        sub={`${FEATURES.eeg.total}-d`}
        color="var(--eeg)"
        delay={0} reduced={reduced}
      />
      <Box
        x={28} y={fnirsY} w={140} h={52}
        title="fNIRS features"
        sub={`${FEATURES.fnirs.total}-d`}
        color="var(--fnirs)"
        delay={0.05} reduced={reduced}
      />

      {/* ── Projection ── */}
      <Box
        x={208} y={eegY} w={110} h={52}
        title="Projection"
        sub={`→ ${MODEL.projDim}-d`}
        delay={0.25} reduced={reduced}
      />
      <Box
        x={208} y={fnirsY} w={110} h={52}
        title="Projection"
        sub={`→ ${MODEL.projDim}-d`}
        delay={0.25} reduced={reduced}
      />

      {/* ── Encoders ── */}
      <Box
        x={358} y={eegY} w={110} h={52}
        title="Encoder"
        sub={`${MODEL.nBlocks} blocks → ${MODEL.encDim}-d`}
        delay={0.45} reduced={reduced}
      />
      <Box
        x={358} y={fnirsY} w={110} h={52}
        title="Encoder"
        sub={`${MODEL.nBlocks} blocks → ${MODEL.encDim}-d`}
        delay={0.45} reduced={reduced}
      />

      {/* ── Fusion ── */}
      <Box
        x={486} y={midY} w={110} h={52}
        title="Attention"
        sub="gated fusion"
        color="var(--accent)"
        delay={0.65} reduced={reduced}
      />

      {/* ── Head ── */}
      <Box
        x={636} y={midY} w={116} h={52}
        title="Classifier"
        sub={`FC ${MODEL.fcUnits.join(" → ")}`}
        color="var(--accent)"
        delay={0.85} reduced={reduced}
      />

      {/* Output labels */}
      <motion.g
        initial={reduced ? false : { opacity: 0 }}
        whileInView={reduced ? undefined : { opacity: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.4, delay: 1.05 }}
      >
        <text x={694} y={midY + 74} textAnchor="middle" className="fill-low font-mono" style={{ fontSize: 11, fontWeight: 500 }}>
          P(Low)
        </text>
        <text x={694} y={midY + 90} textAnchor="middle" className="fill-high font-mono" style={{ fontSize: 11, fontWeight: 500 }}>
          P(High)
        </text>
      </motion.g>

      {/* Dropout annotation, the regularisation is aggressive and worth stating */}
      <motion.text
        x={412} y={H - 8}
        textAnchor="middle"
        className="fill-ink-3 font-mono"
        style={{ fontSize: 10 }}
        initial={reduced ? false : { opacity: 0 }}
        whileInView={reduced ? undefined : { opacity: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.4, delay: 1.15 }}
      >
        encoder dropout {MODEL.encoderDropout} · classifier dropout {MODEL.fcDropout}
      </motion.text>
    </svg>
  );
}
