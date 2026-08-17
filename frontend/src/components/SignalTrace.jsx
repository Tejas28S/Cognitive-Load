import { useId } from "react";

// The signature element: a single trace that begins as jagged EEG spikes
// (electrical, left) and relaxes into a smooth fNIRS hemodynamic curve
// (hemodynamic, right). The indigo→cyan gradient IS the EEG+fNIRS fusion the
// whole project is about - not decoration.

const EEG_SPIKES = [
  [0, 50], [12, 44], [20, 59], [30, 28], [40, 66], [50, 39], [62, 56],
  [72, 20], [82, 70], [92, 45], [104, 54], [116, 34], [126, 62], [138, 47],
  [150, 53], [162, 32], [174, 68], [186, 43], [198, 55], [210, 49], [220, 50],
];

// EEG zigzag → then a smooth cubic hemodynamic bump that returns to baseline.
const TRACE_PATH = `M${EEG_SPIKES.map((p) => p.join(",")).join(" L")} C 250,50 270,15 332,17 C 384,19 412,53 470,51 C 522,50 562,50 600,50`;

export default function SignalTrace({
  width = "100%",
  height = 60,
  className = "",
  animate = false,
  strokeWidth = 2,
  glow = false,
}) {
  const id = useId().replace(/:/g, "");
  return (
    <svg
      className={className}
      width={width}
      height={height}
      viewBox="0 0 600 100"
      preserveAspectRatio="none"
      fill="none"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id={`sig-${id}`} x1="0" y1="0" x2="600" y2="0" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="var(--eeg)" />
          <stop offset="0.42" stopColor="var(--eeg)" />
          <stop offset="0.62" stopColor="var(--fnirs)" />
          <stop offset="1" stopColor="var(--fnirs)" />
        </linearGradient>
        {glow && (
          <filter id={`glow-${id}`} x="-5%" y="-40%" width="110%" height="180%">
            <feGaussianBlur stdDeviation="3" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        )}
      </defs>
      <path
        d={TRACE_PATH}
        stroke={`url(#sig-${id})`}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
        filter={glow ? `url(#glow-${id})` : undefined}
        className={animate ? "sig-draw" : undefined}
      />
      <style>{`
        .sig-draw {
          stroke-dasharray: 1400;
          stroke-dashoffset: 1400;
          animation: sigDraw 2.2s var(--ease) 0.15s forwards;
        }
        @keyframes sigDraw { to { stroke-dashoffset: 0; } }
        @media (prefers-reduced-motion: reduce) {
          .sig-draw { animation: none; stroke-dashoffset: 0; }
        }
      `}</style>
    </svg>
  );
}

// Compact mark for the nav brand - same motif, tighter.
export function BrandMark() {
  const id = useId().replace(/:/g, "");
  return (
    <svg className="brand-mark" viewBox="0 0 120 60" fill="none" aria-hidden="true">
      <defs>
        <linearGradient id={`bm-${id}`} x1="0" y1="0" x2="120" y2="0" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="var(--eeg)" />
          <stop offset="0.5" stopColor="var(--eeg)" />
          <stop offset="0.72" stopColor="var(--fnirs)" />
          <stop offset="1" stopColor="var(--fnirs)" />
        </linearGradient>
      </defs>
      <path
        d="M2,30 L10,22 L16,40 L24,10 L32,46 L40,24 L48,32 C60,32 62,8 78,10 C92,12 96,44 108,32 L118,30"
        stroke={`url(#bm-${id})`}
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  );
}
