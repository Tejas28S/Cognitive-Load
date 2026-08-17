// The domain-adversarial setup. The fused embedding feeds two heads: a TASK head
// (predict cognitive load - minimized) and a SUBJECT discriminator (predict who -
// adversarial). A gradient-reversal layer sits before the discriminator: identity
// on the forward pass, ×(−λ) on the backward pass, so the same signal that trains
// the discriminator pushes the ENCODER to erase subject identity.
export default function DannDiagram() {
  return (
    <div className="chart-wrap">
      <svg viewBox="0 0 940 300" width="100%" role="img"
        aria-label="Fused embedding feeds a task head (minimized) and, through a gradient-reversal layer, a subject discriminator (adversarial)."
        style={{ minWidth: 640, fontFamily: "var(--font-sans)" }}>
        <defs>
          <marker id="ahd" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">
            <path d="M0,0 L7,3.2 L0,6.4 Z" fill="#9aa4b2" />
          </marker>
          <marker id="ahr" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">
            <path d="M0,0 L7,3.2 L0,6.4 Z" fill="var(--high)" />
          </marker>
        </defs>

        <g stroke="#c4ccd6" strokeWidth="1.6" fill="none" markerEnd="url(#ahd)">
          <path d="M158,150 H214" />
          <path d="M368,150 C398,150 398,80 430,80" />
          <path d="M368,150 C398,150 398,214 430,214" />
          <path d="M602,80 H660" />
          <path d="M626,214 H660" />
        </g>

        {/* Inputs → encoder → embedding */}
        <Box x={22} y={126} w={136} h={48} title="EEG + fNIRS" sub="fused input" tint />
        <Box x={214} y={122} w={154} h={56} title="Encoders" sub="→ fused embedding" accent />

        {/* Task head (minimize) */}
        <Box x={430} y={56} w={172} h={48} title="Task head" sub="cognitive load" />
        <g fontFamily="var(--font-mono)" fontSize="11.5" fontWeight="600">
          <text x="672" y="76" fill="var(--low)">Low</text>
          <text x="716" y="76" fill="var(--medium)">Medium</text>
          <text x="776" y="76" fill="var(--high)">High</text>
        </g>
        <text x="516" y="128" textAnchor="middle" fontFamily="var(--font-mono)" fontSize="11" fill="#0f766e">
          minimize task loss
        </text>

        {/* GRL + subject discriminator (adversarial) */}
        <Box x={430} y={190} w={90} h={48} title="GRL" sub="−λ" stop />
        <Box x={528} y={190} w={132} h={48} title="Subject head" sub="which subject?" />
        <text x="700" y="219" fontFamily="var(--font-mono)" fontSize="11.5" fontWeight="600" fill="var(--high)">
          adversarial
        </text>

        {/* Reverse-gradient annotation */}
        <path d="M474,190 C474,150 400,150 372,150" stroke="var(--high)" strokeWidth="1.6"
          strokeDasharray="5 4" fill="none" markerEnd="url(#ahr)" />
        <text x="404" y="182" fontFamily="var(--font-mono)" fontSize="10.5" fill="var(--high)">
          reversed grad ×(−λ)
        </text>
      </svg>
    </div>
  );
}

function Box({ x, y, w, h, title, sub, tint, accent, stop }) {
  const stroke = stop ? "var(--high)" : accent ? "var(--accent)" : tint ? "var(--fnirs)" : "var(--border-strong)";
  const fill = stop ? "rgba(220,38,38,0.06)" : accent ? "var(--accent-soft)" : tint ? "rgba(6,182,212,0.09)" : "var(--surface)";
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx="10" fill={fill} stroke={stroke}
        strokeWidth={accent || stop ? 1.6 : 1.2} />
      <text x={x + w / 2} y={y + h / 2 - 2} textAnchor="middle" fontFamily="var(--font-sans)"
        fontSize="13" fontWeight="600" fill="var(--ink)">{title}</text>
      <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fontFamily="var(--font-mono)"
        fontSize="10.5" fill={stop ? "var(--high)" : "var(--ink-3)"}>{sub}</text>
    </g>
  );
}
