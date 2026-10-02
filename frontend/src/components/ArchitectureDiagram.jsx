import { useModelVersion } from "../lib/modelVersion";
import { colorForLabel } from "../constants";
import { FEATURES } from "../lib/projectFacts";

// Hand-drawn architecture of the currently served model: two modality encoders
// fuse under a learned attention gate into a small classifier head (the Hybrid
// net) whose softmax is the final prediction. The legacy Random Forest ensemble
// is disabled in serving (model_service.py sets self.rf = None) - the Hybrid
// network alone outperformed it, so nothing is averaged in at inference time.
// Inputs carry the modality colors.
export default function ArchitectureDiagram() {
  // The output labels track whichever bundle is being served, so the binary
  // model does not advertise a Medium class it cannot predict.
  const { active } = useModelVersion();
  const classes = active?.class_names?.length ? active.class_names : ["Low", "High"];

  return (
    <div className="chart-wrap">
      <svg viewBox="0 0 800 220" width="100%" role="img"
        aria-label={`EEG and fNIRS features feed two encoders that fuse via attention into a classifier head, producing ${classes.join(" / ")} probabilities directly (no ensemble at inference time).`}
        style={{ minWidth: 560, fontFamily: "var(--font-sans)" }}>
        <defs>
          <marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">
            <path d="M0,0 L7,3.2 L0,6.4 Z" fill="#9aa4b2" />
          </marker>
        </defs>

        {/* edges (drawn first, under nodes) */}
        <g stroke="#c4ccd6" strokeWidth="1.6" fill="none" markerEnd="url(#ah)">
          <path d="M170,86 C200,86 200,120 232,120" />
          <path d="M170,150 C200,150 205,124 232,124" />
          <path d="M366,122 H408" />
          <path d="M542,122 H584" />
          <path d="M716,122 H748" />
        </g>

        {/* Inputs */}
        <Node x={22} y={64} w={148} h={44} title="EEG"
          sub={`${FEATURES.eeg.total} band-power feats`} tint="var(--eeg)" />
        <Node x={22} y={128} w={148} h={44} title="fNIRS · HbO+HbR"
          sub={`${FEATURES.fnirs.total} response feats`} tint="var(--fnirs)" />

        {/* Encoders */}
        <Node x={232} y={98} w={134} h={48} title="EEG encoder" sub="MLP + residual" />
        <Node x={232} y={158} w={134} h={44} title="fNIRS encoder" sub="MLP + residual" />

        {/* Fusion */}
        <Node x={408} y={100} w={134} h={44} title="Attention fusion" sub="learned gate" accent />

        {/* Head */}
        <Node x={584} y={100} w={132} h={44} title="Classifier head" sub="→ softmax" accent />

        {/* Output labels */}
        <g fontFamily="var(--font-mono)" fontSize="12" fontWeight="600">
          {classes.map((c, i) => (
            <text key={c} x="756" y={116 + i * 18 - (classes.length - 1) * 9} textAnchor="start" fill={colorForLabel(c)}>
              {c}
            </text>
          ))}
        </g>
      </svg>
    </div>
  );
}

function Node({ x, y, w, h, title, sub, tint, accent, big }) {
  const stroke = accent ? "var(--accent)" : tint || "var(--border-strong)";
  const fill = accent ? "var(--accent-soft)" : tint ? tintBg(tint) : "var(--surface)";
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx="10"
        fill={fill} stroke={stroke} strokeWidth={accent ? 1.6 : 1.2} />
      <text x={x + w / 2} y={y + (sub ? h / 2 - 2 : h / 2 + 4)} textAnchor="middle"
        fontFamily="var(--font-sans)" fontSize={big ? 14 : 13} fontWeight="600"
        fill="var(--ink)">{title}</text>
      {sub && (
        <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle"
          fontFamily="var(--font-mono)" fontSize="10.5" fill="var(--ink-3)">{sub}</text>
      )}
    </g>
  );
}

// translucent tint background from a modality color
function tintBg(v) {
  if (v === "var(--eeg)") return "rgba(99,102,241,0.09)";
  if (v === "var(--fnirs)") return "rgba(6,182,212,0.10)";
  return "var(--surface)";
}
