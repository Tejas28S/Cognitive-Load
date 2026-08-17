// Hand-authored SVG contrasting the two evaluation protocols and WHY the fixed
// split leaks. Each colored dot is one trial; its shape-group is a subject.
// Fixed split: a subject's trials land in BOTH train and test (dashed leak
// arrows) → the model can memorize that person. LOSO: whole subjects are held
// out, so every test subject is unseen. Theme-aware via CSS vars.
const TRAIN = "var(--eeg)";
const TEST = "var(--fnirs)";

function SubjectRow({ y, dots, label }) {
  return (
    <g>
      <text x="8" y={y + 4} fontSize="11" fill="var(--ink-3)" fontFamily="var(--font-mono)">
        {label}
      </text>
      {dots.map((d, i) => (
        <circle key={i} cx={70 + i * 22} cy={y} r="7" fill={d === "train" ? TRAIN : TEST} />
      ))}
    </g>
  );
}

export default function LeakageSchematic() {
  return (
    <svg viewBox="0 0 920 300" width="100%" role="img"
      aria-label="Fixed-split subject leakage versus leave-one-subject-out">
      {/* Panel titles */}
      <text x="10" y="24" fontSize="14" fontWeight="700" fill="var(--ink)" fontFamily="var(--font-display)">
        Fixed split - subject leakage
      </text>
      <text x="480" y="24" fontSize="14" fontWeight="700" fill="var(--ink)" fontFamily="var(--font-display)">
        Leave-one-subject-out (LOSO)
      </text>

      {/* Divider */}
      <line x1="460" y1="10" x2="460" y2="290" stroke="var(--border)" strokeWidth="1.5" strokeDasharray="4 4" />

      {/* ── Fixed split: each subject's trials split across train & test ── */}
      <SubjectRow y={70} label="Subj A" dots={["train", "train", "test", "train", "test"]} />
      <SubjectRow y={110} label="Subj B" dots={["test", "train", "train", "test", "train"]} />
      <SubjectRow y={150} label="Subj C" dots={["train", "test", "train", "test", "train"]} />
      {/* Leak arrows: same subject present in train and test */}
      <path d="M92,84 C 120,110 150,110 158,96" fill="none" stroke="var(--high)" strokeWidth="1.6" strokeDasharray="4 3" markerEnd="url(#leak)" />
      <text x="120" y="205" fontSize="11" fill="var(--high)" fontFamily="var(--font-mono)">
        same person in train &amp; test
      </text>
      <text x="70" y="250" fontSize="11.5" fill="var(--ink-2)">
        Model can lean on a person's baseline →
      </text>
      <text x="70" y="268" fontSize="11.5" fill="var(--ink-2)">
        optimistic score, poor transfer.
      </text>

      {/* ── LOSO: whole subjects held out ── */}
      <g transform="translate(470,0)">
        <SubjectRow y={70} label="Subj A" dots={["train", "train", "train", "train", "train"]} />
        <SubjectRow y={110} label="Subj B" dots={["train", "train", "train", "train", "train"]} />
        <SubjectRow y={150} label="Subj C" dots={["test", "test", "test", "test", "test"]} />
        <rect x="60" y="136" width="122" height="28" rx="8" fill="none" stroke="var(--fnirs)" strokeWidth="1.6" />
        <text x="66" y="205" fontSize="11" fill="var(--fnirs)" fontFamily="var(--font-mono)">
          one whole subject held out
        </text>
        <text x="60" y="250" fontSize="11.5" fill="var(--ink-2)">
          Test person is entirely unseen →
        </text>
        <text x="60" y="268" fontSize="11.5" fill="var(--ink-2)">
          the honest generalization estimate.
        </text>
      </g>

      {/* Legend */}
      <g transform="translate(300,286)">
        <circle cx="0" cy="-4" r="6" fill={TRAIN} />
        <text x="12" y="0" fontSize="11" fill="var(--ink-2)">train trial</text>
        <circle cx="90" cy="-4" r="6" fill={TEST} />
        <text x="102" y="0" fontSize="11" fill="var(--ink-2)">test trial</text>
      </g>

      <defs>
        <marker id="leak" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="var(--high)" />
        </marker>
      </defs>
    </svg>
  );
}
