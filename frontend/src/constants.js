// Shared design constants. Low/Medium/High are ordinal cognitive-load levels;
// the calm→warm ramp is used consistently across every chart (validated
// colorblind-safe: worst adjacent CVD ΔE 16.9, all pass contrast vs surface).

export const CLASS_NAMES = ["Low", "Medium", "High"];

export const CLASS_COLORS = {
  Low: "#0d9488", // teal — calm / under-engaged end
  Medium: "#d97706", // amber — the sustained-engagement middle
  High: "#dc2626", // red — high load / fatigue-risk end
};

// Softer fills for backgrounds / shaded zones.
export const CLASS_TINTS = {
  Low: "rgba(13, 148, 136, 0.14)",
  Medium: "rgba(217, 119, 6, 0.14)",
  High: "rgba(220, 38, 38, 0.14)",
};

export const MODALITY_COLORS = {
  eeg: "#4f46e5", // indigo
  fnirs: "#0891b2", // cyan
};

// Categorical pair for research-page comparisons (baseline↔variant, 468↔936,
// within↔LOSO). Validated colorblind-safe (indigo↔cyan adjacent ΔE 54 deutan,
// 27 tritan; both pass chroma + contrast). Reference/weaker = indigo (focus on
// the honest number), variant/tested = cyan. Legend always labels them.
export const COMPARE_COLORS = ["#4f46e5", "#0891b2"];
export const REFERENCE_GRAY = "#94a3b8"; // for dashed reference lines (chance, thresholds)

export const colorForLabel = (label) => CLASS_COLORS[label] || "#64748b";

export const labelName = (idx) => CLASS_NAMES[idx] ?? String(idx);

export const fmtPct = (x) => `${(x * 100).toFixed(1)}%`;
export const fmt = (x, d = 4) =>
  x === null || x === undefined || Number.isNaN(x) ? "—" : Number(x).toFixed(d);
