import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { COMPARE_COLORS, MODALITY_COLORS, REFERENCE_GRAY } from "../../constants";

const AXIS = { fontSize: 11, fill: "var(--ink-2)" };
const GRID = "var(--rule)";

const tooltipStyle = {
  background: "var(--surface)",
  border: "1px solid var(--rule)",
  borderRadius: 8,
  fontSize: 12,
  color: "var(--ink)",
};

const n = (v, d = 4) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d);

// ── ROC ────────────────────────────────────────────────────────────────────
export function RocChart({ curves }) {
  if (!curves?.roc?.length) return <Empty label="No ROC curve in this run" />;
  // Merge the per-class curves onto one chart; each class keeps its own points.
  return (
    <ResponsiveContainer width="100%" height={318}>
      <LineChart margin={{ top: 4, right: 12, bottom: 24, left: 4 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis
          type="number"
          dataKey="fpr"
          domain={[0, 1]}
          tick={AXIS}
          label={{ value: "false positive rate", position: "insideBottom", offset: -12, ...AXIS }}
        />
        <YAxis type="number" dataKey="tpr" domain={[0, 1]} tick={AXIS}
          label={{ value: "TPR", angle: -90, position: "insideLeft", ...AXIS }} />
        <Tooltip contentStyle={tooltipStyle} formatter={(v) => n(v, 3)} />
        <Legend verticalAlign="top" height={22} wrapperStyle={{ fontSize: 11 }} />
        <ReferenceLine
          segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
          stroke={REFERENCE_GRAY}
          strokeDasharray="4 4"
        />
        {curves.roc.map((c, i) => (
          <Line
            key={c.class_name}
            data={c.points}
            dataKey="tpr"
            name={`${c.class_name} (AUC ${n(c.auc, 3)})`}
            stroke={COMPARE_COLORS[i % COMPARE_COLORS.length]}
            dot={false}
            strokeWidth={2}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

// ── Precision-Recall ───────────────────────────────────────────────────────
export function PrChart({ curves }) {
  if (!curves?.pr?.length) return <Empty label="No precision-recall curve in this run" />;
  return (
    <ResponsiveContainer width="100%" height={318}>
      <LineChart margin={{ top: 4, right: 12, bottom: 24, left: 4 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis type="number" dataKey="recall" domain={[0, 1]} tick={AXIS}
          label={{ value: "recall", position: "insideBottom", offset: -12, ...AXIS }} />
        <YAxis type="number" dataKey="precision" domain={[0, 1]} tick={AXIS}
          label={{ value: "precision", angle: -90, position: "insideLeft", ...AXIS }} />
        <Tooltip contentStyle={tooltipStyle} formatter={(v) => n(v, 3)} />
        <Legend verticalAlign="top" height={22} wrapperStyle={{ fontSize: 11 }} />
        {curves.pr.map((c) => (
          <ReferenceLine key={`b-${c.class_name}`} y={c.baseline} stroke={REFERENCE_GRAY}
            strokeDasharray="4 4" />
        ))}
        {curves.pr.map((c, i) => (
          <Line
            key={c.class_name}
            data={c.points}
            dataKey="precision"
            name={`${c.class_name} (AP ${n(c.average_precision, 3)})`}
            stroke={COMPARE_COLORS[i % COMPARE_COLORS.length]}
            dot={false}
            strokeWidth={2}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

// ── Training / validation curves ───────────────────────────────────────────
export function TrainingCurves({ history, metric = "f1_macro", label }) {
  const mean = history?.mean_over_folds;
  if (!mean?.epoch?.length) return <Empty label="No training history in this run" />;
  const rows = mean.epoch.map((e, i) => ({
    epoch: e,
    train: mean[`train_${metric}`]?.[i] ?? null,
    validation: mean[`val_${metric}`]?.[i] ?? null,
    folds: mean.n_folds?.[i] ?? null,
  }));
  return (
    <ResponsiveContainer width="100%" height={296}>
      <LineChart data={rows} margin={{ top: 4, right: 12, bottom: 24, left: 4 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis dataKey="epoch" tick={AXIS}
          label={{ value: "epoch", position: "insideBottom", offset: -12, ...AXIS }} />
        <YAxis tick={AXIS} label={{ value: label || metric, angle: -90, position: "insideLeft", ...AXIS }} />
        <Tooltip
          contentStyle={tooltipStyle}
          formatter={(v, k) => [n(v, 4), k]}
          labelFormatter={(e, p) => `epoch ${e}${p?.[0]?.payload?.folds ? ` · ${p[0].payload.folds} folds still training` : ""}`}
        />
        <Legend verticalAlign="top" height={22} wrapperStyle={{ fontSize: 11 }} />
        <Line dataKey="train" stroke={COMPARE_COLORS[0]} dot={false} strokeWidth={2}
          isAnimationActive={false} name="train" />
        <Line dataKey="validation" stroke={COMPARE_COLORS[1]} dot={false} strokeWidth={2}
          isAnimationActive={false} name="validation" />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ── Per-subject held-out performance ───────────────────────────────────────
export function PerSubjectChart({ rows, metric = "f1_macro", chance, mean }) {
  if (!rows?.length) return <Empty label="No per-subject metrics in this run" />;
  const data = [...rows]
    .map((r) => ({ subject: `S${r.subject_id}`, value: r[metric], n: r.n_windows }))
    .sort((a, b) => (a.value ?? -1) - (b.value ?? -1));
  return (
    <ResponsiveContainer width="100%" height={Math.max(260, data.length * 13)}>
      <BarChart data={data} margin={{ top: 8, right: 16, bottom: 28, left: 4 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="subject" tick={{ ...AXIS, fontSize: 9.5 }} interval={0} angle={-90}
          textAnchor="end" height={44} />
        <YAxis tick={AXIS} domain={[0, 1]}
          label={{ value: metric.replace(/_/g, " "), angle: -90, position: "insideLeft", ...AXIS }} />
        <Tooltip contentStyle={tooltipStyle}
          formatter={(v, k, p) => [`${n(v, 4)} (${p.payload.n} windows)`, metric]} />
        {chance != null && (
          <ReferenceLine y={chance} stroke={REFERENCE_GRAY} strokeDasharray="4 4"
            label={{ value: `chance ${n(chance, 2)}`, fontSize: 10, fill: "var(--ink-2)",
                     position: "insideTopRight" }} />
        )}
        {mean != null && (
          <ReferenceLine y={mean} stroke="var(--ink)" strokeWidth={1}
            label={{ value: `mean ${n(mean, 3)}`, fontSize: 10, fill: "var(--ink)",
                     position: "insideTopLeft" }} />
        )}
        <Bar dataKey="value" isAnimationActive={false}>
          {data.map((d, i) => (
            <Cell key={i} fill={mean != null && d.value < mean ? COMPARE_COLORS[1] : COMPARE_COLORS[0]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── Feature importance ─────────────────────────────────────────────────────
export function FeatureImportanceChart({ items, valueKey = "mean_abs_shap", labelKey = "feature", height }) {
  if (!items?.length) return <Empty label="No SHAP values in this run" />;
  const data = [...items].reverse().map((r) => ({
    name: r[labelKey],
    value: r[valueKey],
    modality: r.modality,
  }));
  return (
    <ResponsiveContainer width="100%" height={height || Math.max(280, data.length * 17)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 20, bottom: 20, left: 8 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" tick={AXIS}
          label={{ value: "mean |SHAP|", position: "insideBottom", offset: -10, ...AXIS }} />
        <YAxis type="category" dataKey="name" width={168} tick={{ ...AXIS, fontSize: 9.5 }} interval={0} />
        <Tooltip contentStyle={tooltipStyle} formatter={(v) => n(v, 5)} />
        <Bar dataKey="value" isAnimationActive={false}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.modality === "EEG" ? MODALITY_COLORS.eeg : MODALITY_COLORS.fnirs} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── t-SNE ──────────────────────────────────────────────────────────────────
const TSNE_PALETTE = [
  "#4f46e5", "#0891b2", "#0d9488", "#d97706", "#dc2626", "#7c3aed",
  "#059669", "#db2777", "#2563eb", "#ca8a04", "#475569", "#be123c",
];

export function TsneScatter({ points, colorBy = "y_true_name", height = 340 }) {
  if (!points?.length) return <Empty label="No embedding in this run" />;
  const groups = {};
  points.forEach((p) => {
    const key = String(colorBy === "subject_id" ? `S${p.subject_id}` : p[colorBy]);
    (groups[key] = groups[key] || []).push({ x: p.tsne_x, y: p.tsne_y });
  });
  const keys = Object.keys(groups).sort();
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 8, right: 12, bottom: 8, left: 4 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis type="number" dataKey="x" tick={false} axisLine={false} label={null} />
        <YAxis type="number" dataKey="y" tick={false} axisLine={false} label={null} />
        <ZAxis range={[8, 8]} />
        <Tooltip contentStyle={tooltipStyle} cursor={{ strokeDasharray: "3 3" }}
          formatter={() => ""} labelFormatter={() => ""} />
        <Legend wrapperStyle={{ fontSize: 10.5 }} />
        {keys.map((k, i) => (
          <Scatter key={k} name={k} data={groups[k]} fill={TSNE_PALETTE[i % TSNE_PALETTE.length]}
            fillOpacity={0.55} isAnimationActive={false} />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

// ── Experiment comparison ──────────────────────────────────────────────────
export function ExperimentComparisonChart({ experiments, metric = "f1_macro_mean", baselineId }) {
  if (!experiments?.length) return <Empty label="No experiments in this run" />;
  const data = [...experiments]
    .filter((e) => e[metric] != null)
    .sort((a, b) => a[metric] - b[metric])
    .map((e) => ({
      name: e.experiment_id,
      value: e[metric],
      std: e.f1_macro_std,
      isBaseline: e.experiment_id === baselineId,
    }));
  return (
    <ResponsiveContainer width="100%" height={Math.max(300, data.length * 19)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, bottom: 22, left: 8 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" tick={AXIS} domain={[0, "auto"]}
          label={{ value: "development LOSO macro F1", position: "insideBottom", offset: -10, ...AXIS }} />
        <YAxis type="category" dataKey="name" width={168} tick={{ ...AXIS, fontSize: 9.5 }} interval={0} />
        <Tooltip contentStyle={tooltipStyle}
          formatter={(v, k, p) => [`${n(v, 4)} ± ${n(p.payload.std, 4)}`, "macro F1"]} />
        <Bar dataKey="value" isAnimationActive={false}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.isBaseline ? REFERENCE_GRAY : COMPARE_COLORS[0]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function Empty({ label }) {
  return (
    <div className="muted" style={{ padding: "28px 8px", textAlign: "center", fontSize: 12.5 }}>
      {label}
    </div>
  );
}
