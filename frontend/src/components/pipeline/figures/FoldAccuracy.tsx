import {
  Bar,
  BarChart,
  Cell,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getRunPerSubject } from "../../../api/client";
import { useApi } from "../../../useApi";
import { useChartColors, usePrefersReducedMotion } from "../../../lib/chartTheme";
import { HEADLINE_RUN } from "../../../lib/projectFacts";

type Fold = {
  subject_id: number;
  fold: number;
  n_windows: number;
  accuracy: number;
  f1_macro: number;
  roc_auc: number;
};

type PerSubjectResponse = {
  per_subject: Fold[];
  per_subject_experiment: string | null;
};

/**
 * Every LOSO fold, one bar per held-out subject.
 *
 * This is the figure that makes the headline mean honest. A single "63%" reads
 * as a steady capability; the spread here shows the same frozen model ranging
 * from near chance to the high seventies depending on whose brain it is
 * looking at. The dashed line is the mean the headline quotes.
 */
export default function FoldAccuracy({ height = 300 }: { height?: number }) {
  const c = useChartColors();
  const reduced = usePrefersReducedMotion();

  const { data, loading, error } = useApi<PerSubjectResponse>(
    () =>
      getRunPerSubject<PerSubjectResponse>(
        HEADLINE_RUN.runId,
        HEADLINE_RUN.baselineExperiment
      ),
    []
  );

  const folds = data?.per_subject ?? [];

  if (loading) {
    return (
      <div
        className="animate-pulse rounded-sm bg-surface-2"
        style={{ height }}
        aria-busy="true"
      />
    );
  }

  if (error || !folds.length) {
    return (
      <div
        className="flex items-center justify-center rounded-sm border border-dashed border-line-strong bg-surface-2 p-8 text-center"
        style={{ minHeight: height }}
      >
        <p className="max-w-[46ch] text-sm text-ink-2">
          Per-fold accuracies are unavailable, the run archive entry{" "}
          <code className="font-mono text-xs text-ink-3">
            {HEADLINE_RUN.runId}
          </code>{" "}
          could not be read. Start the API, or check the archive is present.
          This figure never substitutes cached numbers.
        </p>
      </div>
    );
  }

  const rows = folds.map((f) => ({
    subject: `S${f.subject_id}`,
    accuracy: f.accuracy,
    auc: f.roc_auc,
    windows: f.n_windows,
  }));

  const mean = rows.reduce((a, r) => a + r.accuracy, 0) / rows.length;
  const best = Math.max(...rows.map((r) => r.accuracy));
  const worst = Math.min(...rows.map((r) => r.accuracy));

  return (
    <div className="min-w-[600px]">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 4, left: -8 }}>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis
            dataKey="subject"
            tick={{ fill: c.axis, fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: c.grid }}
            interval={0}
          />
          <YAxis
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            tick={{ fill: c.axis, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={46}
          />
          <Tooltip
            cursor={{ fill: "rgba(var(--accent-rgb), 0.06)" }}
            contentStyle={{
              background: c.surface,
              border: `1px solid ${c.grid}`,
              borderRadius: 9,
              fontSize: 12,
            }}
            labelStyle={{ color: c.ink, fontWeight: 600 }}
            formatter={(value) => [
              `${(Number(value) * 100).toFixed(1)}%`,
              "Accuracy",
            ]}
          />

          {/* Chance, then the mean the headline reports. */}
          <ReferenceLine
            y={0.5}
            stroke={c.reference}
            strokeDasharray="2 4"
            label={{
              value: "chance",
              position: "insideTopLeft",
              fill: c.axis,
              fontSize: 10,
            }}
          />
          <ReferenceLine
            y={mean}
            stroke={c.accent}
            strokeDasharray="5 4"
            label={{
              value: `mean ${(mean * 100).toFixed(1)}%`,
              position: "insideTopRight",
              fill: c.accent,
              fontSize: 10,
            }}
          />

          <Bar
            dataKey="accuracy"
            radius={[2, 2, 0, 0]}
            isAnimationActive={!reduced}
            animationDuration={700}
          >
            {rows.map((r) => (
              <Cell
                key={r.subject}
                // Below chance is the honest alarm colour: for those subjects
                // the model is worse than a coin flip, and shading them like
                // the rest would hide it.
                fill={r.accuracy < 0.5 ? c.high : c.accent}
                fillOpacity={r.accuracy < 0.5 ? 0.75 : 0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <dl className="mt-4 flex flex-wrap gap-x-8 gap-y-2 font-mono text-2xs text-ink-3">
        <div className="flex gap-2">
          <dt>folds</dt>
          <dd className="text-ink-2 tnum">{rows.length}</dd>
        </div>
        <div className="flex gap-2">
          <dt>best</dt>
          <dd className="text-ink-2 tnum">{(best * 100).toFixed(1)}%</dd>
        </div>
        <div className="flex gap-2">
          <dt>worst</dt>
          <dd className="text-ink-2 tnum">{(worst * 100).toFixed(1)}%</dd>
        </div>
        <div className="flex gap-2">
          <dt>spread</dt>
          <dd className="text-ink-2 tnum">
            {((best - worst) * 100).toFixed(1)} pts
          </dd>
        </div>
      </dl>
    </div>
  );
}
