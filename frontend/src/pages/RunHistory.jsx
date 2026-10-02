import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { getRuns } from "../api/client";
import { useApi, ErrorBox, Loading } from "../useApi";
import Reveal from "../components/ui/Reveal";

const n = (v, d = 4) =>
  v === null || v === undefined || Number.isNaN(v) ? "-" : Number(v).toFixed(d);

const when = (iso) => {
  if (!iso) return "-";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
};

// Every run status gets one semantic colour token from the design system,
// no hand-picked hexes, so it tracks light/dark theme automatically.
const STATUS_TOKENS = {
  completed: { var: "--ok", label: "completed" },
  running: { var: "--accent", label: "running" },
  failed: { var: "--stop", label: "failed" },
  created: { var: "--ink-3", label: "created" },
};

function StatusPill({ status, stale }) {
  const tok = stale
    ? { var: "--warn", label: "interrupted" }
    : STATUS_TOKENS[status] || { var: "--ink-3", label: status || "unknown" };

  return (
    <span
      className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-1 font-mono text-2xs font-semibold tracking-wide"
      style={{
        background: `color-mix(in srgb, var(${tok.var}) 14%, transparent)`,
        color: `color-mix(in srgb, var(${tok.var}) 78%, var(--ink))`,
      }}
    >
      <span
        aria-hidden="true"
        className="h-1.5 w-1.5 shrink-0 rounded-full"
        style={{ background: `var(${tok.var})` }}
      />
      {tok.label}
    </span>
  );
}

function Filter({ label, value, onChange, options }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="eyebrow-mono text-ink-3">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="min-w-[180px] rounded-sm border border-line-strong bg-surface px-3 py-2 font-mono text-sm text-ink transition-colors duration-150 hover:border-accent focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}

const COLUMNS = [
  { key: "run", label: "Run", align: "left" },
  { key: "status", label: "Status", align: "left" },
  { key: "created", label: "Created", align: "left" },
  { key: "dataset", label: "Dataset", align: "left" },
  { key: "mode", label: "Mode", align: "left" },
  { key: "protocol", label: "Protocol", align: "left" },
  { key: "loso", label: "LOSO macro F1", align: "right" },
  { key: "lockbox", label: "Lockbox macro F1", align: "right" },
  { key: "exps", label: "Exps", align: "right" },
  { key: "open", label: "", align: "left" },
];

export default function RunHistory() {
  const { data, loading, error } = useApi(() => getRuns({ limit: 300 }), []);
  const [mode, setMode] = useState("all");
  const [status, setStatus] = useState("all");

  const runs = useMemo(() => data?.runs || [], [data]);
  const statuses = useMemo(
    () => [...new Set(runs.map((r) => r.status))].sort(),
    [runs]
  );
  const statusCounts = useMemo(() => {
    const c = {};
    runs.forEach((r) => {
      const key = r.stale ? "interrupted" : r.status;
      c[key] = (c[key] || 0) + 1;
    });
    return c;
  }, [runs]);

  const filtered = useMemo(
    () =>
      runs.filter(
        (r) =>
          (mode === "all" || r.classification_mode === mode) &&
          (status === "all" || r.status === status)
      ),
    [runs, mode, status]
  );

  if (loading) return <div className="container"><Loading /></div>;
  if (error) return <div className="container"><ErrorBox error={error} /></div>;

  return (
    <>
      <header className="border-b border-line bg-paper-edge">
        <div className="mx-auto max-w-(--container-wrap-wide) px-6 py-16 lg:py-20">
          <Reveal>
            <p className="eyebrow-mono text-accent">Archive</p>
          </Reveal>
          <Reveal delay={0.05}>
            <h1 className="mt-4 max-w-[26ch] font-display text-3xl font-bold tracking-tight text-ink text-balance sm:text-4xl">
              Run history
            </h1>
          </Reveal>
          <Reveal delay={0.1}>
            <p className="mt-5 max-w-[74ch] text-lg text-ink-2 text-pretty">
              Every training run ever executed, listed from the run archive
              at{" "}
              <code className="rounded-xs border border-line bg-surface px-1.5 py-0.5 font-mono text-sm text-ink">
                {data?.runs_root}
              </code>
              . Runs are never overwritten and failed runs are never deleted,
              a failure is kept with its metadata, config and logs so it
              can be debugged.
            </p>
          </Reveal>

          {runs.length > 0 && (
            <Reveal delay={0.15}>
              <div className="mt-9 flex flex-wrap items-center gap-x-8 gap-y-3 border-t border-line pt-7">
                <span className="text-sm font-medium text-ink">
                  {runs.length} run{runs.length === 1 ? "" : "s"} total
                </span>
                {Object.entries(statusCounts).map(([key, count]) => (
                  <span key={key} className="inline-flex items-center gap-2.5">
                    <StatusPill
                      status={key === "interrupted" ? "running" : key}
                      stale={key === "interrupted"}
                    />
                    <span className="tnum text-sm font-medium text-ink-2">
                      {count}
                    </span>
                  </span>
                ))}
              </div>
            </Reveal>
          )}
        </div>
      </header>

      <section className="py-12 lg:py-16">
        <div className="mx-auto max-w-(--container-wrap-wide) px-6">
          <Reveal>
            <div className="flex flex-wrap items-end gap-6 rounded-md border border-line bg-surface p-5 shadow-sm">
              <Filter
                label="Classification mode"
                value={mode}
                onChange={setMode}
                options={["all", ...(data?.modes || [])]}
              />
              <Filter
                label="Status"
                value={status}
                onChange={setStatus}
                options={["all", ...statuses]}
              />
              <span className="ml-auto font-mono text-xs text-ink-3">
                {filtered.length} of {runs.length} runs
              </span>
            </div>
          </Reveal>

          {filtered.length === 0 ? (
            <Reveal delay={0.05}>
              <div className="mt-6 rounded-md border border-line bg-surface-2 p-6 text-sm text-ink-2">
                No runs match this filter. Run{" "}
                <code className="rounded-xs border border-line bg-surface px-1.5 py-0.5 font-mono text-xs text-ink">
                  .venv/bin/python run_campaign.py
                </code>{" "}
                to create one.
              </div>
            </Reveal>
          ) : (
            <Reveal delay={0.05}>
              <div className="mt-6 overflow-x-auto rounded-md border border-line bg-surface shadow-sm">
                <table className="w-full min-w-[900px] border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-line">
                      {COLUMNS.map((c) => (
                        <th
                          key={c.key}
                          className={`whitespace-nowrap px-4 py-3 font-mono text-2xs font-semibold tracking-wide text-ink-3 uppercase ${
                            c.align === "right" ? "text-right" : "text-left"
                          }`}
                        >
                          {c.label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((r) => (
                      <tr
                        key={r.run_id}
                        className="border-b border-line transition-colors duration-150 last:border-b-0 hover:bg-surface-2"
                      >
                        <td className="px-4 py-3.5 align-top">
                          <Link
                            to={`/evaluation?run=${r.run_id}`}
                            className="font-mono text-sm font-semibold text-ink transition-colors hover:text-accent-ink"
                          >
                            {r.run_id}
                          </Link>
                          {r.tag && (
                            <div className="mt-0.5 text-xs text-ink-3">
                              {r.tag}
                            </div>
                          )}
                          {(r.warnings || []).length > 0 && (
                            <div
                              className="mt-1 text-xs font-medium"
                              style={{ color: "var(--warn)" }}
                            >
                              {r.warnings.length} warning
                              {r.warnings.length > 1 ? "s" : ""}
                            </div>
                          )}
                        </td>
                        <td className="px-4 py-3.5 align-top">
                          <StatusPill status={r.status} stale={r.stale} />
                          {r.status === "failed" && r.error?.message && (
                            <div className="mt-1.5 line-clamp-2 max-w-[240px] text-xs text-ink-3">
                              {r.stage}: {String(r.error.message).slice(0, 90)}
                            </div>
                          )}
                          {r.status === "running" && !r.stale && (
                            <div className="mt-1.5 text-xs text-ink-3">
                              {r.stage}
                            </div>
                          )}
                        </td>
                        <td className="px-4 py-3.5 align-top whitespace-nowrap text-ink-2">
                          {when(r.created_at)}
                        </td>
                        <td className="px-4 py-3.5 align-top text-ink-2">
                          {r.dataset || "-"}
                        </td>
                        <td className="px-4 py-3.5 align-top">
                          {r.classification_mode || "-"}
                          {r.class_names?.length ? (
                            <div className="mt-0.5 text-xs text-ink-3">
                              {r.class_names.join(" / ")}
                            </div>
                          ) : null}
                        </td>
                        <td className="max-w-[200px] px-4 py-3.5 align-top text-ink-2">
                          {r.protocol || "-"}
                        </td>
                        <td className="tnum px-4 py-3.5 text-right align-top font-semibold text-ink">
                          {n(r.metrics?.loso_f1_macro)}
                        </td>
                        <td className="tnum px-4 py-3.5 text-right align-top text-ink-2">
                          {n(r.metrics?.lockbox_f1_macro)}
                        </td>
                        <td className="tnum px-4 py-3.5 text-right align-top text-ink-2">
                          {r.n_experiments ?? "-"}
                        </td>
                        <td className="px-4 py-3.5 align-top">
                          <Link
                            to={`/evaluation?run=${r.run_id}`}
                            className="whitespace-nowrap text-sm font-medium text-accent transition-colors hover:text-accent-ink"
                          >
                            open →
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Reveal>
          )}

          <p className="mt-6 max-w-[80ch] text-sm leading-relaxed text-ink-3">
            A run marked <em className="text-ink-2">interrupted</em> recorded
            itself as running but has not written an update in over 30
            minutes, its process was most likely killed. The status stored
            on disk is left untouched; only this view flags it.
          </p>
        </div>
      </section>
    </>
  );
}
