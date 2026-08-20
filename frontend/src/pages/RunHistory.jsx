import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { StatusBadge } from "../components/evaluation/RunSelector";
import { getRuns } from "../api/client";
import { useApi, ErrorBox, Loading } from "../useApi";

const n = (v, d = 4) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d);

const when = (iso) => {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
};

export default function RunHistory() {
  const { data, loading, error } = useApi(() => getRuns({ limit: 300 }), []);
  const [mode, setMode] = useState("all");
  const [status, setStatus] = useState("all");

  const runs = useMemo(() => data?.runs || [], [data]);
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

  const statuses = [...new Set(runs.map((r) => r.status))].sort();

  return (
    <div className="container">
      <div className="page-header">
        <div className="eyebrow eyebrow--accent" style={{ marginBottom: 8 }}>Archive</div>
        <h1>Run History</h1>
        <p className="lede">
          Every training run ever executed, listed from the run archive at{" "}
          <span className="mono">{data?.runs_root}</span>. Runs are never
          overwritten and failed runs are never deleted — a failure is kept with
          its metadata, config and logs so it can be debugged.
        </p>
      </div>

      <div className="card" style={{ display: "flex", gap: 22, flexWrap: "wrap", marginBottom: 22 }}>
        <Filter label="Classification mode" value={mode} onChange={setMode}
          options={["all", ...(data?.modes || [])]} />
        <Filter label="Status" value={status} onChange={setStatus}
          options={["all", ...statuses]} />
        <div style={{ marginLeft: "auto", alignSelf: "flex-end" }}>
          <span className="mono muted" style={{ fontSize: 12 }}>
            {filtered.length} of {runs.length} runs
          </span>
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="note">
          No runs match this filter. Run{" "}
          <span className="mono">.venv/bin/python run_campaign.py</span> to create one.
        </div>
      ) : (
        <div className="card" style={{ overflowX: "auto", padding: 0 }}>
          <table className="mono" style={{ width: "100%", fontSize: 11.5, borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ textAlign: "left", color: "var(--ink-2)" }}>
                {["Run", "Status", "Created", "Dataset", "Mode", "Protocol",
                  "LOSO macro F1", "Lockbox macro F1", "Exps", ""].map((h, i) => (
                  <th key={h} style={{
                    padding: "10px 12px",
                    textAlign: i >= 6 && i <= 8 ? "right" : "left",
                    borderBottom: "1px solid var(--rule)",
                    whiteSpace: "nowrap",
                  }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => (
                <tr key={r.run_id} style={{ borderBottom: "1px solid var(--rule)" }}>
                  <td style={{ padding: "10px 12px" }}>
                    <Link to={`/evaluation?run=${r.run_id}`} style={{ fontWeight: 600 }}>
                      {r.run_id}
                    </Link>
                    {r.tag && (
                      <div className="muted" style={{ fontSize: 10.5, marginTop: 2 }}>{r.tag}</div>
                    )}
                    {(r.warnings || []).length > 0 && (
                      <div style={{ fontSize: 10, marginTop: 3, color: "#b45309" }}>
                        {r.warnings.length} warning{r.warnings.length > 1 ? "s" : ""}
                      </div>
                    )}
                  </td>
                  <td style={{ padding: "10px 12px" }}>
                    <StatusBadge status={r.status} stale={r.stale} />
                    {r.status === "failed" && r.error?.message && (
                      <div className="muted" style={{ fontSize: 10, marginTop: 3, maxWidth: 220 }}>
                        {r.stage}: {String(r.error.message).slice(0, 70)}
                      </div>
                    )}
                    {r.status === "running" && !r.stale && (
                      <div className="muted" style={{ fontSize: 10, marginTop: 3 }}>{r.stage}</div>
                    )}
                  </td>
                  <td style={{ padding: "10px 12px", color: "var(--ink-2)", whiteSpace: "nowrap" }}>
                    {when(r.created_at)}
                  </td>
                  <td style={{ padding: "10px 12px", color: "var(--ink-2)" }}>{r.dataset || "—"}</td>
                  <td style={{ padding: "10px 12px" }}>
                    {r.classification_mode || "—"}
                    {r.class_names?.length ? (
                      <div className="muted" style={{ fontSize: 10 }}>{r.class_names.join(" / ")}</div>
                    ) : null}
                  </td>
                  <td style={{ padding: "10px 12px", color: "var(--ink-2)", maxWidth: 200 }}>
                    {r.protocol || "—"}
                  </td>
                  <td style={{ padding: "10px 12px", textAlign: "right", fontWeight: 600,
                    fontVariantNumeric: "tabular-nums" }}>
                    {n(r.metrics?.loso_f1_macro)}
                  </td>
                  <td style={{ padding: "10px 12px", textAlign: "right",
                    fontVariantNumeric: "tabular-nums" }}>
                    {n(r.metrics?.lockbox_f1_macro)}
                  </td>
                  <td style={{ padding: "10px 12px", textAlign: "right", color: "var(--ink-2)" }}>
                    {r.n_experiments ?? "—"}
                  </td>
                  <td style={{ padding: "10px 12px" }}>
                    <Link to={`/evaluation?run=${r.run_id}`}>open →</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="muted" style={{ fontSize: 12, marginTop: 16, lineHeight: 1.6 }}>
        A run marked <em>interrupted</em> recorded itself as running but has not
        written an update in over 30 minutes — its process was most likely killed.
        The status stored on disk is left untouched; only this view flags it.
      </p>
    </div>
  );
}

function Filter({ label, value, onChange, options }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
      <span className="stat-label" style={{ fontSize: 10.5, textTransform: "uppercase" }}>
        {label}
      </span>
      <select
        className="mono"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{
          padding: "6px 10px", borderRadius: 8, border: "1px solid var(--rule)",
          background: "var(--surface)", color: "var(--ink)", fontSize: 12,
        }}
      >
        {options.map((o) => (
          <option key={o} value={o}>{o}</option>
        ))}
      </select>
    </div>
  );
}
