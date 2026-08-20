// Run selector. Choosing a run re-loads every section of the page from that
// run's artifacts, so a page can never mix run A's metrics with run B's plots.

const statusStyle = (status, stale) => {
  if (stale) return { bg: "rgba(217,119,6,0.14)", fg: "#b45309", text: "interrupted" };
  return (
    {
      completed: { bg: "rgba(13,148,136,0.14)", fg: "#0f766e", text: "completed" },
      running: { bg: "rgba(79,70,229,0.14)", fg: "#4338ca", text: "running" },
      failed: { bg: "rgba(220,38,38,0.14)", fg: "#b91c1c", text: "failed" },
      created: { bg: "rgba(100,116,139,0.14)", fg: "#475569", text: "created" },
    }[status] || { bg: "rgba(100,116,139,0.14)", fg: "#475569", text: status || "unknown" }
  );
};

export function StatusBadge({ status, stale }) {
  const s = statusStyle(status, stale);
  return (
    <span
      className="mono"
      style={{
        background: s.bg,
        color: s.fg,
        borderRadius: 999,
        padding: "2px 9px",
        fontSize: 10.5,
        fontWeight: 600,
        letterSpacing: "0.03em",
        whiteSpace: "nowrap",
      }}
    >
      {s.text}
    </span>
  );
}

const shortDate = (iso) => {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
};

export default function RunSelector({ runs, value, onChange, label = "Evaluation run" }) {
  const selected = runs.find((r) => r.run_id === value);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 320 }}>
      <label
        className="stat-label"
        htmlFor="run-selector"
        style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.04em" }}
      >
        {label}
      </label>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <select
          id="run-selector"
          className="mono"
          value={value || ""}
          onChange={(e) => onChange(e.target.value)}
          style={{
            padding: "8px 10px",
            borderRadius: 8,
            border: "1px solid var(--rule)",
            background: "var(--surface)",
            color: "var(--ink)",
            fontSize: 12.5,
            minWidth: 300,
          }}
        >
          {runs.length === 0 && <option value="">No runs in the archive</option>}
          {runs.map((r) => (
            <option key={r.run_id} value={r.run_id}>
              {r.run_id}
              {r.tag ? ` — ${r.tag}` : ""}
              {r.status !== "completed" ? ` [${r.status}]` : ""}
            </option>
          ))}
        </select>
        {selected && <StatusBadge status={selected.status} stale={selected.stale} />}
      </div>
      {selected && (
        <div className="muted" style={{ fontSize: 11.5 }}>
          {shortDate(selected.created_at)}
          {selected.n_experiments ? ` · ${selected.n_experiments} experiments` : ""}
          {selected.protocol ? ` · ${selected.protocol}` : ""}
        </div>
      )}
    </div>
  );
}
