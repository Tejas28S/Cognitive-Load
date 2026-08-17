// Data-export helpers: turn prediction results into CSV / JSON downloads and
// copy-to-clipboard. Kept dependency-free so any page can export its own data.

const triggerDownload = (blob, filename) => {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Revoke after the click has been dispatched.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

const csvCell = (v) => {
  if (v === null || v === undefined) return "";
  const s = String(v);
  // Quote if it contains a comma, quote, or newline; double interior quotes.
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

// rows: array of flat objects. Columns are the union of keys (first row's order
// first, then any extras), so ragged rows still export cleanly.
export function toCsv(rows) {
  if (!rows || rows.length === 0) return "";
  const cols = [];
  const seen = new Set();
  for (const row of rows) {
    for (const k of Object.keys(row)) {
      if (!seen.has(k)) {
        seen.add(k);
        cols.push(k);
      }
    }
  }
  const head = cols.map(csvCell).join(",");
  const body = rows.map((r) => cols.map((c) => csvCell(r[c])).join(",")).join("\n");
  return `${head}\n${body}`;
}

const ensureExt = (name, ext) =>
  name.toLowerCase().endsWith(`.${ext}`) ? name : `${name}.${ext}`;

export function downloadCsv(rows, filename = "export") {
  const csv = toCsv(rows);
  triggerDownload(new Blob([csv], { type: "text/csv;charset=utf-8" }), ensureExt(filename, "csv"));
  return rows.length;
}

export function downloadJson(data, filename = "export") {
  const json = JSON.stringify(data, null, 2);
  triggerDownload(new Blob([json], { type: "application/json" }), ensureExt(filename, "json"));
}

export async function copyJson(data) {
  const json = typeof data === "string" ? data : JSON.stringify(data, null, 2);
  await navigator.clipboard.writeText(json);
}

// A datestamped slug for filenames, e.g. "live_subject-3_2026-07-16".
export function stampFilename(parts) {
  const date = new Date().toISOString().slice(0, 10);
  return [...(Array.isArray(parts) ? parts : [parts]), date]
    .filter(Boolean)
    .join("_");
}
