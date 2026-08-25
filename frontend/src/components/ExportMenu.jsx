import { useEffect, useRef, useState } from "react";
import { downloadNodePng } from "../lib/exportImage";
import { downloadNodePdf } from "../lib/exportPdf";
import { downloadCsv, downloadJson, copyJson } from "../lib/exportData";
import { useToast } from "../lib/toast";

// A compact "Export ▾" dropdown. Renders only the actions it has data for:
//   pngRef   - a ref to the DOM node to rasterize (PNG)
//   pdfRef   - a ref to the DOM node to rasterize into a paginated PDF
//   csvRows  - array of flat row objects (CSV)
//   jsonData - any serializable value (JSON download + copy)
// `filename` is the base name (no extension). Feedback goes through toasts.
export default function ExportMenu({ pngRef, pdfRef, csvRows, jsonData, filename = "export", label = "Export" }) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const wrapRef = useRef(null);
  const toast = useToast();

  useEffect(() => {
    if (!open) return;
    const onDown = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false);
    };
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const run = async (fn, doneMsg) => {
    setBusy(true);
    try {
      await fn();
      if (doneMsg) toast.success(doneMsg);
    } catch (e) {
      toast.error(e?.message || "Export failed");
    } finally {
      setBusy(false);
      setOpen(false);
    }
  };

  const hasPng = !!pngRef;
  const hasPdf = !!pdfRef;
  const hasCsv = Array.isArray(csvRows) && csvRows.length > 0;
  const hasJson = jsonData != null;
  if (!hasPng && !hasPdf && !hasCsv && !hasJson) return null;

  return (
    <div className="export-menu" ref={wrapRef} data-export-exclude="true">
      <button
        type="button"
        className="btn btn-ghost export-trigger"
        onClick={() => setOpen((o) => !o)}
        disabled={busy}
        aria-haspopup="menu"
        aria-expanded={open}
      >
        {busy ? "Exporting…" : label}
        <span className="export-caret" aria-hidden="true">▾</span>
      </button>
      {open && (
        <div className="export-pop" role="menu">
          {hasPng && (
            <button
              role="menuitem"
              onClick={() =>
                run(() => downloadNodePng(pngRef.current, filename), "Saved PNG")
              }
            >
              <span className="export-ic">▦</span> Download PNG
            </button>
          )}
          {hasPdf && (
            <button
              role="menuitem"
              onClick={() =>
                run(() => downloadNodePdf(pdfRef.current, filename), "Saved PDF")
              }
            >
              <span className="export-ic">▤</span> Download PDF
            </button>
          )}
          {hasCsv && (
            <button
              role="menuitem"
              onClick={() =>
                run(() => {
                  const n = downloadCsv(csvRows, filename);
                  return n;
                }, `Saved CSV (${csvRows.length} rows)`)
              }
            >
              <span className="export-ic">≣</span> Download CSV
            </button>
          )}
          {hasJson && (
            <button
              role="menuitem"
              onClick={() => run(() => downloadJson(jsonData, filename), "Saved JSON")}
            >
              <span className="export-ic">{"{}"}</span> Download JSON
            </button>
          )}
          {hasJson && (
            <button
              role="menuitem"
              onClick={() => run(() => copyJson(jsonData), "Copied JSON to clipboard")}
            >
              <span className="export-ic">⧉</span> Copy JSON
            </button>
          )}
        </div>
      )}
    </div>
  );
}
