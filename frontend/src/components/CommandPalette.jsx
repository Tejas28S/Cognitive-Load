import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useTheme } from "../lib/theme";

// ⌘K / Ctrl-K command palette: fuzzy-ish quick nav across every page plus a few
// global actions (theme). Opens on the shortcut or the nav "⌘K" affordance.
const PAGES = [
  { to: "/", label: "Home", hint: "Landing page & project abstract" },
  { to: "/dashboard", label: "Dashboard", hint: "The served binary model at a glance" },
  { to: "/overview", label: "Overview", hint: "Headline result & summary" },
  { to: "/workflow", label: "Workflow", hint: "The research pipeline, stage by stage" },
  { to: "/live", label: "Live Prediction", hint: "Run a held-out window" },
  { to: "/session", label: "Session Trend", hint: "Play back a subject's trials" },
  { to: "/upload", label: "Upload Data", hint: "Predict on a features CSV" },
  { to: "/insights", label: "Model Insights", hint: "Confusion, SHAP, attention" },
  { to: "/evaluation", label: "Evaluation Metrics", hint: "Full metrics for one training run" },
  { to: "/runs", label: "Run History", hint: "Every run in the archive" },
  { to: "/methodology", label: "Methodology", hint: "Why LOSO ≠ fixed split" },
  { to: "/about", label: "About", hint: "Motivation & working principles" },
  { to: "/bios", label: "Team", hint: "Who built this" },
  { to: "/faq", label: "FAQ", hint: "Plain-language answers" },
];

export default function CommandPalette({ open, setOpen }) {
  const navigate = useNavigate();
  const { choice, cycle } = useTheme();
  const [q, setQ] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef(null);

  // Global shortcut: ⌘K / Ctrl-K toggles; "/" opens when not typing in a field.
  useEffect(() => {
    const onKey = (e) => {
      const k = e.key.toLowerCase();
      if ((e.metaKey || e.ctrlKey) && k === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      } else if (
        k === "/" &&
        !open &&
        !/^(input|textarea|select)$/i.test(document.activeElement?.tagName || "")
      ) {
        e.preventDefault();
        setOpen(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, setOpen]);

  useEffect(() => {
    if (open) {
      setQ("");
      setActive(0);
      // Focus after the element mounts.
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  const commands = useMemo(() => {
    const nav = PAGES.map((p) => ({
      id: `go:${p.to}`,
      label: p.label,
      hint: p.hint,
      group: "Go to",
      run: () => navigate(p.to),
    }));
    const actions = [
      {
        id: "theme",
        label: `Theme: ${choice} - cycle`,
        hint: "Light → Dark → System",
        group: "Actions",
        run: cycle,
      },
    ];
    return [...nav, ...actions];
  }, [navigate, choice, cycle]);

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    if (!needle) return commands;
    return commands.filter((c) =>
      `${c.label} ${c.hint}`.toLowerCase().includes(needle)
    );
  }, [q, commands]);

  useEffect(() => {
    setActive((a) => Math.min(a, Math.max(0, filtered.length - 1)));
  }, [filtered.length]);

  if (!open) return null;

  const choose = (cmd) => {
    if (!cmd) return;
    cmd.run();
    setOpen(false);
  };

  const onKeyDown = (e) => {
    if (e.key === "Escape") setOpen(false);
    else if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((a) => Math.min(a + 1, filtered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => Math.max(a - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      choose(filtered[active]);
    }
  };

  let lastGroup = null;

  return (
    <div className="cmdk-overlay" onMouseDown={() => setOpen(false)}>
      <div
        className="cmdk"
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <input
          ref={inputRef}
          className="cmdk-input"
          placeholder="Jump to a page or run an action…"
          value={q}
          onChange={(e) => {
            setQ(e.target.value);
            setActive(0);
          }}
          onKeyDown={onKeyDown}
        />
        <div className="cmdk-list">
          {filtered.length === 0 && <div className="cmdk-empty">No matches</div>}
          {filtered.map((c, i) => {
            const showGroup = c.group !== lastGroup;
            lastGroup = c.group;
            return (
              <div key={c.id}>
                {showGroup && <div className="cmdk-group">{c.group}</div>}
                <button
                  className={`cmdk-item${i === active ? " is-active" : ""}`}
                  onMouseEnter={() => setActive(i)}
                  onClick={() => choose(c)}
                >
                  <span className="cmdk-label">{c.label}</span>
                  <span className="cmdk-hint">{c.hint}</span>
                </button>
              </div>
            );
          })}
        </div>
        <div className="cmdk-foot">
          <span><kbd>↑</kbd><kbd>↓</kbd> navigate</span>
          <span><kbd>↵</kbd> select</span>
          <span><kbd>esc</kbd> close</span>
        </div>
      </div>
    </div>
  );
}
