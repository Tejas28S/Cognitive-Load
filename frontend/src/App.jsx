import { useState } from "react";
import { NavLink, Route, Routes, Link, useLocation } from "react-router-dom";
import Overview from "./pages/Overview";
import Research from "./pages/Research";
import LivePrediction from "./pages/LivePrediction";
import SessionTrend from "./pages/SessionTrend";
import ModelInsights from "./pages/ModelInsights";
import Methodology from "./pages/Methodology";
import UploadData from "./pages/UploadData";
import { BrandMark } from "./components/SignalTrace";
import ThemeToggle from "./components/ThemeToggle";
import CommandPalette from "./components/CommandPalette";
import ScrollProgress from "./components/ScrollProgress";

// Detect the platform modifier for the ⌘K hint (⌘ on mac, Ctrl elsewhere).
const isMac =
  typeof navigator !== "undefined" && /mac/i.test(navigator.platform || "");

// Two clusters: the research narrative, then the live demo. The separator
// encodes that split - reading vs. interacting - rather than decorating.
const groups = [
  [
    { to: "/", label: "Overview", end: true },
    { to: "/research", label: "Research" },
  ],
  [
    { to: "/live", label: "Live" },
    { to: "/session", label: "Session" },
    { to: "/upload", label: "Upload" },
  ],
  [
    { to: "/insights", label: "Insights" },
    { to: "/methodology", label: "Methodology" },
  ],
];

export default function App() {
  const [paletteOpen, setPaletteOpen] = useState(false);
  const location = useLocation();
  return (
    <div className="app">
      <nav className="nav">
        <div className="nav-inner">
          <Link to="/" className="brand">
            <BrandMark />
            <span style={{ fontSize: "1.35rem", textTransform: "uppercase", letterSpacing: "0.02em" }}>Cognitive&nbsp;Load</span>
          </Link>
          <div className="nav-links">
            {groups.map((group, gi) => (
              <span key={gi} style={{ display: "contents" }}>
                {gi > 0 && <span className="nav-sep" aria-hidden="true" />}
                {group.map((l) => (
                  <NavLink key={l.to} to={l.to} end={l.end}>
                    {l.label}
                  </NavLink>
                ))}
              </span>
            ))}
          </div>
          <div className="nav-tools">
            <button
              type="button"
              className="cmdk-hint-btn"
              onClick={() => setPaletteOpen(true)}
              title="Command palette"
              aria-label="Open command palette"
            >
              <span className="cmdk-hint-ic" aria-hidden="true">⌘</span>
              <kbd>{isMac ? "⌘" : "Ctrl"} K</kbd>
            </button>
            <ThemeToggle />
          </div>
        </div>
        <ScrollProgress />
      </nav>

      <CommandPalette open={paletteOpen} setOpen={setPaletteOpen} />

      <main key={location.pathname} className="route-view">
        <Routes location={location}>
          <Route path="/" element={<Overview />} />
          <Route path="/research" element={<Research />} />
          <Route path="/live" element={<LivePrediction />} />
          <Route path="/session" element={<SessionTrend />} />
          <Route path="/upload" element={<UploadData />} />
          <Route path="/insights" element={<ModelInsights />} />
          <Route path="/methodology" element={<Methodology />} />
        </Routes>
      </main>

      <footer className="site-footer">
        <div className="foot-inner">
          <span>
            Cognitive load from EEG + fNIRS · TU Berlin / Shin n-back · 3-class,
            leave-one-subject-out
          </span>
          <span className="mono">
            served model: hybrid_rf_ensemble · every value traces to a committed
            artifact
          </span>
        </div>
      </footer>
    </div>
  );
}
