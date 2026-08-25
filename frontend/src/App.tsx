import { useState } from "react";
import { NavLink, Route, Routes, Link, useLocation } from "react-router-dom";
import Landing from "./pages/Landing";
import Overview from "./pages/Overview";
import Workflow from "./pages/Workflow";
import Research from "./pages/Research";
import LivePrediction from "./pages/LivePrediction";
import SessionTrend from "./pages/SessionTrend";
import ModelInsights from "./pages/ModelInsights";
import Methodology from "./pages/Methodology";
import UploadData from "./pages/UploadData";
import EvaluationMetrics from "./pages/EvaluationMetrics";
import RunHistory from "./pages/RunHistory";
import About from "./pages/About";
import Bios from "./pages/Bios";
import Faq from "./pages/Faq";
import { BrandMark } from "./components/SignalTrace";
import ThemeToggle from "./components/ThemeToggle";
import ModelSwitch from "./components/ModelSwitch";
import { useModelVersion } from "./lib/modelVersion";
import CommandPalette from "./components/CommandPalette";
import ScrollProgress from "./components/ScrollProgress";
import NavDropdown from "./components/NavDropdown";

// Detect the platform modifier for the ⌘K hint (⌘ on mac, Ctrl elsewhere).
const isMac =
  typeof navigator !== "undefined" && /mac/i.test(navigator.platform || "");

// Reading order of the site, not a decorative grouping: what/why (Overview),
// how it was built (Workflow), how it's validated (Methodology), try it and get
// a report (Assess), the full quantitative record (Evidence), the narrative
// dossier (Research), and the plain-language context (About).
const ASSESS_ITEMS = [
  { to: "/live", label: "Live Prediction", hint: "Run a held-out window, get a report" },
  { to: "/session", label: "Session Trend", hint: "Play back a subject's trials" },
  { to: "/upload", label: "Upload Data", hint: "Predict on a features CSV" },
];
const EVIDENCE_ITEMS = [
  { to: "/insights", label: "Model Insights", hint: "Confusion, SHAP, attention" },
  { to: "/evaluation", label: "Evaluation", hint: "Full metrics for one training run" },
  { to: "/runs", label: "Run Archive", hint: "Every run in the archive" },
];
const ABOUT_ITEMS = [
  { to: "/about", label: "About", hint: "Motivation and working principles" },
  { to: "/bios", label: "Team", hint: "Who built this" },
  { to: "/faq", label: "FAQ", hint: "Plain-language answers" },
];

export default function App() {
  const [paletteOpen, setPaletteOpen] = useState(false);
  const location = useLocation();
  const { active } = useModelVersion();
  const nClasses = active?.num_classes || active?.class_names?.length;

  return (
    <div className="app">
      <nav className="nav">
        <div className="nav-inner">
          <Link to="/" className="brand">
            <BrandMark />
            {/* Styled in index.css, not inline: the narrow-viewport rules need
                to shrink and then drop this wordmark, and an inline font-size
                would win over every media query. */}
            <span className="brand-word">Cognitive&nbsp;Load</span>
          </Link>
          <div className="nav-links">
            <NavLink to="/overview">Overview</NavLink>
            <NavLink to="/workflow">Workflow</NavLink>
            <NavLink to="/methodology">Methodology</NavLink>
            <NavDropdown label="Assess" items={ASSESS_ITEMS} />
            <NavDropdown label="Evidence" items={EVIDENCE_ITEMS} />
            <NavLink to="/research">Research</NavLink>
            <NavDropdown label="About" items={ABOUT_ITEMS} />
          </div>
          <div className="nav-tools">
            <ModelSwitch variant="compact" />
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
          <Route path="/" element={<Landing />} />
          <Route path="/overview" element={<Overview />} />
          <Route path="/workflow" element={<Workflow />} />
          <Route path="/research" element={<Research />} />
          <Route path="/live" element={<LivePrediction />} />
          <Route path="/session" element={<SessionTrend />} />
          <Route path="/upload" element={<UploadData />} />
          <Route path="/insights" element={<ModelInsights />} />
          <Route path="/evaluation" element={<EvaluationMetrics />} />
          <Route path="/runs" element={<RunHistory />} />
          <Route path="/methodology" element={<Methodology />} />
          <Route path="/about" element={<About />} />
          <Route path="/bios" element={<Bios />} />
          <Route path="/faq" element={<Faq />} />
        </Routes>
      </main>

      <footer className="site-footer">
        <div className="foot-inner">
          <span>
            Cognitive load from EEG + fNIRS · TU Berlin / Shin n-back ·{" "}
            {nClasses ? `${nClasses}-class` : "leave-one-subject-out"}
            {nClasses ? ", leave-one-subject-out" : ""}
          </span>
          <span className="mono">
            served model: Hybrid network (EEG + fNIRS fusion) · every value
            traces to a committed artifact
          </span>
        </div>
      </footer>
    </div>
  );
}
