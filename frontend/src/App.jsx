import { NavLink, Route, Routes, Link } from "react-router-dom";
import Overview from "./pages/Overview";
import Research from "./pages/Research";
import LivePrediction from "./pages/LivePrediction";
import SessionTrend from "./pages/SessionTrend";
import ModelInsights from "./pages/ModelInsights";
import Methodology from "./pages/Methodology";
import UploadData from "./pages/UploadData";
import { BrandMark } from "./components/SignalTrace";

// Two clusters: the research narrative, then the live demo. The separator
// encodes that split — reading vs. interacting — rather than decorating.
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
  return (
    <div className="app">
      <nav className="nav">
        <div className="nav-inner">
          <Link to="/" className="brand">
            <BrandMark />
            Cognitive&nbsp;Load
            <span className="brand-tag">Capstone</span>
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
        </div>
      </nav>

      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/research" element={<Research />} />
        <Route path="/live" element={<LivePrediction />} />
        <Route path="/session" element={<SessionTrend />} />
        <Route path="/upload" element={<UploadData />} />
        <Route path="/insights" element={<ModelInsights />} />
        <Route path="/methodology" element={<Methodology />} />
      </Routes>

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
