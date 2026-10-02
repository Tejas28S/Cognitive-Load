import { useModelVersion, bundleLabel } from "../lib/modelVersion";

// Segmented control that picks which trained model the whole app talks to.
// An unavailable bundle stays visible but disabled, with the backend's own
// reason in the tooltip: a missing option the user can see and understand
// beats one that silently disappears.

const ORDER = ["binary", "3class"];

const SUBTITLE = {
  binary: "Low / High",
  "3class": "Low / Medium / High",
};

export default function ModelSwitch({ variant = "compact", showLabel = true }) {
  const { version, setVersion, bundles, ready } = useModelVersion();

  // Nothing to switch between until the backend has described its bundles.
  if (!ready || bundles.length < 2) return null;

  const ordered = [...bundles].sort(
    (a, b) => ORDER.indexOf(a.version) - ORDER.indexOf(b.version)
  );

  return (
    <div className={`model-switch model-switch--${variant}`}>
      {showLabel && <span className="model-switch-label">Model</span>}
      <div className="model-switch-seg" role="group" aria-label="Classification model">
        {ordered.map((b) => {
          const active = b.version === version;
          const label = bundleLabel(b);
          const sub = SUBTITLE[b.version] || b.class_names?.join(" / ");
          return (
            <button
              key={b.version}
              type="button"
              className={`model-switch-opt${active ? " is-active" : ""}${b.available ? "" : " is-disabled"}`}
              onClick={() => b.available && setVersion(b.version)}
              disabled={!b.available}
              aria-pressed={active}
              title={
                b.available
                  ? `Serve the ${label} model (${sub})`
                  : `${label} unavailable, ${b.reason}`
              }
            >
              <span className="model-switch-opt-main">{label}</span>
              {variant === "hero" && (
                <span className="model-switch-opt-sub">
                  {b.available ? sub : "unavailable"}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
