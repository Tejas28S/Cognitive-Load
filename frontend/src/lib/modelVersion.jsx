import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { getBundles } from "../api/client";

// Which trained model the whole app is talking to: "binary" (Low/High) or
// "3class" (Low/Medium/High). The choice is persisted and resolved against what
// the backend can actually serve, so a stored pick for a bundle whose artifacts
// have gone missing falls back instead of erroring every page.

const STORAGE_KEY = "cl-model-version";
const ModelVersionContext = createContext(null);

const readStored = () => {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return v === "binary" || v === "3class" ? v : null;
  } catch {
    return null;
  }
};

export function ModelVersionProvider({ children }) {
  const [bundles, setBundles] = useState(null);
  const [error, setError] = useState(null);
  const [choice, setChoice] = useState(readStored);

  useEffect(() => {
    let alive = true;
    getBundles()
      .then((d) => alive && setBundles(d.bundles || []))
      .catch((e) => alive && setError(e?.message || "Request failed"));
    return () => {
      alive = false;
    };
  }, []);

  const available = useMemo(
    () => (bundles || []).filter((b) => b.available),
    [bundles]
  );

  // Before /api/bundles answers, assume the stored choice so the first render
  // does not fetch under one version and then immediately refetch under another.
  const version = useMemo(() => {
    if (bundles === null) return choice || "binary";
    if (choice && available.some((b) => b.version === choice)) return choice;
    return available[0]?.version || choice || "binary";
  }, [choice, available, bundles]);

  const setVersion = (next) => {
    setChoice(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      /* ignore */
    }
  };

  const active = (bundles || []).find((b) => b.version === version) || null;

  const value = useMemo(
    () => ({
      version,
      setVersion,
      bundles: bundles || [],
      available,
      active,
      // Class names come from the served model_config, never from a constant —
      // binary is [Low, High], 3-class is [Low, Medium, High].
      classNames: active?.class_names?.length ? active.class_names : null,
      ready: bundles !== null || error !== null,
      error,
    }),
    [version, bundles, available, active, error]
  );

  return (
    <ModelVersionContext.Provider value={value}>
      {children}
    </ModelVersionContext.Provider>
  );
}

export function useModelVersion() {
  const ctx = useContext(ModelVersionContext);
  if (!ctx)
    throw new Error("useModelVersion must be used inside <ModelVersionProvider>");
  return ctx;
}

// Human label for a bundle: "2-class · Low / High".
export function bundleLabel(b) {
  if (!b) return "";
  const n = b.num_classes || b.class_names?.length;
  return n ? `${n}-class` : b.version;
}
