import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { getBundles } from "../api/client";

// Which trained model the whole app is talking to: "binary" (Low/High) or
// "3class" (Low/Medium/High). The choice is persisted and resolved against what
// the backend can actually serve, so a stored pick for a bundle whose artifacts
// have gone missing falls back instead of erroring every page.

export type ModelVersionId = "binary" | "3class";

export type Bundle = {
  version: ModelVersionId;
  available: boolean;
  num_classes?: number;
  class_names?: string[];
  /** Why an unavailable bundle can't be served. */
  reason?: string;
  [key: string]: unknown;
};

export type ModelVersionValue = {
  version: ModelVersionId;
  setVersion: (next: ModelVersionId) => void;
  bundles: Bundle[];
  available: Bundle[];
  active: Bundle | null;
  classNames: string[] | null;
  ready: boolean;
  error: string | null;
};

const STORAGE_KEY = "cl-model-version";
const ModelVersionContext = createContext<ModelVersionValue | null>(null);

const readStored = (): ModelVersionId | null => {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return v === "binary" || v === "3class" ? v : null;
  } catch {
    return null;
  }
};

export function ModelVersionProvider({ children }: { children: ReactNode }) {
  const [bundles, setBundles] = useState<Bundle[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [choice, setChoice] = useState<ModelVersionId | null>(readStored);

  useEffect(() => {
    let alive = true;
    getBundles<{ bundles?: Bundle[] }>()
      .then((d) => {
        if (alive) setBundles(d.bundles || []);
      })
      .catch((e: unknown) => {
        if (!alive) return;
        setError(e instanceof Error ? e.message : "Request failed");
      });
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
  const version = useMemo<ModelVersionId>(() => {
    if (bundles === null) return choice || "binary";
    if (choice && available.some((b) => b.version === choice)) return choice;
    return available[0]?.version || choice || "binary";
  }, [choice, available, bundles]);

  const setVersion = (next: ModelVersionId) => {
    setChoice(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      /* ignore */
    }
  };

  const active = (bundles || []).find((b) => b.version === version) || null;

  const value = useMemo<ModelVersionValue>(
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

export function useModelVersion(): ModelVersionValue {
  const ctx = useContext(ModelVersionContext);
  if (!ctx)
    throw new Error("useModelVersion must be used inside <ModelVersionProvider>");
  return ctx;
}

// Human label for a bundle: "2-class · Low / High".
export function bundleLabel(b: Bundle | null | undefined): string {
  if (!b) return "";
  const n = b.num_classes || b.class_names?.length;
  return n ? `${n}-class` : b.version;
}
