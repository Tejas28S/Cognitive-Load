import { useEffect, useState, type DependencyList } from "react";

export type ApiState<T> = {
  data: T | null;
  loading: boolean;
  error: string | null;
};

/**
 * Minimal async-data hook: runs `fn` on mount (and when `deps` change),
 * exposing { data, loading, error }.
 *
 * The `alive` flag is what makes a fast route change safe: a response that
 * lands after the component unmounted is dropped rather than setting state.
 */
export function useApi<T>(
  fn: () => Promise<T>,
  deps: DependencyList = []
): ApiState<T> {
  const [state, setState] = useState<ApiState<T>>({
    data: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    let alive = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    Promise.resolve()
      .then(fn)
      .then((data) => {
        if (alive) setState({ data, loading: false, error: null });
      })
      .catch((err: unknown) => {
        if (!alive) return;
        const message =
          err instanceof Error ? err.message : "Request failed";
        setState({ data: null, loading: false, error: message });
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}

export function Loading() {
  // Shimmer skeleton - a calmer, more premium load state than a text spinner.
  return (
    <div aria-busy="true" aria-live="polite">
      <span className="skeleton" style={{ display: "block", width: 120, height: 12, borderRadius: 6, marginBottom: 12 }} />
      <span className="skeleton" style={{ display: "block", width: "52%", height: 34, borderRadius: 8, marginBottom: 22 }} />
      <div className="grid grid-2">
        {[0, 1].map((i) => (
          <div className="card" key={i}>
            <span className="skeleton" style={{ display: "block", width: "40%", height: 20, borderRadius: 6, marginBottom: 14 }} />
            <span className="skeleton" style={{ display: "block", width: "100%", height: 200, borderRadius: 10 }} />
          </div>
        ))}
      </div>
    </div>
  );
}

export function ErrorBox({ error }: { error: string | null }) {
  return (
    <div className="error">
      Could not reach the API ({error}). Is the backend running on{" "}
      <span className="mono">localhost:8000</span>?
    </div>
  );
}
