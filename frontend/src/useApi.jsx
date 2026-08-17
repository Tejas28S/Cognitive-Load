import { useEffect, useState } from "react";

// Minimal async-data hook: runs `fn` on mount (and when deps change),
// exposing { data, loading, error }.
export function useApi(fn, deps = []) {
  const [state, setState] = useState({ data: null, loading: true, error: null });
  useEffect(() => {
    let alive = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    Promise.resolve()
      .then(fn)
      .then((data) => alive && setState({ data, loading: false, error: null }))
      .catch(
        (err) =>
          alive &&
          setState({
            data: null,
            loading: false,
            error: err?.message || "Request failed",
          })
      );
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

export function ErrorBox({ error }) {
  return (
    <div className="error">
      Could not reach the API ({error}). Is the backend running on{" "}
      <span className="mono">localhost:8000</span>?
    </div>
  );
}
