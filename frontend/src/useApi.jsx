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
  return <div className="loading">Loading…</div>;
}

export function ErrorBox({ error }) {
  return (
    <div className="error">
      Could not reach the API ({error}). Is the backend running on{" "}
      <span className="mono">localhost:8000</span>?
    </div>
  );
}
