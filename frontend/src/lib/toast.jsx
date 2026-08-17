import { createContext, useCallback, useContext, useRef, useState } from "react";

// Lightweight toast system. useToast() returns a `toast` fn plus helpers
// (toast.success / .error / .info). Toasts auto-dismiss and stack bottom-right.

const ToastContext = createContext(null);
let seq = 0;

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);
  const timers = useRef(new Map());

  const dismiss = useCallback((id) => {
    setToasts((t) => t.filter((x) => x.id !== id));
    const timer = timers.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timers.current.delete(id);
    }
  }, []);

  const push = useCallback(
    (message, { type = "info", duration = 3200 } = {}) => {
      const id = ++seq;
      setToasts((t) => [...t, { id, message, type }]);
      if (duration > 0) {
        timers.current.set(
          id,
          setTimeout(() => dismiss(id), duration)
        );
      }
      return id;
    },
    [dismiss]
  );

  // `toast(msg)` plus typed shorthands.
  const toast = useCallback(
    (message, opts) => push(message, opts),
    [push]
  );
  toast.success = (m, o) => push(m, { ...o, type: "success" });
  toast.error = (m, o) => push(m, { ...o, type: "error", duration: 5000 });
  toast.info = (m, o) => push(m, { ...o, type: "info" });

  return (
    <ToastContext.Provider value={toast}>
      {children}
      <div className="toast-stack" role="status" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className={`toast toast--${t.type}`}>
            <span className="toast-dot" aria-hidden="true" />
            <span className="toast-msg">{t.message}</span>
            <button
              type="button"
              className="toast-x"
              onClick={() => dismiss(t.id)}
              aria-label="Dismiss"
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used inside <ToastProvider>");
  return ctx;
}
