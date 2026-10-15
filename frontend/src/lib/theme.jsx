import { createContext, useContext, useEffect, useMemo, useState } from "react";

// Theme system: "light" | "dark" | "system". The choice is persisted; "system"
// tracks the OS preference live. The resolved theme is written to
// <html data-theme="…">, which index.css keys its palette overrides off.

const STORAGE_KEY = "cl-theme";
const ThemeContext = createContext(null);

const prefersDark = () =>
  typeof window !== "undefined" &&
  window.matchMedia &&
  window.matchMedia("(prefers-color-scheme: dark)").matches;

const readStored = () => {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return v === "light" || v === "dark" || v === "system" ? v : "system";
  } catch {
    return "system";
  }
};

const resolve = (choice) =>
  choice === "system" ? (prefersDark() ? "dark" : "light") : choice;

export function ThemeProvider({ children }) {
  const [choice, setChoice] = useState(readStored);
  const [resolved, setResolved] = useState(() => resolve(readStored()));

  // Apply resolved theme to <html> and keep it in sync with the choice + OS.
  useEffect(() => {
    const apply = () => {
      const next = resolve(choice);
      setResolved(next);
      document.documentElement.setAttribute("data-theme", next);
      document.documentElement.style.colorScheme = next;
    };
    apply();

    if (choice !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, [choice]);

  const setTheme = (next) => {
    setChoice(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      /* ignore */
    }
  };

  // Cycle light → dark → system for the compact toggle button.
  const cycle = () =>
    setTheme(choice === "light" ? "dark" : choice === "dark" ? "system" : "light");

  const value = useMemo(
    () => ({ choice, resolved, setTheme, cycle }),
    [choice, resolved, setTheme]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used inside <ThemeProvider>");
  return ctx;
}
