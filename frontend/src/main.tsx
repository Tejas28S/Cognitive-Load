import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";

// Self-hosted fonts (no external CDN - works offline with the local backend).
// Spectral = scholarly display serif · Inter = UI/body · IBM Plex Mono = data/labels
// · Fraunces = card headings.
import "@fontsource-variable/inter";
import "@fontsource/spectral/400.css";
import "@fontsource/spectral/600.css";
import "@fontsource/spectral/700.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
// Fraunces = the card voice. A variable display serif with SOFT and WONK axes,
// so it carries personality without leaving the editorial register the way a
// script or handwriting face would.
import "@fontsource-variable/fraunces";

// app.css is the Tailwind entry; it imports index.css into the `legacy` layer.
import "./styles/app.css";
import App from "./App";
import { ThemeProvider } from "./lib/theme";
import { ModelVersionProvider } from "./lib/modelVersion";
import { ToastProvider } from "./lib/toast";

const rootEl = document.getElementById("root");
if (!rootEl) throw new Error("#root not found");

createRoot(rootEl).render(
  <StrictMode>
    <ThemeProvider>
      <ToastProvider>
        <ModelVersionProvider>
          <BrowserRouter>
            <App />
          </BrowserRouter>
        </ModelVersionProvider>
      </ToastProvider>
    </ThemeProvider>
  </StrictMode>
);
