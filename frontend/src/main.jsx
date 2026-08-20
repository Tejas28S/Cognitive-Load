import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'

// Self-hosted fonts (no external CDN - works offline with the local backend).
// Spectral = scholarly display serif · Inter = UI/body · IBM Plex Mono = data/labels.
import '@fontsource-variable/inter'
import '@fontsource/spectral/400.css'
import '@fontsource/spectral/600.css'
import '@fontsource/spectral/700.css'
import '@fontsource/ibm-plex-mono/400.css'
import '@fontsource/ibm-plex-mono/500.css'

import './index.css'
import App from './App.jsx'
import { ThemeProvider } from './lib/theme'
import { ModelVersionProvider } from './lib/modelVersion'
import { ToastProvider } from './lib/toast'

createRoot(document.getElementById('root')).render(
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
  </StrictMode>,
)
