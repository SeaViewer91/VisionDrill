import '@visiondrill/ui/src/ui.css';
import './drill.css';
import { ErrorBoundary, installGlobalErrorOverlay } from '@visiondrill/ui';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { DrillProvider } from './store';

installGlobalErrorOverlay();

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <DrillProvider>
        <App />
      </DrillProvider>
    </ErrorBoundary>
  </StrictMode>,
);
