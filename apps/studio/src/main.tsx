import '@visiondrill/ui/src/ui.css';
import './studio.css';
import { ErrorBoundary, installGlobalErrorOverlay } from '@visiondrill/ui';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { StudioProvider } from './store';

installGlobalErrorOverlay();

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <StudioProvider>
        <App />
      </StudioProvider>
    </ErrorBoundary>
  </StrictMode>,
);
