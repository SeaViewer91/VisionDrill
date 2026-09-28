import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// Tauri 개발 서버 규약: 고정 포트, 화면 지우지 않음
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: { port: 1421, strictPort: true },
  envPrefix: ['VITE_', 'TAURI_'],
  build: { target: 'es2022', chunkSizeWarningLimit: 2000 },
  // sql.js는 CommonJS라 개발 서버에서 미리 번들해야 워크스페이스 패키지에서 import 가능
  optimizeDeps: { include: ['sql.js'] },
});
