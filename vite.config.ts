import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

const BACKEND_TARGET = 'http://127.0.0.1:8000';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  optimizeDeps: {
    exclude: ['lucide-react'],
  },
  server: {
    proxy: {
      '/status': { target: BACKEND_TARGET, changeOrigin: true },
      '/persons': { target: BACKEND_TARGET, changeOrigin: true },
      '/events': { target: BACKEND_TARGET, changeOrigin: true },
      '/evidence': { target: BACKEND_TARGET, changeOrigin: true },
      '/ws': { target: BACKEND_TARGET, changeOrigin: true, ws: true },
    },
  },
});
