import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// Cross-origin isolation lets the CPU fallback use several threads.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

// Server steps talk to the RetiVASC server (python -m retivasc_server serve); in production it serves this UI itself.
const proxy = { '/api': { target: process.env.RETIVASC_SERVER ?? 'http://127.0.0.1:8080' } };

export default defineConfig({
  base: process.env.VITE_BASE_PATH ?? '/',
  plugins: [react()],
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  server: { port: 5178, strictPort: true, headers: isolation, proxy },
  preview: { port: 5179, strictPort: true, headers: isolation, proxy },
});
