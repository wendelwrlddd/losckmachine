import { defineConfig } from 'vite';

export default defineConfig({
  // Base config
  server: {
    open: true
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets'
  }
});
