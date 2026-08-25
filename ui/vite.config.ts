import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  // GitHub Pages serves the demo from /LocalSLM/, so assets need that prefix.
  // The dev server and any local build stay at the root.
  base: process.env.VITE_DEMO === "1" ? "/LocalSLM/" : "/",
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": new URL("./src", import.meta.url).pathname } },
  server: {
    port: 5273,
    strictPort: true,
    // Same-origin in dev so SSE and fetch avoid the CORS dance entirely.
    proxy: { "/api": { target: "http://localhost:8420", changeOrigin: true } },
  },
});
