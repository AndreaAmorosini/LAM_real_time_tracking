import { defineConfig } from "vite";

export default defineConfig({
  base: "/webgl/",
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/ws": {
        target: "ws://127.0.0.1:7861",
        ws: true,
      },
      "/oac_assets": {
        target: "http://127.0.0.1:7861",
      },
      "/api": {
        target: "http://127.0.0.1:7861",
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
