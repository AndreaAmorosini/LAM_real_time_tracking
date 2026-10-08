import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL(".", import.meta.url));

export default {
  root,
  base: "/",
  server: {
    host: "127.0.0.1",
    port: 5174,
    fs: { allow: [fileURLToPath(new URL("..", import.meta.url))] },
    proxy: {
      "/api": "http://127.0.0.1:7861",
      "/oac_assets": "http://127.0.0.1:7861",
      "/ws": { target: "ws://127.0.0.1:7861", ws: true },
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
};
