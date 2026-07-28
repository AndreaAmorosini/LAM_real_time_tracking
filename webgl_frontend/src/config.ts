function numberFromEnv(name: string, fallback: number): number {
  const value = import.meta.env[name];
  const parsed = Number(value);

  if (!Number.isFinite(parsed)) {
    return fallback;
  }

  return parsed;
}

export const WEBGL_CONFIG = {
  wsFps: numberFromEnv("VITE_WEBGL_WS_FPS", 30),
  statusIntervalMs: numberFromEnv("VITE_WEBGL_STATUS_INTERVAL_MS", 250),
  sortEveryNFrames: numberFromEnv("VITE_WEBGL_SORT_EVERY_N_FRAMES", 2),

  revealInMs: numberFromEnv("VITE_WEBGL_REVEAL_IN_MS", 3200),
  revealOutMs: numberFromEnv("VITE_WEBGL_REVEAL_OUT_MS", 2600),
  revealFadeDistance: numberFromEnv("VITE_WEBGL_REVEAL_FADE_DISTANCE", 1.35),

  jawGain: numberFromEnv("VITE_WEBGL_JAW_GAIN", 1.0),
};
