import * as GaussianSplats3D from "../../webgl_frontend/node_modules/gaussian-splat-renderer-for-lam/build/gaussian-splat-renderer-for-lam.module.js";

type Blendshapes = Record<string, number>;
type Renderer = { getExpressionData?: () => Blendshapes };

const NativeWebSocket = window.WebSocket;
let liveSocket: ManagedLiveSocket | null = null;
let renderer: Renderer | null = null;
let originalExpression: (() => Blendshapes) | null = null;
let preset: Blendshapes | null = null;
let from: Blendshapes = {};
let transitionStart = 0;

class ManagedLiveSocket {
  private socket: WebSocket | null = null;
  private paused = false;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;

  constructor(readonly url: string, private readonly protocols?: string | string[]) {
    this.connect();
  }

  get readyState() { return this.socket?.readyState ?? NativeWebSocket.CLOSED; }

  private connect() {
    const socket = new NativeWebSocket(this.url, this.protocols);
    this.socket = socket;
    socket.onopen = event => { if (this.socket === socket) this.onopen?.(event); };
    socket.onmessage = event => { if (this.socket === socket && !this.paused) this.onmessage?.(event); };
    socket.onerror = event => { if (this.socket === socket) this.onerror?.(event); };
    socket.onclose = event => {
      if (this.socket === socket) {
        this.socket = null;
        this.onclose?.(event);
      }
    };
  }

  send(data: string) { this.socket?.send(data); }

  pause() {
    if (this.paused) return;
    this.paused = true;
    const socket = this.socket;
    this.socket = null;
    socket?.close(1000, "preset attivo");
  }

  resume() {
    if (!this.paused) return;
    this.paused = false;
    this.connect();
  }

  close(code?: number, reason?: string) {
    this.paused = false;
    const socket = this.socket;
    this.socket = null;
    socket?.close(code, reason);
    if (liveSocket === this) liveSocket = null;
  }
}

// Only /ws/live is managed; other WebSocket connections remain native.
window.WebSocket = new Proxy(NativeWebSocket, {
  construct(target, args: [string | URL, (string | string[])?]) {
    const [url, protocols] = args;
    if (new URL(String(url), location.href).pathname !== "/ws/live") {
      return Reflect.construct(target, args);
    }
    liveSocket = new ManagedLiveSocket(String(url), protocols);
    return liveSocket;
  },
}) as typeof WebSocket;

const rendererClass = GaussianSplats3D.GaussianSplatRenderer as any;
const loadRenderer = rendererClass.getInstance.bind(rendererClass);
rendererClass.getInstance = async (...args: any[]) => {
  const instance = await loadRenderer(...args) as Renderer | undefined;
  if (!instance) return instance;
  renderer = instance;
  preset = null;
  originalExpression = instance.getExpressionData?.bind(instance) ?? null;
  instance.getExpressionData = () => {
    const tracked = originalExpression?.() ?? {};
    if (!preset) return tracked;
    const progress = Math.min(1, (performance.now() - transitionStart) / 350);
    const eased = progress * progress * (3 - 2 * progress);
    return Object.fromEntries(Object.keys(preset).map(key => [
      key, (from[key] ?? 0) * (1 - eased) + preset![key] * eased,
    ]));
  };
  return instance;
};

export function activatePreset(values: Blendshapes): boolean {
  if (!renderer || !originalExpression) return false;
  from = renderer.getExpressionData?.() ?? {};
  const neutral = Object.fromEntries(Object.keys(originalExpression()).map(key => [key, 0]));
  preset = { ...neutral, ...values };
  transitionStart = performance.now();
  liveSocket?.pause();
  return true;
}

export function resumeLive() {
  preset = null;
  liveSocket?.resume();
}

export function resetPreset() {
  preset = null;
  renderer = null;
  originalExpression = null;
}
