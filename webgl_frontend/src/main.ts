import * as GaussianSplats3D from "gaussian-splat-renderer-for-lam";

type BlendshapeMap = Record<string, number>;

type ExportResponse = {
  avatar_id: string;
  signature?: string;
  cached?: string;
  zip_path: string;
  asset_url: string;
  webgl_url: string;
};

type WebglFrameMessage = {
  type: "webgl_frame";
  detected: boolean;
  blendshapes: BlendshapeMap;
  facial_matrix?: number[][] | null;
  fps_capture?: number;
  fps_sent?: number;
  status?: string;
};


const ARKIT_NAMES = [
  "browDownLeft",
  "browDownRight",
  "browInnerUp",
  "browOuterUpLeft",
  "browOuterUpRight",
  "mouthCheekPuff",
  "cheekPuff",
  "cheekSquintLeft",
  "cheekSquintRight",
  "eyeBlinkLeft",
  "eyeBlinkRight",
  "eyeLookDownLeft",
  "eyeLookDownRight",
  "eyeLookInLeft",
  "eyeLookInRight",
  "eyeLookOutLeft",
  "eyeLookOutRight",
  "eyeLookUpLeft",
  "eyeLookUpRight",
  "eyeSquintLeft",
  "eyeSquintRight",
  "eyeWideLeft",
  "eyeWideRight",
  "jawForward",
  "jawLeft",
  "jawOpen",
  "jawRight",
  "mouthClose",
  "mouthDimpleLeft",
  "mouthDimpleRight",
  "mouthFrownLeft",
  "mouthFrownRight",
  "mouthFunnel",
  "mouthLeft",
  "mouthLowerDownLeft",
  "mouthLowerDownRight",
  "mouthPressLeft",
  "mouthPressRight",
  "mouthPucker",
  "mouthRight",
  "mouthRollLower",
  "mouthRollUpper",
  "mouthShrugLower",
  "mouthShrugUpper",
  "mouthSmileLeft",
  "mouthSmileRight",
  "mouthStretchLeft",
  "mouthStretchRight",
  "mouthUpperUpLeft",
  "mouthUpperUpRight",
  "noseSneerLeft",
  "noseSneerRight",
  "tongueOut",
];

function makeNeutralBlendshapes(): BlendshapeMap {
  const out: BlendshapeMap = {};
  for (const name of ARKIT_NAMES) out[name] = 0.0;
  return out;
}

function clamp01(value: unknown): number {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0.0;
  return Math.max(0.0, Math.min(1.0, n));
}

function normalizeBlendshapes(input: BlendshapeMap | undefined | null): BlendshapeMap {
  const out = makeNeutralBlendshapes();

  const RESPONSIVE_GAIN: Record<string, number> = {
    jawOpen: 1.00,
    mouthSmileLeft: 1.20,
    mouthSmileRight: 1.20,
    eyeBlinkLeft: 1.35,
    eyeBlinkRight: 1.35,
    browInnerUp: 1.25,
    browOuterUpLeft: 1.25,
    browOuterUpRight: 1.25,
    noseSneerLeft: 1.30,
    noseSneerRight: 1.30,
  };

  if (!input) return out;

  for (const [key, value] of Object.entries(input)) {
    out[key] = clamp01(value);
  }

  if ((out["mouthCheekPuff"] ?? 0) === 0 && (out["cheekPuff"] ?? 0) > 0) {
    out["mouthCheekPuff"] = out["cheekPuff"];
  }

  for (const [key, gain] of Object.entries(RESPONSIVE_GAIN)) {
    out[key] = clamp01((out[key] ?? 0) * gain);
  }

  return out;
}

function getRequiredElement<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el as T;
}

const avatarEl = getRequiredElement<HTMLDivElement>("avatar");
const statusEl = getRequiredElement<HTMLPreElement>("status");
const photoInput = getRequiredElement<HTMLInputElement>("photoInput");
const createBtn = getRequiredElement<HTMLButtonElement>("createBtn");
const stopBtn = getRequiredElement<HTMLButtonElement>("stopBtn");
const blenderPathInput = document.getElementById("blenderPathInput") as HTMLInputElement | null;
const mappingModeSelect = getRequiredElement<HTMLSelectElement>("mappingModeSelect");
let lastStatusUiUpdate = 0;
const STATUS_UI_INTERVAL_MS = 250;

mappingModeSelect.onchange = () => {
  const mode = mappingModeSelect.value || "stable";

  appendStatus(`Mapping mode changed to: ${mode}`);

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: "set_mapping_mode",
      mapping_mode: mode,
    }));
  }
};

const debugEnableInput = getRequiredElement<HTMLInputElement>("debugEnableInput");
const debugAdditiveInput = getRequiredElement<HTMLInputElement>("debugAdditiveInput");
const debugBlendshapeSelect = getRequiredElement<HTMLSelectElement>("debugBlendshapeSelect");
const debugBlendshapeValue = getRequiredElement<HTMLInputElement>("debugBlendshapeValue");
const debugBlendshapeValueLabel = getRequiredElement<HTMLSpanElement>("debugBlendshapeValueLabel");
const debugResetBlendshapeBtn = getRequiredElement<HTMLButtonElement>("debugResetBlendshapeBtn");

const debugBoneSelect = getRequiredElement<HTMLSelectElement>("debugBoneSelect");
const debugBoneRotX = getRequiredElement<HTMLInputElement>("debugBoneRotX");
const debugBoneRotY = getRequiredElement<HTMLInputElement>("debugBoneRotY");
const debugBoneRotZ = getRequiredElement<HTMLInputElement>("debugBoneRotZ");
const debugBoneRotXLabel = getRequiredElement<HTMLSpanElement>("debugBoneRotXLabel");
const debugBoneRotYLabel = getRequiredElement<HTMLSpanElement>("debugBoneRotYLabel");
const debugBoneRotZLabel = getRequiredElement<HTMLSpanElement>("debugBoneRotZLabel");
const debugResetBoneBtn = getRequiredElement<HTMLButtonElement>("debugResetBoneBtn");


let latestBlendshapes: BlendshapeMap = makeNeutralBlendshapes();
let latestHeadMatrix: number[][] | null = null;
let trackedBones: Record<string, any> = {};
let neutralBoneRotations: Record<string, any> = {};
let patchedMixer = false;
let latestDetected = false;
let renderer: any = null;
let ws: WebSocket | null = null;

let availableBones: any[] = [];

function initDebugBlendshapeUi() {
  debugBlendshapeSelect.innerHTML = "";

  for (const name of ARKIT_NAMES) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    debugBlendshapeSelect.appendChild(option);
  }

  debugBlendshapeValue.oninput = () => {
    debugBlendshapeValueLabel.textContent = Number(debugBlendshapeValue.value).toFixed(2);
  };

  debugResetBlendshapeBtn.onclick = () => {
    debugBlendshapeValue.value = "0";
    debugBlendshapeValueLabel.textContent = "0.00";
  };
}

function refreshDebugBoneUi() {
  debugBoneSelect.innerHTML = "";
  availableBones = [];

  const bones = renderer?.viewer?.skinModel?.skeleton?.bones || [];

  for (const bone of bones) {
    availableBones.push(bone);

    const option = document.createElement("option");
    option.value = bone.name;
    option.textContent = bone.name;
    debugBoneSelect.appendChild(option);
  }

  // console.log("=== DEBUG BONE UI BONES ===");
  // console.log(availableBones.map((b: any) => b.name).join("\n"));
}

function initDebugBoneUi() {
  const updateLabels = () => {
    debugBoneRotXLabel.textContent = Number(debugBoneRotX.value).toFixed(2);
    debugBoneRotYLabel.textContent = Number(debugBoneRotY.value).toFixed(2);
    debugBoneRotZLabel.textContent = Number(debugBoneRotZ.value).toFixed(2);
    applyDebugBoneRotation();
  };

  debugBoneRotX.oninput = updateLabels;
  debugBoneRotY.oninput = updateLabels;
  debugBoneRotZ.oninput = updateLabels;

  debugBoneSelect.onchange = () => {
    debugBoneRotX.value = "0";
    debugBoneRotY.value = "0";
    debugBoneRotZ.value = "0";
    updateLabels();
  };

  debugResetBoneBtn.onclick = () => {
    debugBoneRotX.value = "0";
    debugBoneRotY.value = "0";
    debugBoneRotZ.value = "0";
    updateLabels();
  };

  updateLabels();
}

function applyDebugBlendshape(input: BlendshapeMap): BlendshapeMap {
  const out = { ...input };

  if (!debugEnableInput.checked) {
    return out;
  }

  const name = debugBlendshapeSelect.value;
  const value = clamp01(Number(debugBlendshapeValue.value));

  if (debugAdditiveInput.checked) {
    out[name] = clamp01((out[name] ?? 0) + value);
  } else {
    for (const key of Object.keys(out)) {
      out[key] = 0.0;
    }
    out[name] = value;
  }

  return out;
}

function applyDebugBoneRotation() {
  if (!renderer) return;

  const boneName = debugBoneSelect.value;
  if (!boneName) return;

  const bone = availableBones.find((b: any) => b.name === boneName);
  if (!bone) return;

  const neutral = neutralBoneRotations[boneName] || bone.rotation.clone();
  neutralBoneRotations[boneName] = neutral;

  const rx = Number(debugBoneRotX.value);
  const ry = Number(debugBoneRotY.value);
  const rz = Number(debugBoneRotZ.value);

  bone.rotation.x = neutral.x + rx;
  bone.rotation.y = neutral.y + ry;
  bone.rotation.z = neutral.z + rz;

  bone.updateMatrixWorld(true);

  const skeleton = renderer?.viewer?.skinModel?.skeleton;
  if (skeleton) {
    skeleton.update();
  }
}

function reconnectWebSocket() {
  if (ws) {
    ws.close(1000, "mapping mode changed");
    ws = null;
  }

  if (renderer) {
    connectWebSocket();
  }
}

function setStatus(text: string) {
  statusEl.textContent = text;
}

function appendStatus(text: string) {
  statusEl.textContent += `\n${text}`;
}

function buildWsUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const mappingMode = encodeURIComponent(mappingModeSelect.value || "stable");
  return `${protocol}://${window.location.host}/ws/live` +
    `?output_mode=webgl` +
    `&ui_fps=30` +
    `&mapping_mode=${mappingMode}`;
}

async function exportAvatarFromPhoto(file: File): Promise<ExportResponse> {
  const form = new FormData();
  form.append("image", file);

  const blenderPath = blenderPathInput?.value?.trim() || "";
  if (blenderPath.length > 0) {
    form.append("blender_path", blenderPath);
  }

  setStatus("Uploading photo and creating avatar...\nThis can take a while.");

  const response = await fetch("/api/oac/export", {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Export failed: HTTP ${response.status}\n${text}`);
  }

  return (await response.json()) as ExportResponse;
}

// -------------------------------------------------------------------
// FPS FIX
// -------------------------------------------------------------------
function throttleSplatSort(sortEveryNFrames = 2) {
  const viewer = renderer?.viewer;
  if (!viewer || typeof viewer.runSplatSort !== "function") return;

  const originalRunSplatSort = viewer.runSplatSort.bind(viewer);
  let frame = 0;

  viewer.runSplatSort = (force = false, forceSortAll = false) => {
    frame += 1;

    if (frame % sortEveryNFrames !== 0) {
      return Promise.resolve(false);
    }

    return originalRunSplatSort(force, forceSortAll);
  };

  console.log(`[PERF] Splat sort throttled: every ${sortEveryNFrames} frames`);
}



// --------------------------------------------------------------------
// POP-IN/OUT Animation
// --------------------------------------------------------------------

type AvatarRevealMode = "in" | "out";

function easeInOutCubic(t: number): number {
  return t < 0.5
    ? 4.0 * t * t * t
    : 1.0 - Math.pow(-2.0 * t + 2.0, 3.0) / 2.0;
}

function easeInOutQuint(t: number): number {
  return t < 0.5
    ? 16.0 * t * t * t * t * t
    : 1.0 - Math.pow(-2.0 * t + 2.0, 5.0) / 2.0;
}


function getSplatMesh(): any | null {
  return renderer?.viewer?.splatMesh ?? null;
}

function getSplatMaxRadius(mesh: any): number {
  return Math.max(
    Number(mesh?.maxSplatDistanceFromSceneCenter ?? 0),
    Number(mesh?.visibleRegionBufferRadius ?? 0),
    Number(mesh?.visibleRegionRadius ?? 0),
    1.0,
  );
}

function setSplatRevealAmount(amount01: number) {
  const mesh = getSplatMesh();
  if (!mesh?.material?.uniforms) return;

  const amount = Math.max(0.0, Math.min(1.0, amount01));
  const maxRadius = getSplatMaxRadius(mesh);

  // Deve combaciare con il valore nello shader:
  // float fadeDistance = 0.75;
  const fadeDistance = 1.35;

  // amount=0 => tutto invisibile
  // amount=1 => tutto visibile
  const fadeStartRadius = -fadeDistance + (maxRadius + fadeDistance) * amount;

  mesh.visibleRegionRadius = fadeStartRadius;
  mesh.visibleRegionFadeStartRadius = fadeStartRadius;
  mesh.visibleRegionChanging = amount < 1.0;

  const uniforms = mesh.material.uniforms;

  if (uniforms.visibleRegionRadius) {
    uniforms.visibleRegionRadius.value = fadeStartRadius;
  }

  if (uniforms.visibleRegionFadeStartRadius) {
    uniforms.visibleRegionFadeStartRadius.value = fadeStartRadius;
  }

  if (uniforms.fadeInComplete) {
    uniforms.fadeInComplete.value = amount >= 1.0 ? 1 : 0;
  }

  if (uniforms.currentTime) {
    uniforms.currentTime.value = performance.now();
  }

  mesh.material.uniformsNeedUpdate = true;
}

function waitForSplatMeshReady(timeoutMs = 5000): Promise<void> {
  const start = performance.now();

  return new Promise((resolve, reject) => {
    const tick = () => {
      const mesh = getSplatMesh();
      const count = Number(mesh?.getSplatCount?.() ?? 0);

      if (mesh && count > 0) {
        resolve();
        return;
      }

      if (performance.now() - start > timeoutMs) {
        reject(new Error("Timed out waiting for splat mesh"));
        return;
      }

      requestAnimationFrame(tick);
    };

    tick();
  });
}

function animateSplatReveal(
  mode: AvatarRevealMode,
  durationMs = 1200,
): Promise<void> {
  return new Promise((resolve) => {
    const start = performance.now();

    const tick = () => {
      const elapsed = performance.now() - start;
      const linear = Math.max(0.0, Math.min(1.0, elapsed / durationMs));
      const eased = easeInOutQuint(linear);

      const amount = mode === "in"
        ? eased
        : 1.0 - eased;

      setSplatRevealAmount(amount);

      if (linear < 1.0) {
        requestAnimationFrame(tick);
      } else {
        setSplatRevealAmount(mode === "in" ? 1.0 : 0.0);
        resolve();
      }
    };

    tick();
  });
}


async function initRenderer(assetPath: string) {
  if (ws) {
    ws.close(1000, "switch avatar");
    ws = null;
  }

  if (renderer && typeof renderer.dispose === "function") {
    renderer.dispose();
    renderer = null;
  }

  latestBlendshapes = makeNeutralBlendshapes();

  setStatus(`Loading WebGL avatar:\n${assetPath}`);

  renderer = await GaussianSplats3D.GaussianSplatRenderer.getInstance(
    avatarEl,
    assetPath,
    {
      backgroundColor: "0x111111",
      alpha: 1.0,

      downloadProgress: (p: number) => {
        setStatus(`Downloading avatar ${(p * 100).toFixed(1)}%`);
      },

      loadProgress: (p: number) => {
        setStatus(`Loading avatar ${(p * 100).toFixed(1)}%`);
      },

      getExpressionData: () => ({ ...latestBlendshapes }),
    }
  );

  await waitForSplatMeshReady();
  
  setSplatRevealAmount(0.0);
  await animateSplatReveal("in", 3200);
  
  appendStatus("Avatar reveal completed");

  appendStatus("Renderer ready");
  setTimeout(() => {
    // debugRendererObjects();
    initHeadPoseBones();
    refreshDebugBoneUi();
    patchMixerForHeadPose();
  }, 1000);
}

function debugRendererObjects() {
  console.log("renderer", renderer);
  console.log("viewer", renderer?.viewer);
  console.log("viewer.avatarMesh", renderer?.viewer?.avatarMesh);
  console.log("viewer.skinModel", renderer?.viewer?.skinModel);
  console.log("viewer.boneRoot", renderer?.viewer?.boneRoot);

  const skinModel = renderer?.viewer?.skinModel;

  if (skinModel?.skeleton?.bones) {
    console.log("SKELETON BONES:", skinModel.skeleton.bones.map((b: any) => b.name));

    for (const bone of skinModel.skeleton.bones) {
      console.log("BONE", bone.name, bone);
    }
  } else {
    console.warn("No skinModel.skeleton.bones found");
  }

  const root = renderer?.viewer?.avatarMesh;
  if (root && typeof root.traverse === "function") {
    root.traverse((obj: any) => {
      console.log("OBJ", obj.type, obj.name, obj);
    });
  }
}

function initHeadPoseBones() {
  trackedBones = {};
  neutralBoneRotations = {};

  const skinModel = renderer?.viewer?.skinModel;
  const bones = skinModel?.skeleton?.bones || [];

  const wanted = [
    "head",
    "neckUpper",
    "neckLower",
    "Chin",
    "Chin_end",
    "lowerJaw",
  ];

  for (const bone of bones) {
    if (wanted.includes(bone.name)) {
      trackedBones[bone.name] = bone;
      neutralBoneRotations[bone.name] = bone.rotation.clone();
      console.log("[HEAD POSE] tracking bone:", bone.name, bone);
    }
  }

  console.log("[HEAD POSE] trackedBones:", Object.keys(trackedBones));
}

function patchMixerForHeadPose() {
  if (patchedMixer) return;
  if (!renderer?.mixer || typeof renderer.mixer.update !== "function") {
    console.warn("[HEAD POSE] renderer.mixer not ready");
    return;
  }

  const originalUpdate = renderer.mixer.update.bind(renderer.mixer);

  renderer.mixer.update = (dt: number) => {
    const result = originalUpdate(dt);

    // Apply after animation mixer, before viewer update.
    applyHeadPose(latestHeadMatrix);

    return result;
  };

  patchedMixer = true;
  console.log("[HEAD POSE] patched AnimationMixer.update");
}



function applyHeadPose(matrix: number[][] | null) {
  if (!matrix || !renderer) return;

  const m = matrix;

  const yaw = Math.atan2(m[0][2], m[2][2]);
  const pitch = Math.atan2(
    -m[1][2],
    Math.sqrt(m[1][0] * m[1][0] + m[1][1] * m[1][1])
  );
  const roll = Math.atan2(m[1][0], m[1][1]);

  const gainYaw = 0.45;
  const gainPitch = 0.45;
  const gainRoll = 0.30;

  const rx = pitch * gainPitch;
  const ry = yaw * gainYaw;
  const rz = -roll * gainRoll;

  const parts = [
    { name: "neckLower", weight: 0.20 },
    { name: "neckUpper", weight: 0.35 },
    { name: "head", weight: 0.45 },
  ];

  for (const part of parts) {
    const bone = trackedBones[part.name];
    const neutral = neutralBoneRotations[part.name];

    if (!bone || !neutral) continue;

    bone.rotation.x = neutral.x + rx * part.weight;
    bone.rotation.y = neutral.y + ry * part.weight;
    bone.rotation.z = neutral.z + rz * part.weight;
    bone.updateMatrixWorld(true);
  }

  const skeleton = renderer?.viewer?.skinModel?.skeleton;
  if (skeleton) {
    skeleton.update();
  }
}


function connectWebSocket() {
  const url = buildWsUrl();

  appendStatus(`Connecting MediaPipe backend:\n${url}`);

  ws = new WebSocket(url);

  ws.onopen = () => {
    appendStatus("MediaPipe WebSocket connected");
  };

  ws.onmessage = (event: MessageEvent<string>) => {
    let msg: WebglFrameMessage | any;

    try {
      msg = JSON.parse(event.data);
    } catch {
      console.warn("Invalid JSON message", event.data);
      return;
    }

    if (msg.type !== "webgl_frame") {
      if (msg.status) setStatus(String(msg.status));
      return;
    }

    latestDetected = Boolean(msg.detected);
    // latestBlendshapes = normalizeBlendshapes(msg.blendshapes);
    latestBlendshapes = applyDebugBlendshape(
      normalizeBlendshapes(msg.blendshapes)
    );
    const derived = msg.derived_blendshapes || {};
    const rel = msg.reliability || {};
    latestHeadMatrix = msg.facial_matrix ?? null;
    applyHeadPose(latestHeadMatrix);
    applyDebugBoneRotation();

    const lines = [
      "=== WEBGL LIVE ===",
      `detected      : ${latestDetected}`,
      `mapping_mode  : ${mappingModeSelect.value}`,
      `fps_capture   : ${(msg.fps_capture ?? 0).toFixed(2)}`,
      `fps_sent      : ${(msg.fps_sent ?? 0).toFixed(2)}`,
      "",
      "=== EXPRESSIONS ===",
      `jawOpen       : ${(latestBlendshapes.jawOpen ?? 0).toFixed(3)}`,
      `smileLeft     : ${(latestBlendshapes.mouthSmileLeft ?? 0).toFixed(3)}`,
      `smileRight    : ${(latestBlendshapes.mouthSmileRight ?? 0).toFixed(3)}`,
      `blinkLeft     : ${(latestBlendshapes.eyeBlinkLeft ?? 0).toFixed(3)}`,
      `blinkRight    : ${(latestBlendshapes.eyeBlinkRight ?? 0).toFixed(3)}`,
      `browInnerUp   : ${(latestBlendshapes.browInnerUp ?? 0).toFixed(3)}`,
      `mouthPucker   : ${(latestBlendshapes.mouthPucker ?? 0).toFixed(3)}`,
      `mouthFunnel   : ${(latestBlendshapes.mouthFunnel ?? 0).toFixed(3)}`,
      "=== BROWS / NOSE / CHEEKS ===",
      `browInnerUp   : ${(latestBlendshapes.browInnerUp ?? 0).toFixed(3)}`,
      `browOuterL    : ${(latestBlendshapes.browOuterUpLeft ?? 0).toFixed(3)}`,
      `browOuterR    : ${(latestBlendshapes.browOuterUpRight ?? 0).toFixed(3)}`,
      `browDownL     : ${(latestBlendshapes.browDownLeft ?? 0).toFixed(3)}`,
      `browDownR     : ${(latestBlendshapes.browDownRight ?? 0).toFixed(3)}`,
      `noseSneerL    : ${(latestBlendshapes.noseSneerLeft ?? 0).toFixed(3)}`,
      `noseSneerR    : ${(latestBlendshapes.noseSneerRight ?? 0).toFixed(3)}`,
      `cheekSquintL  : ${(latestBlendshapes.cheekSquintLeft ?? 0).toFixed(3)}`,
      `cheekSquintR  : ${(latestBlendshapes.cheekSquintRight ?? 0).toFixed(3)}`,
      `cheekPuff     : ${(latestBlendshapes.cheekPuff ?? 0).toFixed(3)}`,
      "=== DERIVED LANDMARKS ===",
      `drv noseSneerL  : ${(derived.noseSneerLeft ?? 0).toFixed(3)}`,
      `drv noseSneerR  : ${(derived.noseSneerRight ?? 0).toFixed(3)}`,
      `drv browInnerUp : ${(derived.browInnerUp ?? 0).toFixed(3)}`,
      `drv browOuterL  : ${(derived.browOuterUpLeft ?? 0).toFixed(3)}`,
      `drv browOuterR  : ${(derived.browOuterUpRight ?? 0).toFixed(3)}`,
      `drv browDownL   : ${(derived.browDownLeft ?? 0).toFixed(3)}`,
      `drv browDownR   : ${(derived.browDownRight ?? 0).toFixed(3)}`,
      "=== RELIABILITY ===",
      `jaw_eye       : ${(rel.jaw_eye ?? 1).toFixed(3)}`,
      `derived       : ${(rel.derived ?? 1).toFixed(3)}`,
      `pose_amount   : ${(rel.pose_amount ?? 0).toFixed(3)}`,
      `motion_amount : ${(rel.motion_amount ?? 0).toFixed(3)}`,
    ];

    const now = performance.now();
    
    if (now - lastStatusUiUpdate > STATUS_UI_INTERVAL_MS) {
      setStatus(lines.join("\n"));
      lastStatusUiUpdate = now;
    }
  };

  ws.onerror = (event) => {
    console.error("WebSocket error", event);
    appendStatus("WebSocket error");
  };

  ws.onclose = (event) => {
    appendStatus(`WebSocket closed code=${event.code} reason=${event.reason}`);
    ws = null;
  };
}

function animateSplatRevealOut(durationMs = 1300): Promise<void> {
  return new Promise((resolve) => {
    const start = performance.now();

    const tick = () => {
      if (!renderer) {
        resolve();
        return;
      }

      const elapsed = performance.now() - start;
      const t = Math.max(0.0, Math.min(1.0, elapsed / durationMs));
      const eased = easeInOutQuint(t);

      setSplatRevealAmount(1.0 - eased);

      if (t < 1.0) {
        requestAnimationFrame(tick);
      } else {
        setSplatRevealAmount(0.0);
        resolve();
      }
    };

    tick();
  });
}

let isStopping = false;

async function stopTrackingAndUnloadAvatar() {
  if (isStopping) return;
  isStopping = true;

  appendStatus("Stopping tracking...");

  if (ws) {
    try {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "stop" }));
      }
    } catch (err) {
      console.warn("Could not send stop message", err);
    }

    try {
      ws.close(1000, "user stop");
    } catch (err) {
      console.warn("Could not close websocket", err);
    }

    ws = null;
  }

  latestBlendshapes = makeNeutralBlendshapes();
  latestHeadMatrix = null;

  if (renderer) {
    appendStatus("Animating avatar unload...");

    try {
      await animateSplatRevealOut(2600);
    } catch (err) {
      console.warn("Avatar unload animation failed", err);
    }
  }

  if (renderer && typeof renderer.dispose === "function") {
    try {
      renderer.dispose();
    } catch (err) {
      console.warn("Renderer dispose failed", err);
    }
  }

  renderer = null;
  trackedBones = {};
  neutralBoneRotations = {};
  patchedMixer = false;
  availableBones = [];

  avatarEl.innerHTML = "";

  setStatus("Stopped. Avatar unloaded.");

  isStopping = false;
}


stopBtn.onclick = () => {
  void stopTrackingAndUnloadAvatar();
};


createBtn.onclick = async () => {
  try {
    const file = photoInput.files?.[0];
    if (!file) {
      setStatus("Select a photo first.");
      return;
    }

    createBtn.disabled = true;

    const result = await exportAvatarFromPhoto(file);

    setStatus(
      [
        result.cached ? "Avatar loaded from cache." : "Avatar export completed.",
        `avatar_id: ${result.avatar_id}`,
        `cached: ${Boolean(result.cached)}`,
        `signature: ${result.signature ?? ""}`,
        `asset_url: ${result.asset_url}`,
        "",
        "Starting WebGL renderer...",
      ].join("\n")
    );

    await initRenderer(result.asset_url);
    throttleSplatSort(2);
    connectWebSocket();
  } catch (err) {
    console.error(err);
    setStatus(`ERROR:\n${String(err)}`);
  } finally {
    createBtn.disabled = false;
  }
};

initDebugBlendshapeUi();
initDebugBoneUi();

window.addEventListener("beforeunload", () => {
  if (ws) ws.close(1000, "page unload");
  if (renderer && typeof renderer.dispose === "function") renderer.dispose();
});
