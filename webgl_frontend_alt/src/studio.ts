// The existing renderer and tracking controls are reused without changing their implementation.
import { activatePreset, resetPreset, resumeLive } from "./preset-bridge";
import { expressionPresets } from "./expression-presets";

// Install the Studio-only renderer/socket adapter before the shared UI module runs.
await import("../../webgl_frontend/src/main.ts");

const get = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const singleMode = get<HTMLButtonElement>("singleModeBtn");
const multiMode = get<HTMLButtonElement>("multiModeBtn");
const multiEnabled = get<HTMLInputElement>("multiViewEnableInput");
const multiFields = get<HTMLDivElement>("multiFields");
const photoInput = get<HTMLInputElement>("photoInput");
const createBtn = get<HTMLButtonElement>("createBtn");
const cameraBtn = get<HTMLButtonElement>("cameraBtn");
const privacyDialog = get<HTMLDivElement>("privacyDialog");
const privacyConsent = get<HTMLInputElement>("privacyConsentInput");
const continuePrivacy = get<HTMLButtonElement>("continuePrivacyBtn");
const captureDialog = get<HTMLDivElement>("captureDialog");
const loadingDialog = get<HTMLDivElement>("loadingDialog");
const status = get<HTMLPreElement>("status");
const video = get<HTMLVideoElement>("guidedVideo");
const presetsPanel = get<HTMLElement>("presetsPanel");
const resumeLiveBtn = get<HTMLButtonElement>("resumeLiveBtn");

const poses = [
  { name: "Frontale", id: "frontPhotoInput", hint: "Guarda davanti a te e centra il volto nell'ovale.", symbol: "●" },
  { name: "Destra", id: "rightPhotoInput", hint: "Ruota lentamente il viso verso la tua destra. Mantieni il volto nell'ovale.", symbol: "→" },
  { name: "Sinistra", id: "leftPhotoInput", hint: "Ruota lentamente il viso verso la tua sinistra.", symbol: "←" },
  { name: "Alto", id: "upPhotoInput", hint: "Solleva leggermente il mento e guarda verso l'alto.", symbol: "↑" },
  { name: "Basso", id: "downPhotoInput", hint: "Abbassa leggermente il mento e guarda verso il basso.", symbol: "↓" },
] as const;
const messages = [
  "Preparazione delle immagini...",
  "Analisi dei dettagli del volto...",
  "Ricostruzione della geometria dell'avatar...",
  "Ottimizzazione delle espressioni...",
  "Generazione degli asset WebGL...",
  "Preparazione dell'anteprima live...",
];
let stream: MediaStream | null = null;
let index = 0;
let mode: "single" | "multi" = "single";
let loadingTimer: number | undefined;
let messageIndex = 0;

function selectMode(next: "single" | "multi") {
  mode = next;
  multiEnabled.checked = next === "multi";
  multiFields.hidden = next !== "multi";
  singleMode.classList.toggle("selected", next === "single");
  multiMode.classList.toggle("selected", next === "multi");
  if (next === "multi" && !get<HTMLInputElement>("frontPhotoInput").files?.length && photoInput.files?.length) {
    assignFile(get<HTMLInputElement>("frontPhotoInput"), photoInput.files[0]);
  }
}

function assignFile(input: HTMLInputElement, file: File) {
  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  input.dispatchEvent(new Event("change", { bubbles: true }));
}

function updatePose() {
  const pose = poses[index];
  get<HTMLElement>("captureTitle").textContent = mode === "single" ? "Foto frontale" : `Posa ${index + 1}: ${pose.name}`;
  get<HTMLElement>("poseHint").textContent = pose.hint;
  get<HTMLElement>("guideDirection").textContent = pose.symbol;
  get<HTMLElement>("poseCounter").textContent = `${String(index + 1).padStart(2, "0")} / ${mode === "single" ? "01" : "05"}`;
  get<HTMLElement>("poseStrip").innerHTML = (mode === "single" ? poses.slice(0, 1) : poses)
    .map((item, i) => `<span class="${i < index ? "done" : ""}">${item.name}</span>`).join("");
  get<HTMLButtonElement>("retryPoseBtn").hidden = index === 0;
  get<HTMLButtonElement>("snapPoseBtn").textContent = index === 4 || mode === "single" ? "Scatta e genera avatar" : "Scatta e continua";
}

function closeCapture() {
  captureDialog.hidden = true;
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
}

singleMode.onclick = () => selectMode("single");
multiMode.onclick = () => selectMode("multi");
get<HTMLInputElement>("frontPhotoInput").addEventListener("change", event => {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (file) assignFile(photoInput, file);
});
get<HTMLButtonElement>("closeGuideBtn").onclick = closeCapture;
get<HTMLButtonElement>("retryPoseBtn").onclick = () => { index = Math.max(0, index - 1); updatePose(); };
function closePrivacy() {
  privacyDialog.hidden = true;
  privacyConsent.checked = false;
  continuePrivacy.disabled = true;
  get<HTMLButtonElement>("guideStartBtn").focus();
}

get<HTMLButtonElement>("closePrivacyBtn").onclick = closePrivacy;
get<HTMLButtonElement>("cancelPrivacyBtn").onclick = closePrivacy;
privacyConsent.onchange = () => { continuePrivacy.disabled = !privacyConsent.checked; };
get<HTMLButtonElement>("guideStartBtn").onclick = () => {
  privacyConsent.checked = false;
  continuePrivacy.disabled = true;
  privacyDialog.hidden = false;
  privacyConsent.focus();
};
continuePrivacy.onclick = async () => {
  if (!privacyConsent.checked) return;
  closePrivacy();
  if (!navigator.mediaDevices?.getUserMedia) {
    status.textContent = "La webcam richiede localhost o una connessione HTTPS.";
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } } });
    video.srcObject = stream;
    await video.play();
    index = 0;
    updatePose();
    captureDialog.hidden = false;
  } catch (error) {
    closeCapture();
    status.textContent = `Impossibile accedere alla webcam: ${String(error)}`;
  }
};

get<HTMLButtonElement>("snapPoseBtn").onclick = async () => {
  if (!stream || !video.videoWidth) return;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext("2d");
  if (!context) return;
  // The preview alone is mirrored; the uploaded image keeps the camera orientation.
  context.drawImage(video, 0, 0);
  const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, "image/jpeg", .92));
  if (!blob) { status.textContent = "Acquisizione non riuscita. Riprova."; return; }
  const pose = poses[index];
  const file = new File([blob], `lam-${pose.name.toLowerCase()}-${Date.now()}.jpg`, { type: "image/jpeg" });
  if (mode === "single") {
    assignFile(photoInput, file);
  } else {
    assignFile(get<HTMLInputElement>(pose.id), file);
    if (index === 0) assignFile(photoInput, file);
  }
  if (mode === "multi" && index < poses.length - 1) { index++; updatePose(); return; }
  closeCapture();
  createBtn.click();
};

function stopLoading() {
  window.clearInterval(loadingTimer);
  loadingTimer = undefined;
  loadingDialog.hidden = true;
}

function startLoading() {
  if (loadingTimer) return;
  messageIndex = 0;
  get<HTMLElement>("loadingMessage").textContent = messages[0];
  loadingDialog.hidden = false;
  loadingTimer = window.setInterval(() => {
    messageIndex = Math.min(messageIndex + 1, messages.length - 1);
    get<HTMLElement>("loadingMessage").textContent = messages[messageIndex];
  }, 6500);
}

createBtn.addEventListener("click", () => {
  if (createBtn.disabled) return;
  const front = get<HTMLInputElement>("frontPhotoInput").files?.[0] || photoInput.files?.[0];
  if (front) startLoading();
}, { capture: true });
// The old camera control stays available to the shared module, but is not shown in this UI.
cameraBtn.addEventListener("click", startLoading);
function clearPresetSelection() {
  resumeLiveBtn.hidden = true;
  presetsPanel.querySelectorAll<HTMLButtonElement>("[data-preset]").forEach(button => {
    button.classList.remove("selected");
    button.setAttribute("aria-pressed", "false");
  });
}

presetsPanel.querySelectorAll<HTMLButtonElement>("[data-preset]").forEach(button => {
  button.onclick = () => {
    const name = button.dataset.preset as keyof typeof expressionPresets;
    if (!activatePreset(expressionPresets[name])) return;
    presetsPanel.querySelectorAll<HTMLButtonElement>("[data-preset]").forEach(item => {
      item.classList.toggle("selected", item === button);
      item.setAttribute("aria-pressed", String(item === button));
    });
    resumeLiveBtn.hidden = false;
    status.textContent = `${name}: tracking MediaPipe sospeso, espressione applicata all'avatar.`;
  };
  button.setAttribute("aria-pressed", "false");
});
resumeLiveBtn.onclick = () => {
  resumeLive();
  clearPresetSelection();
  status.textContent = "Riconnessione al tracking MediaPipe...";
};
get<HTMLButtonElement>("stopBtn").addEventListener("click", () => {
  presetsPanel.hidden = true;
  clearPresetSelection();
  resetPreset();
}, { capture: true });
new MutationObserver(() => {
  const text = status.textContent || "";
  if (text.includes("Renderer ready")) presetsPanel.hidden = false;
  if (text.startsWith("Loading WebGL avatar") || text.startsWith("Stopped.")) {
    presetsPanel.hidden = true;
    clearPresetSelection();
  }
  if (loadingTimer && (/^ERROR:|^Stopped\.|Renderer ready|WebSocket connected|^WebSocket closed/.test(text) || /Avatar reveal completed/.test(text))) stopLoading();
}).observe(status, { childList: true, characterData: true, subtree: true });
window.addEventListener("beforeunload", closeCapture);
