import asyncio
import base64
import json
import time
from typing import Optional
import contextlib
import cv2
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from lam.live.live_motion import LiveMotionProvider
from lam.live.debug_draw import draw_face_landmarks, draw_blendshape_debug
from lam.live.lam_live_renderer import LAMLiveRenderer
from lam.live.retargeting.mediapipe_to_flame import MediaPipeToFlameAdapter
from lam.live.source_preprocessor import LAMSourcePreprocessor
from vhap.model import flame


app = FastAPI()


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>LAM Live Tracking Debug</title>
  <style>
    body {
      margin: 0;
      background: #111;
      color: #eee;
      font-family: Arial, sans-serif;
    }

    header {
      padding: 12px 18px;
      background: #1d1d1d;
      border-bottom: 1px solid #333;
    }

    main {
      display: flex;
      gap: 16px;
      padding: 16px;
    }

    #feed {
      width: auto;
      max-width: 100%;
      height: auto;
      background: #000;
      border: 1px solid #333;
    }

    aside {
      width: 360px;
      flex-shrink: 0;
    }

    button {
      padding: 8px 12px;
      margin-right: 8px;
      background: #333;
      color: white;
      border: 1px solid #555;
      cursor: pointer;
    }

    button:hover {
      background: #444;
    }

    pre {
      background: #1b1b1b;
      border: 1px solid #333;
      padding: 12px;
      white-space: pre-wrap;
      min-height: 240px;
      max-height: 75 vh;
      overflow: auto;
      font-family: monospace;
      font-size: 13px;
      line-height: 1.35;
    }

    .row {
      margin-bottom: 12px;
    }

    label {
      display: block;
      margin-bottom: 4px;
      color: #bbb;
    }

    input {
      width: 100%;
      padding: 6px;
      background: #222;
      color: #eee;
      border: 1px solid #444;
    }

    select {
      width: 100%;
      padding: 6px;
      background: #222;
      color: #eee;
      border: 1px solid #444;
    }
    
  </style>
</head>

<body>
  <header>
    <h2>LAM Live MediaPipe Tracking Debug</h2>
  </header>

  <main>
    <section>
      <img id="feed" />
    </section>

    <aside>
      <div class="row">
        <button onclick="connect()">Start</button>
        <button onclick="disconnect()">Stop</button>
      </div>

      <div class="row">
        <label>UI FPS target</label>
        <input id="ui_fps" type="number" value="20" min="1" max="60" />
      </div>

      <div class="row">
        <label>JPEG quality</label>
        <input id="jpeg_quality" type="number" value="85" min="20" max="100" />
      </div>

      <div class="row">
        <label>Draw tessellation</label>
        <input id="draw_tessellation" type="checkbox" />
      </div>

      <div class="row">
        <label>Source input mode</label>
        <select id="source_input_mode">
          <option value="preprocessed">Preprocessed LAM image</option>
          <option value="raw">Raw image</option>
        </select>
      </div>
      
      <div class="row">
        <label>Source image path</label>
        <input
          id="source_image_path"
          type="text"
          value=""
          placeholder="raw image or .../images/00000_00.png"
        />
      </div>

      <div class="row">
        <label>FLAME backend</label>
        <select id="flame_backend">
          <option value="standard">Standard FLAME 100D</option>
          <option value="arkit">ARKit FLAME 52D</option>
        </select>
      </div>
    
      <div class="row">
        <label>Output mode</label>
        <select id="output_mode">
          <option value="debug">MediaPipe Debug</option>
          <option value="lam">LAM Render</option>
        </select>
      </div>

      <pre id="status">Disconnected</pre>
    </aside>
  </main>

  <script>
    let ws = null;
    let stopRequested = false;

    function buildWsUrl() {
      const uiFps = document.getElementById("ui_fps").value || "20";
      const jpegQuality = document.getElementById("jpeg_quality").value || "80";
      const drawTessellation = document.getElementById("draw_tessellation").checked ? "1" : "0";

      const sourceInputMode = encodeURIComponent(document.getElementById("source_input_mode").value || "preprocessed");
      const sourceImagePath = encodeURIComponent(document.getElementById("source_image_path").value || "");
      const outputMode = encodeURIComponent(document.getElementById("output_mode").value || "debug");
      const flameBackend = encodeURIComponent(document.getElementById("flame_backend").value || "standard");
      const protocol = location.protocol === "https:" ? "wss" : "ws";

      
      return `${protocol}://${location.host}/ws/live` +
        `?ui_fps=${uiFps}` +
        `&jpeg_quality=${jpegQuality}` +
        `&draw_tessellation=${drawTessellation}` +
        `&source_input_mode=${sourceInputMode}` +
        `&source_image_path=${sourceImagePath}` +
        `&flame_backend=${flameBackend}` +
        `&output_mode=${outputMode}`;
    }

    function connect() {
      stopRequested = false;
      if (ws !== null) {
        ws.close();
        ws = null;
      }

      const url = buildWsUrl();
      ws = new WebSocket(url);

      ws.onopen = () => {
        document.getElementById("status").textContent = "Connected";
      };

      ws.onmessage = (event) => {
        if (stopRequested) {
            return;
        }
        const data = JSON.parse(event.data);

        if (data.image) {
          document.getElementById("feed").src = "data:image/jpeg;base64," + data.image;
        }

        if (data.status) {
          document.getElementById("status").textContent = data.status;
        }
      };

      ws.onerror = (event) => {
        console.error("WebSocket error", event);
        document.getElementById("status").textContent += "\\nWebSocket error";
      };

      ws.onclose = (event) => {
        console.warn("WebSocket closed", event);
        document.getElementById("status").textContent +=
          `\\nDisconnected code=${event.code} reason=${event.reason}`;
        ws = null;
      };
    }

    function disconnect() {
      stopRequested = true;
    
      document.getElementById("status").textContent += "\\nStopping...";
    
      if (ws !== null) {
        try {
          ws.send(JSON.stringify({ type: "stop" }));
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
    }
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    return HTMLResponse(HTML)


def _fmt_float(v, width=7, precision=3):
    try:
        return f"{float(v):{width}.{precision}f}"
    except Exception:
        return f"{0.0:{width}.{precision}f}"


def _fmt_vec(values, width=7, precision=3):
    return "[" + ", ".join(_fmt_float(v, width, precision) for v in values) + "]"


def _blend(tracking, name):
    if tracking is None or not tracking.detected:
        return 0.0
    return float(tracking.blendshapes.get(name, 0.0))


def build_status(
    ok: bool,
    fps_capture: float,
    fps_sent: float,
    flame_params,
    tracking,
    output_mode: str = "debug",
) -> str:
    lines = []

    lines.append("=== STREAM ===")
    lines.append(f"mode         : {output_mode}")
    lines.append(f"detected     : {str(ok)}")
    lines.append(f"capture_fps  : {_fmt_float(fps_capture)}")
    lines.append(f"sent_fps     : {_fmt_float(fps_sent)}")

    lines.append("")
    lines.append("=== MEDIAPIPE HIGH PRIORITY ===")
    lines.append(f"jawOpen          : {_fmt_float(_blend(tracking, 'jawOpen'))}")
    lines.append(f"jawLeft          : {_fmt_float(_blend(tracking, 'jawLeft'))}")
    lines.append(f"jawRight         : {_fmt_float(_blend(tracking, 'jawRight'))}")
    lines.append(f"mouthSmileLeft   : {_fmt_float(_blend(tracking, 'mouthSmileLeft'))}")
    lines.append(f"mouthSmileRight  : {_fmt_float(_blend(tracking, 'mouthSmileRight'))}")
    lines.append(f"mouthPucker      : {_fmt_float(_blend(tracking, 'mouthPucker'))}")
    lines.append(f"mouthFunnel      : {_fmt_float(_blend(tracking, 'mouthFunnel'))}")
    lines.append(f"eyeBlinkLeft     : {_fmt_float(_blend(tracking, 'eyeBlinkLeft'))}")
    lines.append(f"eyeBlinkRight    : {_fmt_float(_blend(tracking, 'eyeBlinkRight'))}")
    lines.append(f"eyeSquintLeft    : {_fmt_float(_blend(tracking, 'eyeSquintLeft'))}")
    lines.append(f"eyeSquintRight   : {_fmt_float(_blend(tracking, 'eyeSquintRight'))}")
    lines.append(f"browOuterUpLeft  : {_fmt_float(_blend(tracking, 'browOuterUpLeft'))}")
    lines.append(f"browOuterUpRight : {_fmt_float(_blend(tracking, 'browOuterUpRight'))}")
    lines.append(f"browDownLeft     : {_fmt_float(_blend(tracking, 'browDownLeft'))}")
    lines.append(f"browDownRight    : {_fmt_float(_blend(tracking, 'browDownRight'))}")
    lines.append(f"eyeLookUpLeft    : {_fmt_float(_blend(tracking, 'eyeLookUpLeft'))}")
    lines.append(f"eyeLookUpRight   : {_fmt_float(_blend(tracking, 'eyeLookUpRight'))}")
    lines.append(f"eyeLookDownLeft  : {_fmt_float(_blend(tracking, 'eyeLookDownLeft'))}")
    lines.append(f"eyeLookDownRight : {_fmt_float(_blend(tracking, 'eyeLookDownRight'))}")

    if ok and flame_params is not None:
        expr = flame_params["expr"][0, 0].detach().cpu().numpy()
        jaw = flame_params["jaw_pose"][0, 0].detach().cpu().numpy()
        eyes = flame_params["eyes_pose"][0, 0].detach().cpu().numpy()
        rotation = flame_params["rotation"][0, 0].detach().cpu().numpy()
        neck = flame_params["neck_pose"][0, 0].detach().cpu().numpy()
        translation = flame_params["translation"][0, 0].detach().cpu().numpy()

        lines.append("")
        lines.append("=== FLAME PARAMS ===")
        lines.append(f"jaw_pose     : {_fmt_vec(jaw)}")
        lines.append(f"eyes_pose    : {_fmt_vec(eyes)}")
        lines.append(f"rotation     : {_fmt_vec(rotation)}")
        lines.append(f"neck_pose    : {_fmt_vec(neck)}")
        lines.append(f"translation  : {_fmt_vec(translation)}")

        lines.append("")
        lines.append("=== FLAME EXPR FIRST 24 ===")
        for i in range(0, 24, 4):
            chunk = expr[i:i + 4]
            label = f"expr[{i:02d}:{i+4:02d}]"
            lines.append(f"{label}  : {_fmt_vec(chunk)}")
    else:
        lines.append("")
        lines.append("=== FLAME PARAMS ===")
        lines.append("no valid flame params")

    return "\n".join(lines)

    
async def safe_send_json(websocket: WebSocket, payload: dict) -> bool:
    try:
        await websocket.send_text(json.dumps(payload))
        return True
    except Exception as e:
        print("[WebSocket send error]", repr(e))
        return False


async def send_status(websocket: WebSocket, status: str) -> bool:
    print("[STATUS]", status.replace("\\n", " | "))
    return await safe_send_json(websocket, {
        "image": None,
        "status": status,
    })

async def client_requested_stop(websocket: WebSocket) -> bool:
    try:
        msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
    except asyncio.TimeoutError:
        return False
    except WebSocketDisconnect:
        return True
    except Exception:
        return True

    try:
        data = json.loads(msg)
        return data.get("type") == "stop"
    except Exception:
        return False


@app.websocket("/ws/live")
async def live_ws(websocket: WebSocket):
    await websocket.accept()

    query = websocket.query_params
    ui_fps = float(query.get("ui_fps", 20))
    jpeg_quality = int(query.get("jpeg_quality", 80))
    draw_tessellation = query.get("draw_tessellation", "0") == "1"

    source_input_mode = query.get("source_input_mode", "preprocessed")
    if source_input_mode not in ["preprocessed", "raw"]:
        source_input_mode = "preprocessed"
    source_image_path = query.get("source_image_path", "")
    output_mode = query.get("output_mode", "debug")
    flame_backend = query.get("flame_backend", "standard")
    if flame_backend not in ["standard", "arkit"]:
        flame_backend = "standard"

    if flame_backend == "arkit":
        lam_config_path = "configs/inference/live-arkit.yaml"
        expr_dim = 52
    else:
        lam_config_path = "configs/inference/lam-20k-8gpu.yaml"
        expr_dim = 100

    if output_mode not in ["debug", "lam"]:            
    # if output_mode == "lam" and flame_params is not None:
    #     flame_params["jaw_pose"][0, 0, 0] = 1.0
    #     flame_params["rotation"][0, 0, 1] = 0.45
    #     flame_params["neck_pose"][0, 0, 1] = 0.15

        output_mode = "debug"

    ui_fps = max(1.0, min(ui_fps, 60.0))
    jpeg_quality = max(20, min(jpeg_quality, 100))

    frame_interval = 1.0 / ui_fps

    # if output_mode == "lam":
    #     ui_fps = min(ui_fps, 2.0)
    #     frame_interval = 1.0 / ui_fps

    provider: Optional[LiveMotionProvider] = None
    lam_renderer: Optional[LAMLiveRenderer] = None
    
    source_betas = torch.zeros(10)
    
    capture_count = 0
    sent_count = 0
    t_capture = time.time()
    t_sent = time.time()
    fps_capture = 0.0
    fps_sent = 0.0
    last_sent = 0.0

    try:
        provider = LiveMotionProvider(
            device="cpu",
            width=320,
            height=240,
            fps=30,
            expr_dim=expr_dim
        )
        provider.open()

        if output_mode == "lam":
            if not source_image_path:
                await send_status(websocket, "ERROR: source_image_path is required for LAM output mode")
                return

            try:
                processed_source_image_path = source_image_path
                if source_input_mode == "raw":
                    await send_status(websocket, f"LAM init: Preprocessing raw image: {source_image_path}")
                    preprocessor = LAMSourcePreprocessor(
                        output_dir="tracking_output_live",
                        detect_iris_landmarks=True,
                    )

                    processed_source_image_path = await asyncio.to_thread(
                        preprocessor.preprocess, source_image_path
                    )
                    await send_status(websocket, f"LAM init: Preprocessed source image saved to: {processed_source_image_path}")

                
                await send_status(websocket, f"LAM init: Initializing LAM renderer with source image: {source_image_path}")
                
                lam_renderer = LAMLiveRenderer(
                    config_path=lam_config_path,
                    model_name="./model_zoo/lam_models/releases/lam/lam-20k/step_045500/",
                    device="cuda",
                    render_size=512,
                )

                await send_status(websocket, "LAM init: Loading LAM model...")
                await asyncio.to_thread(lam_renderer.load_model)

                await send_status(websocket, "LAM init: Preparing source image...")
                await asyncio.to_thread(lam_renderer.prepare_source_image, processed_source_image_path)

                await send_status(websocket, "LAM init: Preparing default camera...")
                await asyncio.to_thread(lam_renderer.prepare_camer_from_motion_dir, "assets/sample_motion/export/Look_In_My_Eyes/")
        
                source_betas = lam_renderer.get_source_betas_cpu()
                
                class NeutralTracking:
                    detected = True
                    blendshapes = {}
                    facial_matrix = None
                
                neutral_adapter = MediaPipeToFlameAdapter(device="cuda", expr_dim=expr_dim)
                neutral_params = neutral_adapter.to_flame_params(
                    NeutralTracking(),
                    source_betas,
                )

                await send_status(websocket, "LAM init: Building avatar once...")
                await asyncio.to_thread(lam_renderer.build_avatar_once, neutral_params)
        
                ok_send = await send_status(
                    websocket,
                    f"LAM renderer ready\nsource={processed_source_image_path}",
                )
                
                if not ok_send:
                    return
        
            except Exception as e:
                await send_status(
                    websocket,
                    f"ERROR initializing LAM renderer:\n{repr(e)}",
                )
                return


        while True:

            if await client_requested_stop(websocket):
                print("[STREAM] client requested stop, stopping loop")
                break
                
            loop_t0 = time.time()
            render_dt = 0.0
            encode_dt = 0.0
            send_dt = 0.0
            
            frame, flame_params, tracking, ok = await asyncio.to_thread(provider.read, source_betas)
            t_after_tracking = time.time()

            capture_count += 1
            now = time.time()

            if now - t_capture >= 1.0:
                fps_capture = capture_count / (now - t_capture)
                capture_count = 0
                t_capture = now

            if frame is None:
                await asyncio.sleep(0.002)
                continue

            if now - last_sent < frame_interval:
                await asyncio.sleep(0.001)
                continue

            last_sent = now

            if output_mode == "lam" and lam_renderer is not None and ok and flame_params is not None:
                try:
                    render_t0 = time.time()
                    debug = await asyncio.to_thread(
                        lam_renderer.render_frame_cached, flame_params
                    )
                    render_dt = time.time() - render_t0
                    # print(f"[LAM render] render time: {render_dt:.3f}s")
            
                except Exception as e:
                    print("[LAM render error]", repr(e))
            
                    debug = draw_face_landmarks(
                        frame,
                        tracking,
                        draw_tesselation=draw_tessellation,
                        draw_contours=True,
                        draw_irises=True,
                    )
            
                    debug = draw_blendshape_debug(
                        debug,
                        tracking,
                        max_items=8,
                    )
            
            else:
                debug = draw_face_landmarks(
                    frame,
                    tracking,
                    draw_tesselation=draw_tessellation,
                    draw_contours=True,
                    draw_irises=True,
                )
            
                debug = draw_blendshape_debug(
                    debug,
                    tracking,
                    max_items=8,
                )

            status = build_status(
                ok=ok,
                fps_capture=fps_capture,
                fps_sent=fps_sent,
                flame_params=flame_params,
                tracking=tracking,
                output_mode=output_mode
            )

            debug_bgr = cv2.cvtColor(debug, cv2.COLOR_RGB2BGR)

            # success, jpg = cv2.imencode(
            #     ".jpg",
            #     debug_bgr,
            #     [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            # )

            encode_t0 = time.time()
            success, jpg = await asyncio.to_thread(
                cv2.imencode,
                ".jpg",
                debug_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            encode_dt = time.time() - encode_t0

            if not success:
                continue

            payload = {
                "image": base64.b64encode(jpg).decode("ascii"),
                "status": status,
            }

            if await client_requested_stop(websocket):
                print("[STREAM] client requested stop, stopping loop")
                break

            send_t0 = time.time()
            send_ok = await safe_send_json(websocket, payload)
            send_dt = time.time() - send_t0
            
            if not send_ok:
                print("[STREAM] client disconnected, stopping loop")
                break

            total_dt = time.time() - loop_t0
            
            print(
                f"[TIMING] total={total_dt:.3f}s "
                f"tracking={t_after_tracking - loop_t0:.3f}s "
                f"render={render_dt:.3f}s "
                f"encode={encode_dt:.3f}s "
                f"send={send_dt:.3f}s"
            )
               
            sent_count += 1
            now = time.time()
            if now - t_sent >= 1.0:
                fps_sent = sent_count / (now - t_sent)
                sent_count = 0
                t_sent = now

            await asyncio.sleep(0)

    except WebSocketDisconnect:
        pass

    finally:
        if provider is not None:
            provider.close()

        if lam_renderer is not None:
            pass


if __name__ == "__main__":
    uvicorn.run(
        "app_live_web:app",
        host="127.0.0.1",
        port=7861,
        reload=False,
        log_level="info",
        ws_ping_interval=30,
        ws_ping_timeout=120
    )
