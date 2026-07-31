import os
import asyncio
import base64
import json
import time
from typing import Optional
import cv2
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lam.live.settings import settings
from lam.live.live_motion import LiveMotionProvider
from lam.live.debug_draw import draw_face_landmarks, draw_blendshape_debug
from lam.live.lam_live_renderer import LAMLiveRenderer
from lam.live.retargeting.mediapipe_to_flame import MediaPipeToFlameAdapter
from lam.live.retargeting.landmark_derived_blendshapes import LandmarkDerivedBlendshapes
from lam.live.retargeting.webgl_blendshapes import build_webgl_payload
from lam.live.retargeting.head_pose_reliability import HeadPoseReliability
from lam.live.source_preprocessor import LAMSourcePreprocessor
from lam.live.refinement.multiview_gaussian_refiner import (
    MultiViewGaussianRefiner,
    load_multiview_target_from_processed_image    
)
from lam.live.oac_exporter import (
    export_oac_zip_from_live_renderer,
    compute_image_signature,
    avatar_id_from_signature,
    oac_zip_path_for_avatar,
    is_valid_oac_zip,
)
from lam.live.cleanup import (
    cleanup_paths,
    register_cleanup,
    cleanup_by_id,
    processed_export_dir_from_image,
    oac_avatar_dir_from_zip
)


app = FastAPI()
os.makedirs(settings.oac_output_root, exist_ok=True)
os.makedirs(settings.webgl_dist_dir, exist_ok=True)

app.mount(
    settings.oac_assets_route,
    StaticFiles(directory=settings.oac_output_root),
    name="oac_assets",
)


def head_motion_factor(tracking):
    if tracking is None or tracking.facial_matrix is None:
        return 1.0

    import math
    import numpy as np

    M = np.asarray(tracking.facial_matrix, dtype=float)

    if M.shape != (4, 4):
        return 1.0

    yaw = math.atan2(M[0][2], M[2][2])
    pitch = math.atan2(
        -M[1][2],
        math.sqrt(M[1][0] ** 2 + M[1][1] ** 2),
    )

    amount = abs(yaw) + abs(pitch)

    # ~10 degrees: full derived landmarks
    low = 0.18

    # ~30 degrees: strongly reduced derived landmarks
    high = 0.55

    if amount <= low:
        return 1.0

    if amount >= high:
        return 0.25

    t = (amount - low) / (high - low)
    return 1.0 * (1.0 - t) + 0.25 * t

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

async def receive_client_control(websocket: WebSocket):
    try:
        msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
    except asyncio.TimeoutError:
        return None
    except WebSocketDisconnect:
        return {"type": "disconnect"}
    except Exception:
        return {"type": "disconnect"}

    try:
        return json.loads(msg)
    except Exception:
        return None


@app.post("/api/oac/export")
async def export_oac_from_image(
    image: UploadFile = File(...),
    blender_path: Optional[str] = Form(None),
):
    upload_dir = settings.upload_dir
    oac_output_root = settings.oac_output_root
    
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(oac_output_root, exist_ok=True)
    
    if not blender_path or blender_path.strip() in ["", "/path/to/blender"]:
        blender_path = settings.blender_path
    
    ext = os.path.splitext(image.filename or "")[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
        ext = ".png"
    
    image_bytes = await image.read()
    
    signature = compute_image_signature(image_bytes)
    avatar_id = avatar_id_from_signature(signature)
    zip_path = oac_zip_path_for_avatar(avatar_id, output_root=oac_output_root)
    
    # Cache hit: return existing valid avatar.
    if is_valid_oac_zip(zip_path, avatar_id):
        zip_name = os.path.basename(zip_path)
        return {
            "avatar_id": avatar_id,
            "signature": signature,
            "cached": True,
            "zip_path": zip_path,
            "asset_url": f"{settings.oac_assets_route}/{zip_name}",
            "webgl_url": f"/?asset={settings.oac_assets_route}/{zip_name}",
            "cleanup_on_stop": False,
            "cleanup_id": None
        }
    
    raw_image_path = os.path.join(upload_dir, avatar_id + ext)
    
    with open(raw_image_path, "wb") as f:
        f.write(image_bytes)

    preprocessor = LAMSourcePreprocessor(
        output_dir=settings.tracking_output_dir,
        detect_iris_landmarks=True,
    )

    processed_source_image_path = await asyncio.to_thread(
        preprocessor.preprocess,
        raw_image_path,
    )

    lam_renderer = LAMLiveRenderer(
        config_path=settings.lam_config_path,
        model_name=settings.lam_model_name,
        device=settings.lam_device,
        render_size=settings.lam_render_size,
    )

    await asyncio.to_thread(lam_renderer.load_model)
    await asyncio.to_thread(lam_renderer.prepare_source_image, processed_source_image_path)

    source_betas = lam_renderer.get_source_betas_cpu()

    class NeutralTracking:
        detected = True
        blendshapes = {}
        facial_matrix = None

    neutral_adapter = MediaPipeToFlameAdapter(device="cuda", expr_dim=100)
    neutral_params = neutral_adapter.to_flame_params(
        NeutralTracking(),
        source_betas,
    )

    await asyncio.to_thread(lam_renderer.build_avatar_once, neutral_params)

    zip_path = await asyncio.to_thread(
        export_oac_zip_from_live_renderer,
        lam_renderer,
        avatar_id,
        blender_path,
        settings.oac_output_root
    )

    zip_name = os.path.basename(zip_path)

    single_cleanup_paths = [
        raw_image_path,
        processed_export_dir_from_image(processed_source_image_path),
        oac_avatar_dir_from_zip(zip_path)
    ]

    cleanup_paths(single_cleanup_paths)

    return {
        "avatar_id": avatar_id,
        "signature": signature,
        "cached": False,
        "zip_path": zip_path,
        "asset_url": f"{settings.oac_assets_route}/{zip_name}",
        "webgl_url": f"/?asset={settings.oac_assets_route}/{zip_name}",
        "cleanup_on_stop": False,
        "cleanup_id": None
    }

@app.post("/api/oac/export-multiview")
async def export_oac_from_multiview_images(
    front: UploadFile = File(...),
    right: Optional[UploadFile] = File(None),
    left: Optional[UploadFile] = File(None),
    up: Optional[UploadFile] = File(None),
    down: Optional[UploadFile] = File(None),
    blender_path: Optional[str] = Form(None),
    refine_iters: int = Form(700),
):    
    if not blender_path or blender_path.strip() in ["", "/path/to/blender"]:
        blender_path = settings.blender_path

    uploads = {
        "front": front,
        "right": right,
        "left": left,
        "up": up,
        "down": down,
    }

    uploads = {k: v for k, v in uploads.items() if v is not None}

    if "front" not in uploads:
        raise RuntimeError("front image is required")

    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.oac_output_root, exist_ok=True)

    image_bytes_by_role = {}
    raw_paths = {}

    for role, upload in uploads.items():
        image_bytes = await upload.read()
        image_bytes_by_role[role] = image_bytes

        ext = os.path.splitext(upload.filename or "")[1].lower()
        if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
            ext = ".png"

        # Prototype signature can be simple initially.
        role_path = os.path.join(
            settings.upload_dir,
            f"multiview_{int(time.time())}_{role}{ext}",
        )

        with open(role_path, "wb") as f:
            f.write(image_bytes)

        raw_paths[role] = role_path

    # TODO later: stable multi-image cache signature.
    signature = compute_image_signature(b"".join(image_bytes_by_role[k] for k in sorted(image_bytes_by_role)))
    avatar_id = avatar_id_from_signature("mv_" + signature)

    preprocessor = LAMSourcePreprocessor(
        output_dir=settings.tracking_output_dir,
        detect_iris_landmarks=True,
    )

    processed_paths = {}
    for role, raw_path in raw_paths.items():
        processed_paths[role] = await asyncio.to_thread(
            preprocessor.preprocess,
            raw_path,
        )

    front_processed = processed_paths["front"]

    lam_renderer = LAMLiveRenderer(
        config_path=settings.lam_config_path,
        model_name=settings.lam_model_name,
        device=settings.lam_device,
        render_size=settings.lam_render_size,
    )

    await asyncio.to_thread(lam_renderer.load_model)
    await asyncio.to_thread(lam_renderer.prepare_source_image, front_processed)

    source_betas = lam_renderer.get_source_betas_cpu()

    class NeutralTracking:
        detected = True
        blendshapes = {}
        facial_matrix = None

    neutral_adapter = MediaPipeToFlameAdapter(device=settings.lam_device, expr_dim=100)
    neutral_params = neutral_adapter.to_flame_params(
        NeutralTracking(),
        source_betas,
    )

    await asyncio.to_thread(lam_renderer.build_avatar_once, neutral_params)

    role_weights = {
        "front": settings.multiview_weight_front,
        "left": settings.multiview_weight_left,
        "right": settings.multiview_weight_right,
        "up": settings.multiview_weight_up,
        "down": settings.multiview_weight_down,
    }
    
    targets = []
    for role, processed_path in processed_paths.items():
        target = load_multiview_target_from_processed_image(
            role=role,
            processed_image_path=processed_path,
            render_size=settings.lam_render_size,
            source_betas=source_betas,
            weight=role_weights.get(role, 0.2)
        )

        targets.append(target)

    refiner = MultiViewGaussianRefiner(
        lam_renderer=lam_renderer,
        targets=targets,
        iters=max(
            1,
            min(
                int(refine_iters or settings.multiview_refine_iters),
                settings.multiview_refine_max_iters,
            ),
        ),
    
        lr_rgb=settings.multiview_refine_lr_rgb,
        lr_opacity=settings.multiview_refine_lr_opacity,
        lr_offset=settings.multiview_refine_lr_offset,
        lr_scale=settings.multiview_refine_lr_scale,
        lr_rotation=settings.multiview_refine_lr_rotation,
    
        lambda_mask=settings.multiview_refine_lambda_mask,
        lambda_rgb_prior=settings.multiview_refine_lambda_rgb_prior,
        lambda_offset=settings.multiview_refine_lambda_offset,
        lambda_scale=settings.multiview_refine_lambda_scale,
        lambda_opacity=settings.multiview_refine_lambda_opacity,
    
        max_offset_delta=settings.multiview_refine_max_offset_delta,
        optimize_scale=settings.multiview_refine_optimize_scale,
        optimize_rotation=settings.multiview_refine_optimize_rotation,
    )

    refined_gs = await asyncio.to_thread(refiner.optimize)

    # Replace generated Gaussian avatar with refined one.
    lam_renderer.gs_model_list[0] = refined_gs

    zip_path = await asyncio.to_thread(
        export_oac_zip_from_live_renderer,
        lam_renderer,
        avatar_id,
        blender_path,
        settings.oac_output_root,
    )

    zip_name = os.path.basename(zip_path)

    intermediate_cleanup_paths = []
    intermediate_cleanup_paths.extend(raw_paths.values())

    for processed_path in processed_paths.values():
        intermediate_cleanup_paths.append(
            processed_export_dir_from_image(processed_path)
        )

    intermediate_cleanup_paths.append(oac_avatar_dir_from_zip(zip_path))
    cleanup_paths(intermediate_cleanup_paths)

    cleanup_id = register_cleanup(
        paths=[zip_path],
        kind="multiview_refined_oac_zip",
    )

    return {
        "avatar_id": avatar_id,
        "signature": signature,
        "cached": False,
        "refined": True,
        "num_views": len(targets),
        "zip_path": zip_path,
        "asset_url": f"{settings.oac_assets_route}/{zip_name}",
        "webgl_url": f"/?asset={settings.oac_assets_route}/{zip_name}",
        "cleanup_on_stop": True,
        "cleanup_id": cleanup_id
    }


@app.get("/api/client-config")
async def client_config():
    return {
        "webglWsFps": settings.webgl_ui_fps,
        "mappingMode": settings.mapping_mode,
        "statusIntervalMs": 250,
        "sortEveryNFrames": 2,
        "jawGain": 1.0,
        "revealInMs": 3200,
        "revealOutMs": 2600,
        "revealFadeDistance": 1.35,
    }



@app.websocket("/ws/live")
async def live_ws(websocket: WebSocket):
    await websocket.accept()

    query = websocket.query_params
    ui_fps = float(query.get("ui_fps", settings.default_ui_fps))
    jpeg_quality = int(query.get("jpeg_quality", settings.jpeg_quality))
    draw_tessellation = query.get("draw_tessellation", "0") == "1"
    debug_landmarks = query.get("debug_landmarks", "0") == "1"
    debug_flame = query.get("debug_flame", "0") == "1"

    source_input_mode = query.get("source_input_mode", "preprocessed")
    if source_input_mode not in ["preprocessed", "raw"]:
        source_input_mode = "preprocessed"
    source_image_path = query.get("source_image_path", "")
    output_mode = query.get("output_mode", "debug")
    mapping_mode = query.get("mapping_mode", settings.mapping_mode)
    if mapping_mode not in ["stable", "expressive", "raw"]:
        mapping_mode = "stable"
    flame_backend = query.get("flame_backend", "standard")
    if flame_backend not in ["standard", "arkit"]:
        flame_backend = "standard"

    if flame_backend == "arkit":
        lam_config_path = settings.lam_arkit_config_path
        expr_dim = 52
    else:
        lam_config_path = settings.lam_config_path
        expr_dim = 100

    if output_mode not in ["debug", "lam", "webgl"]:            
        output_mode = "debug"

    ui_fps = max(1.0, min(ui_fps, 60.0))
    jpeg_quality = max(20, min(jpeg_quality, 100))

    frame_interval = 1.0 / ui_fps

    provider: Optional[LiveMotionProvider] = None
    lam_renderer: Optional[LAMLiveRenderer] = None
    head_reliability = HeadPoseReliability()
    
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
            width=settings.capture_width,
            height=settings.capture_height,
            fps=settings.capture_fps,
            expr_dim=expr_dim,
            # smoothing_alpha=0.20
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
                await asyncio.to_thread(lam_renderer.prepare_camer_from_motion_dir, settings.camera_motion_dir)
        
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

        landmark_derived = LandmarkDerivedBlendshapes(neutral_frames=settings.derived_neutral_frames, alpha=settings.derived_alpha)
        
        while True:

            control = await receive_client_control(websocket)
            
            if control:
                if control.get("type") in ["stop", "disconnect"]:
                    print("[STREAM] client requested stop/disconnect, stopping loop")
                    break
            
                if control.get("type") == "set_mapping_mode":
                    new_mode = control.get("mapping_mode", "stable")
                    if new_mode in ["raw", "stable", "expressive"]:
                        mapping_mode = new_mode
                        print("[STREAM] mapping_mode changed to:", mapping_mode)

                if control.get("type") == "set_debug_options":
                    debug_landmarks = bool(control.get("debug_landmarks", debug_landmarks))
                    debug_flame = bool(control.get("debug_flame", debug_flame))
                    print(
                        "[STREAM] debug options:",
                        "debug_landmarks=", debug_landmarks,
                        "debug_flame=", debug_flame,
                    )
                
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

            if output_mode == "webgl":
                status = build_status(
                        ok=ok,
                        fps_capture=fps_capture,
                        fps_sent=fps_sent,
                        flame_params=flame_params,
                        tracking=tracking,
                        output_mode=output_mode
                    )
                
                payload = build_webgl_payload(
                    tracking=tracking,
                    status=status,
                    fps_capture=fps_capture,
                    fps_sent=fps_sent,
                    mapping_mode=mapping_mode,
                    landmark_derived=landmark_derived,
                    head_reliability=head_reliability
                )
            
                if debug_flame:
                    payload["flame_debug"] = status
            
                if debug_landmarks:
                    debug_frame = draw_face_landmarks(
                        frame,
                        tracking,
                        draw_tesselation=True,
                        draw_contours=True,
                        draw_irises=True,
                    )
            
                    debug_frame = draw_blendshape_debug(
                        debug_frame,
                        tracking,
                        max_items=10,
                    )
            
                    debug_bgr = cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)
            
                    success, jpg = await asyncio.to_thread(
                        cv2.imencode,
                        ".jpg",
                        debug_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 70],
                    )
            
                    if success:
                        payload["debug_image"] = base64.b64encode(jpg).decode("ascii")
    

                send_ok = await safe_send_json(websocket, payload)
                if not send_ok:
                    print("[STREAM] client disconnected, stopping loop")
                    break

                sent_count += 1
                now = time.time()
                if now - t_sent >= 1.0:
                    fps_sent = sent_count / (now - t_sent)
                    sent_count = 0
                    t_sent = now

                await asyncio.sleep(0)
                continue

            if output_mode == "lam" and lam_renderer is not None and ok and flame_params is not None:
                try:
                    render_t0 = time.time()
                    debug = await asyncio.to_thread(
                        lam_renderer.render_frame_cached, flame_params
                    )
                    render_dt = time.time() - render_t0
            
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

            control = await receive_client_control(websocket)
            
            if control:
                if control.get("type") in ["stop", "disconnect"]:
                    print("[STREAM] client requested stop/disconnect, stopping loop")
                    break
            
                if control.get("type") == "set_mapping_mode":
                    new_mode = control.get("mapping_mode", "stable")
                    if new_mode in ["raw", "stable", "expressive"]:
                        mapping_mode = new_mode
                        print("[STREAM] mapping_mode changed to:", mapping_mode)

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

    except asyncio.CancelledError:
        print("[STREAM] websocket cancelled, stopping loop")
        raise
    except WebSocketDisconnect:
        pass

    finally:
        if provider is not None:
            try:
                provider.close()
            except Exception as e:
                print("[STREAM] error closing provider:", repr(e))

        if lam_renderer is not None:
            pass

class CleanupRequest(BaseModel):
    cleanup_id: str

@app.post("/api/cleanup/export")
async def cleanup_export(req: CleanupRequest):
    return cleanup_by_id(req.cleanup_id)

app.mount(
    "/",
    StaticFiles(directory=settings.webgl_dist_dir, html=True),
    name="webgl_root"
)


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
