import os
import re
import shutil
import zipfile
from pathlib import Path
import hashlib

import torch


def safe_avatar_id(name: str) -> str:
    name = os.path.splitext(os.path.basename(name))[0]
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name)
    return name[:64] or "avatar"

OAC_EXPORT_CACHE_VERSION = "oac-v1-lam20k-addteethfalse"


def compute_image_signature(image_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(image_bytes)
    h.update(OAC_EXPORT_CACHE_VERSION.encode("utf-8"))
    return h.hexdigest()


def avatar_id_from_signature(signature: str) -> str:
    return f"avatar_{signature[:16]}"


def oac_zip_path_for_avatar(avatar_id: str, output_root: str = "output/open_avatar_chat") -> str:
    return os.path.join(output_root, avatar_id + ".zip")


def is_valid_oac_zip(zip_path: str, avatar_id: str) -> bool:
    if not os.path.exists(zip_path):
        return False

    required = {
        f"{avatar_id}/",
        f"{avatar_id}/offset.ply",
        f"{avatar_id}/skin.glb",
        f"{avatar_id}/animation.glb",
        f"{avatar_id}/vertex_order.json",
    }

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())
    except Exception:
        return False

    return required.issubset(names)

def zip_directory(source_dir: str, output_zip_path: str):
    source_dir = os.path.abspath(source_dir)
    parent_dir = os.path.dirname(source_dir)
    root_name = os.path.basename(source_dir)

    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Important: gaussian-splat-renderer-for-lam expects an explicit folder entry.
        zf.write(source_dir, root_name + "/")

        for root, dirs, files in os.walk(source_dir):
            for dirname in dirs:
                abs_dir = os.path.join(root, dirname)
                rel_dir = os.path.relpath(abs_dir, parent_dir).replace(os.sep, "/")
                zf.write(abs_dir, rel_dir + "/")

            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, parent_dir).replace(os.sep, "/")
                zf.write(abs_path, rel_path)


def export_oac_zip_from_live_renderer(
    lam_renderer,
    avatar_id: str,
    blender_path: str,
    output_root: str = "output/open_avatar_chat",
) -> str:
    from tools.generateARKITGLBWithBlender import generate_glb

    os.makedirs(output_root, exist_ok=True)

    oac_dir = os.path.join(output_root, avatar_id)
    output_zip_path = os.path.join(output_root, avatar_id + ".zip")

    if os.path.exists(oac_dir):
        shutil.rmtree(oac_dir)
    if os.path.exists(output_zip_path):
        os.remove(output_zip_path)

    os.makedirs(oac_dir, exist_ok=True)

    shape_param = lam_renderer.source_betas
    if shape_param is None:
        raise RuntimeError("LAM source_betas not prepared")

    shape_param = shape_param.detach()
    if shape_param.ndim == 1:
        shape_param = shape_param.unsqueeze(0)

    saved_head_path = lam_renderer.model.renderer.flame_model.save_shaped_mesh(
        shape_param.to("cuda"),
        fd=oac_dir,
    )

    if not lam_renderer.cached_ready:
        raise RuntimeError("LAM cached avatar not built")

    gs = lam_renderer.gs_model_list[0]
    gs.save_ply(
        os.path.join(oac_dir, "offset.ply"),
        rgb2sh=False,
        offset2xyz=True,
    )

    generate_glb(
        input_mesh=Path(saved_head_path),
        template_fbx=Path("./assets/sample_oac/template_file.fbx"),
        output_glb=Path(os.path.join(oac_dir, "skin.glb")),
        blender_exec=Path(blender_path),
    )



    shutil.copy(
        src="./assets/sample_oac/animation.glb",
        dst=os.path.join(oac_dir, "animation.glb"),
    )

    required_files = [
        "offset.ply",
        "skin.glb",
        "animation.glb",
        "vertex_order.json",
    ]
    
    for filename in required_files:
        path = os.path.join(oac_dir, filename)
        if not os.path.exists(path):
            raise RuntimeError(f"OAC export missing required file: {path}")


    if os.path.exists(saved_head_path):
        os.remove(saved_head_path)

    zip_directory(oac_dir, output_zip_path)

    return output_zip_path
