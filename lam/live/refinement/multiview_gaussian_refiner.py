import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lam.models.rendering.gaussian_model import GaussianModel
import numpy as np
from PIL import Image

from lam.runners.infer.head_utils import (
    _load_pose,
    load_flame_params,
)

@dataclass
class MultiViewTarget:
    role: str
    image_path: str
    mask_path: Optional[str]
    flame_param_path: str

    c2w: torch.Tensor
    intr: torch.Tensor

    image: torch.Tensor
    mask: torch.Tensor

    flame_params: dict

    weight: float = 1.0


def _clamp01(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return torch.clamp(x, eps, 1.0 - eps)


def _safe_logit(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = _clamp01(x, eps=eps)
    return torch.log(x / (1.0 - x))


def detach_gaussian(gs: GaussianModel) -> GaussianModel:
    """
    GaussianModel is not an nn.Module and has no detach_like().
    This helper returns a detached clone suitable for export.
    """
    return GaussianModel(
        xyz=gs.xyz.detach().clone(),
        opacity=gs.opacity.detach().clone(),
        rotation=gs.rotation.detach().clone(),
        scaling=gs.scaling.detach().clone(),
        shs=gs.shs.detach().clone(),
        offset=gs.offset.detach().clone(),
    )


def _resize_like(x: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x

    if mode == "nearest":
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    return F.interpolate(x, size=ref.shape[-2:], mode=mode, align_corners=False)


def _find_frame_info_for_image(transforms: dict, export_dir: Path, image_path: Path) -> dict:
    frames = transforms.get("frames", [])
    if not frames:
        raise RuntimeError(f"No frames found in transforms.json under {export_dir}")

    try:
        image_rel = image_path.resolve().relative_to(export_dir.resolve()).as_posix()
    except Exception:
        image_rel = image_path.name

    image_rel_clean = image_rel.lstrip("./")
    image_name = image_path.name

    for frame in frames:
        frame_path = str(frame.get("file_path", "")).lstrip("./")
        if frame_path == image_rel_clean:
            return frame

    for frame in frames:
        frame_path = str(frame.get("file_path", "")).lstrip("./")
        if Path(frame_path).name == image_name:
            return frame

    # Prototype fallback: use first frame deterministically.
    return sorted(frames, key=lambda x: x.get("flame_param_path", ""))[0]


def _flame_param_dict_to_batched(flame: dict) -> dict:
    return {
        "expr": flame["expr"].view(1, 1, -1),
        "rotation": flame["rotation"].view(1, 1, 3),
        "neck_pose": flame["neck_pose"].view(1, 1, 3),
        "jaw_pose": flame["jaw_pose"].view(1, 1, 3),
        "eyes_pose": flame["eyes_pose"].view(1, 1, 6),
        "translation": flame["translation"].view(1, 1, 3),
    }

def _pil_resample_bilinear():
    return getattr(Image, "Resampling", Image).BILINEAR


def _pil_resample_nearest():
    return getattr(Image, "Resampling", Image).NEAREST


def _scale_intrinsics_for_resize(
    intr: torch.Tensor,
    old_w: int,
    old_h: int,
    new_w: int,
    new_h: int,
) -> torch.Tensor:
    intr = intr.clone()

    ratio_x = float(new_w) / max(float(old_w), 1.0)
    ratio_y = float(new_h) / max(float(old_h), 1.0)

    # Same convention used elsewhere in LAM:
    # scale full intrinsic row 0 and row 1.
    intr[0] = intr[0] * ratio_x
    intr[1] = intr[1] * ratio_y

    return intr


def _load_processed_image_and_mask_direct(
    image_path: Path,
    mask_path_str: Optional[str],
    intr: torch.Tensor,
    render_size: int = 512,
    bg_color: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Direct loader for images already produced by LAMSourcePreprocessor.

    Important:
    - No recrop.
    - No center_crop_according_to_mask.
    - Only optional resize to render_size.
    - Intrinsics are scaled only if resize happens.
    """

    pil = Image.open(image_path)

    alpha_mask = None
    if pil.mode == "RGBA":
        rgba = np.asarray(pil).astype(np.float32) / 255.0
        rgb_np = rgba[..., :3]
        alpha_mask = rgba[..., 3]
    else:
        rgb_np = np.asarray(pil.convert("RGB")).astype(np.float32) / 255.0

    old_h, old_w = rgb_np.shape[:2]

    if mask_path_str is not None and Path(mask_path_str).exists():
        mask_pil = Image.open(mask_path_str).convert("L")
        mask_np = np.asarray(mask_pil).astype(np.float32) / 255.0
    elif alpha_mask is not None:
        mask_np = alpha_mask.astype(np.float32)
    else:
        mask_np = np.ones((old_h, old_w), dtype=np.float32)

    if mask_np.shape[:2] != rgb_np.shape[:2]:
        mask_pil = Image.fromarray((mask_np * 255.0).clip(0, 255).astype(np.uint8))
        mask_pil = mask_pil.resize((old_w, old_h), _pil_resample_nearest())
        mask_np = np.asarray(mask_pil).astype(np.float32) / 255.0

    mask_np = np.clip(mask_np, 0.0, 1.0)

    # Composite over same white/gray/black background convention as LAM.
    bg = np.ones_like(rgb_np, dtype=np.float32) * float(bg_color)
    rgb_np = rgb_np * mask_np[..., None] + bg * (1.0 - mask_np[..., None])

    new_h = int(render_size)
    new_w = int(render_size)

    if old_h != new_h or old_w != new_w:
        rgb_pil = Image.fromarray((rgb_np * 255.0).clip(0, 255).astype(np.uint8))
        mask_pil = Image.fromarray((mask_np * 255.0).clip(0, 255).astype(np.uint8))

        rgb_pil = rgb_pil.resize((new_w, new_h), _pil_resample_bilinear())
        mask_pil = mask_pil.resize((new_w, new_h), _pil_resample_nearest())

        rgb_np = np.asarray(rgb_pil).astype(np.float32) / 255.0
        mask_np = np.asarray(mask_pil).astype(np.float32) / 255.0
        mask_np = np.clip(mask_np, 0.0, 1.0)

        intr = _scale_intrinsics_for_resize(
            intr=intr,
            old_w=old_w,
            old_h=old_h,
            new_w=new_w,
            new_h=new_h,
        )

    image = torch.from_numpy(rgb_np).float().permute(2, 0, 1).unsqueeze(0)
    mask = torch.from_numpy(mask_np[..., None]).float().permute(2, 0, 1).unsqueeze(0)

    return image, mask, intr



def load_multiview_target_from_processed_image(
    role: str,
    processed_image_path: str,
    render_size: int = 512,
    source_betas: Optional[torch.Tensor] = None,
    bg_color: float = 1.0,
    weight: float = 1.0,
    aspect_standard: float = 1.0,
    enlarge_ratio: list[float] = [1.0, 1.0],
    multiply: int = 14,
) -> MultiViewTarget:
    """
    Loads one already-processed LAM/FLAME export image as a refinement target.

    Expected processed_image_path example:
        tracking_output_live/export/<id>/images/00000_00.png

    Expected nearby files:
        tracking_output_live/export/<id>/fg_masks/00000_00.png
        tracking_output_live/export/<id>/transforms.json
        tracking_output_live/export/<id>/flame_param/...
    """
    image_path = Path(processed_image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Processed image not found: {image_path}")

    # .../<export_id>/images/00000_00.png -> .../<export_id>
    export_dir = image_path.parents[1]

    transforms_path = export_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing transforms.json: {transforms_path}")

    with transforms_path.open("r", encoding="utf-8") as f:
        transforms = json.load(f)

    frame_info = _find_frame_info_for_image(
        transforms=transforms,
        export_dir=export_dir,
        image_path=image_path,
    )

    c2w, intr = _load_pose(frame_info)

    frame_file_path = str(frame_info.get("file_path", ""))
    mask_path = export_dir / frame_file_path.replace("images/", "fg_masks/").replace(".jpg", ".png")
    if not mask_path.exists():
        # Fallback based on current processed image name.
        mask_path = export_dir / "fg_masks" / image_path.name.replace(".jpg", ".png")

    mask_path_str: Optional[str] = str(mask_path) if mask_path.exists() else None

    flame_rel_path = frame_info.get("flame_param_path")
    if not flame_rel_path:
        raise RuntimeError(f"Frame has no flame_param_path in {transforms_path}")

    flame_param_path = export_dir / flame_rel_path
    if not flame_param_path.exists():
        raise FileNotFoundError(f"Missing flame param: {flame_param_path}")

    # image, mask, intr, _ = preprocess_image(
    #     str(image_path),
    #     mask_path=mask_path_str,
    #     intr=intr.clone(),
    #     pad_ratio=0,
    #     bg_color=bg_color,
    #     max_tgt_size=None,
    #     aspect_standard=aspect_standard,
    #     enlarge_ratio=enlarge_ratio,
    #     render_tgt_size=render_size,
    #     multiply=multiply,
    #     need_mask=True,
    #     get_shape_param=False,
    # )

    image, mask, intr = _load_processed_image_and_mask_direct(
        image_path=image_path,
        mask_path_str=mask_path_str,
        intr=intr.clone(),
        render_size=render_size,
        bg_color=bg_color,
    )

    flame = load_flame_params(str(flame_param_path))
    flame_params = _flame_param_dict_to_batched(flame)

    if source_betas is not None:
        flame_params["betas"] = source_betas.detach().cpu().view(1, -1)

    return MultiViewTarget(
        role=role,
        image_path=str(image_path),
        mask_path=mask_path_str,
        flame_param_path=str(flame_param_path),
        c2w=c2w,
        intr=intr,
        image=image,
        mask=mask,
        flame_params=flame_params,
        weight=float(weight),
    )


class OptimizableGaussianAvatar(nn.Module):
    """
    Test-time optimizable wrapper around a LAM Gaussian avatar.

    Important:
    - Keeps Gaussian count fixed.
    - No densify/prune.
    - Maintains OAC compatibility by preserving offset-based topology.
    """

    def __init__(
        self,
        init_gs: GaussianModel,
        query_points: torch.Tensor,
        max_offset_delta: float = 0.025,
        optimize_scale: bool = True,
        optimize_rotation: bool = False,
    ):
        super().__init__()

        if query_points.ndim == 3:
            query_points = query_points[0]

        self.max_offset_delta = float(max_offset_delta)
        self.optimize_scale = bool(optimize_scale)
        self.optimize_rotation = bool(optimize_rotation)

        self.register_buffer("query_points", query_points.detach().clone())
        self.register_buffer("init_offset", init_gs.offset.detach().clone())
        self.register_buffer("init_scaling", init_gs.scaling.detach().clone())
        self.register_buffer("init_rotation", init_gs.rotation.detach().clone())
        self.register_buffer("init_shs", init_gs.shs.detach().clone())
        self.register_buffer("init_opacity", init_gs.opacity.detach().clone())

        self.offset_delta_raw = nn.Parameter(torch.zeros_like(self.init_offset))

        # Current LAM config uses gs_use_rgb=True, so shs is RGB-like [N, 1, 3].
        self.rgb_logit = nn.Parameter(_safe_logit(init_gs.shs.detach().clone()))
        self.opacity_logit = nn.Parameter(_safe_logit(init_gs.opacity.detach().clone()))

        if self.optimize_scale:
            self.log_scaling_delta = nn.Parameter(torch.zeros_like(self.init_scaling))
        else:
            self.register_buffer("log_scaling_delta", torch.zeros_like(self.init_scaling))

        if self.optimize_rotation:
            self.rotation_raw = nn.Parameter(init_gs.rotation.detach().clone())
        else:
            self.register_buffer("rotation_raw", init_gs.rotation.detach().clone())

    def forward(self) -> GaussianModel:
        offset_delta = torch.tanh(self.offset_delta_raw) * self.max_offset_delta
        offset = self.init_offset + offset_delta
        xyz = self.query_points + offset

        shs = torch.sigmoid(self.rgb_logit)
        opacity = torch.sigmoid(self.opacity_logit)

        scaling = self.init_scaling * torch.exp(self.log_scaling_delta)
        scaling = torch.clamp(scaling, min=1e-5, max=0.03)

        rotation = F.normalize(self.rotation_raw, dim=-1)

        return GaussianModel(
            xyz=xyz,
            opacity=opacity,
            rotation=rotation,
            scaling=scaling,
            shs=shs,
            offset=offset,
        )

    def regularization_loss(self) -> dict[str, torch.Tensor]:
        offset_delta = torch.tanh(self.offset_delta_raw) * self.max_offset_delta
    
        current_shs = torch.sigmoid(self.rgb_logit)
        current_opacity = torch.sigmoid(self.opacity_logit)
    
        loss_rgb_prior = ((current_shs - self.init_shs) ** 2).mean()
        loss_opacity_prior = ((current_opacity - self.init_opacity) ** 2).mean()
    
        loss_offset = (offset_delta ** 2).mean()
        loss_scale = (self.log_scaling_delta ** 2).mean()
    
        return {
            "rgb_prior": loss_rgb_prior,
            "opacity": loss_opacity_prior,
            "offset": loss_offset,
            "scale": loss_scale,
        }


class MultiViewGaussianRefiner:
    def __init__(
        self,
        lam_renderer,
        targets: list[MultiViewTarget],
        iters: int = 700,
        lr_rgb: float = 3e-2,
        lr_opacity: float = 1e-2,
        lr_offset: float = 5e-3,
        lr_scale: float = 2e-3,
        lr_rotation: float = 1e-4,
        lambda_mask: float = 0.2,
        lambda_rgb_prior: float = 10.0,
        lambda_offset: float = 5.0,
        lambda_scale: float = 0.1,
        lambda_opacity: float = 0.01,
        max_offset_delta: float = 0.025,
        optimize_scale: bool = True,
        optimize_rotation: bool = False,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ):
        if not targets:
            raise ValueError("MultiViewGaussianRefiner requires at least one target")

        self.lam_renderer = lam_renderer
        self.targets = targets
        self.iters = int(iters)

        self.lr_rgb = float(lr_rgb)
        self.lr_opacity = float(lr_opacity)
        self.lr_offset = float(lr_offset)
        self.lr_scale = float(lr_scale)
        self.lr_rotation = float(lr_rotation)

        self.lambda_mask = float(lambda_mask)
        self.lambda_rgb_prior = float(lambda_rgb_prior)
        self.lambda_offset = float(lambda_offset)
        self.lambda_scale = float(lambda_scale)
        self.lambda_opacity = float(lambda_opacity)

        self.max_offset_delta = float(max_offset_delta)
        self.optimize_scale = bool(optimize_scale)
        self.optimize_rotation = bool(optimize_rotation)

        self.progress_callback = progress_callback

    def _source_betas(self) -> torch.Tensor:
        betas = self.lam_renderer.source_betas
        if betas is None:
            raise RuntimeError("LAM renderer has no source_betas")
        return betas.detach().view(1, -1)

    def _target_flame_params_on_device(self, target: MultiViewTarget, device: torch.device) -> dict:
        flame_params = {}

        for key, value in target.flame_params.items():
            flame_params[key] = value.to(device)

        if "betas" not in flame_params:
            flame_params["betas"] = self._source_betas().to(device)

        return flame_params

    def _freeze_lam_model(self):
        if self.lam_renderer.model is None:
            raise RuntimeError("LAM model is not loaded")

        self.lam_renderer.model.eval()

        for p in self.lam_renderer.model.parameters():
            p.requires_grad_(False)

    def optimize(self) -> GaussianModel:
        if not self.lam_renderer.cached_ready:
            raise RuntimeError("LAM cached avatar must be built before refinement")

        device = torch.device(self.lam_renderer.device)
        renderer = self.lam_renderer.model.renderer

        self._freeze_lam_model()

        init_gs = self.lam_renderer.gs_model_list[0]
        query_points = self.lam_renderer.query_points.to(device)

        opt_avatar = OptimizableGaussianAvatar(
            init_gs=init_gs,
            query_points=query_points,
            max_offset_delta=self.max_offset_delta,
            optimize_scale=self.optimize_scale,
            optimize_rotation=self.optimize_rotation,
        ).to(device)

        param_groups = [
            {"params": [opt_avatar.rgb_logit], "lr": self.lr_rgb},
            {"params": [opt_avatar.opacity_logit], "lr": self.lr_opacity},
            {"params": [opt_avatar.offset_delta_raw], "lr": self.lr_offset},
        ]

        if self.optimize_scale:
            param_groups.append({"params": [opt_avatar.log_scaling_delta], "lr": self.lr_scale})

        if self.optimize_rotation:
            param_groups.append({"params": [opt_avatar.rotation_raw], "lr": self.lr_rotation})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)

        for step in range(self.iters):
            target = self.targets[step % len(self.targets)]
            target_weight = float(getattr(target, "weight", 1.0))

            gt = target.image.to(device)
            gt_mask = target.mask.to(device)

            c2w = target.c2w.to(device).view(1, 1, 4, 4)
            intr = target.intr.to(device).view(1, 1, 4, 4)
            bg = torch.ones(1, 1, 3, device=device)

            flame_params = self._target_flame_params_on_device(target, device)

            height = int(gt.shape[-2])
            width = int(gt.shape[-1])

            gs = opt_avatar()

            render = renderer.forward_animate_gs(
                [gs],
                query_points,
                flame_params,
                c2w,
                intr,
                height,
                width,
                bg,
            )

            pred = render["comp_rgb"][0, 0].unsqueeze(0)       # [1, 3, H, W]
            pred_mask = render["comp_mask"][0, 0].unsqueeze(0) # [1, 1, H, W]

            pred = _resize_like(pred, gt, mode="bilinear")
            pred_mask = _resize_like(pred_mask, gt_mask, mode="bilinear")

            gt_mask = torch.clamp(gt_mask, 0.0, 1.0)

            denom = (gt_mask.sum() * 3.0).clamp_min(1.0)
            
            rgb_l1_raw = ((pred - gt).abs() * gt_mask).sum() / denom
            mask_l1_raw = F.l1_loss(pred_mask, gt_mask)
            
            rgb_l1 = rgb_l1_raw * target_weight
            mask_l1 = mask_l1_raw * target_weight
            
            regs = opt_avatar.regularization_loss()

            loss = (
                rgb_l1
                + self.lambda_mask * mask_l1
                + self.lambda_rgb_prior * regs["rgb_prior"]
                + self.lambda_offset * regs["offset"]
                + self.lambda_scale * regs["scale"]
                + self.lambda_opacity * regs["opacity"]
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % 50 == 0 or step == self.iters - 1:
                info = {
                    "step": step,
                    "iters": self.iters,
                    "role": target.role,
                    "loss": float(loss.detach().cpu()),
                    "target_weight": target_weight,
                    "rgb_l1": float(rgb_l1.detach().cpu()),
                    "mask_l1": float(mask_l1.detach().cpu()),
                    "rgb_l1_raw": float(rgb_l1_raw.detach().cpu()),
                    "mask_l1_raw": float(mask_l1_raw.detach().cpu()),
                    "reg_rgb_prior": float(regs["rgb_prior"].detach().cpu()),
                    "reg_offset": float(regs["offset"].detach().cpu()),
                    "reg_scale": float(regs["scale"].detach().cpu()),
                    "reg_opacity": float(regs["opacity"].detach().cpu()),
                }

                print(
                    "[MULTIVIEW REFINE]",
                    f"step={info['step']}/{info['iters']}",
                    f"role={info['role']}",
                    f"w={info['target_weight']:.2f}",
                    f"loss={info['loss']:.6f}",
                    f"rgb={info['rgb_l1']:.6f}",
                    f"mask={info['mask_l1']:.6f}",
                )

                if self.progress_callback is not None:
                    self.progress_callback(info)

        final_gs = opt_avatar()
        return detach_gaussian(final_gs)


__all__ = [
    "MultiViewTarget",
    "OptimizableGaussianAvatar",
    "MultiViewGaussianRefiner",
    "detach_gaussian",
    "load_multiview_target_from_processed_image",
]
