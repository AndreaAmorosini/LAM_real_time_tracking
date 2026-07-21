import os
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from safetensors.torch import load_file
import math
from einops import rearrange
import json

from lam.models import ModelLAM
from lam.runners.infer.head_utils import preprocess_image, prepare_motion_seqs


class LAMLiveRenderer:
    def __init__(
        self,
        config_path="configs/inference/lam-20k-8gpu.yaml",
        model_name="./model_zoo/lam_models/releases/lam/lam-20k/step_045500/",
        device="cuda",
        dtype=torch.float32,
        render_size=512,
    ):
        self.config_path = config_path
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.render_size = render_size

        self.cfg = OmegaConf.load(config_path)
        self.model = None

        self.source_image = None
        self.source_betas = None

        self.render_c2ws = None
        self.render_intrs = None
        self.render_bg_colors = None

        self.gs_model_list = None
        self.query_points = None
        self.cached_ready = False

    def load_model(self):
        self.model = ModelLAM(**self.cfg.model)

        ckpt_path = os.path.join(self.model_name, "model.safetensors")
        print("Loading LAM weights:", ckpt_path)

        ckpt = load_file(ckpt_path, device="cpu")
        state_dict = self.model.state_dict()

        for k, v in ckpt.items():
            if k in state_dict and state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                print(f"[WARN] skipped weight {k}: {v.shape}")

        self.model.to(self.device)
        self.model.eval()
        print("LAM model loaded")

    def prepare_source_image(self, image_path):
        """
        Assumes image_path is already an exported/tracked LAM image path:
        .../images/00000_00.png
        with canonical_flame_param.npz two levels above.

        Later we can integrate FlameTrackingSingleImage here.
        """
        mask_path = image_path.replace("/images/", "/fg_masks/").replace(".jpg", ".png")

        if not os.path.exists(mask_path):
            print("[WARN] mask not found:", mask_path)
            mask_path = None

        source_size = self.cfg.dataset.source_image_res

        image, _, _, shape_param = preprocess_image(
            image_path,
            mask_path=mask_path,
            intr=None,
            pad_ratio=0,
            bg_color=1.0,
            max_tgt_size=None,
            aspect_standard=1.0,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size,
            multiply=14,
            need_mask=True,
            get_shape_param=True,
        )

        self.source_image = image.unsqueeze(0).to(self.device, self.dtype)
        self.source_betas = shape_param.to(self.device, self.dtype)

        print("Source image:", self.source_image.shape)
        print("Source betas:", self.source_betas.shape)

    def prepare_default_camera(self):
        render_size = self.render_size

        c2w = torch.eye(4, device=self.device, dtype=self.dtype).view(1, 1, 4, 4)

        intr = torch.eye(4, device=self.device, dtype=self.dtype).view(1, 1, 4, 4)
        intr[:, :, 0, 0] = render_size
        intr[:, :, 1, 1] = render_size
        intr[:, :, 0, 2] = render_size / 2
        intr[:, :, 1, 2] = render_size / 2

        bg = torch.zeros(1, 1, 3, device=self.device, dtype=self.dtype)

        self.render_c2ws = c2w
        self.render_intrs = intr
        self.render_bg_colors = bg

    def get_source_betas_cpu(self):
        return self.source_betas.detach().cpu()

    @torch.no_grad()
    def render_frame_naive(self, flame_params):
        """
        flame_params must be:
        {
          expr: [1, 1, 10],
          rotation: [1, 1, 3],
          neck_pose: [1, 1, 3],
          jaw_pose: [1, 1, 3],
          eyes_pose: [1, 1, 6],
          translation: [1, 1, 3],
          betas: [1, 10],
        }
        """

        flame_params = {
            k: v.to(self.device, self.dtype)
            for k, v in flame_params.items()
        }

        res = self.model.infer_single_view(
            self.source_image,
            None,
            None,
            render_c2ws=self.render_c2ws,
            render_intrs=self.render_intrs,
            render_bg_colors=self.render_bg_colors,
            flame_params=flame_params,
        )

        rgb = res["comp_rgb"].detach().cpu().numpy()

        # Expected shape: [Nv, H, W, 3]
        if rgb.ndim == 4:
            rgb = rgb[0]

        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = (rgb * 255).astype(np.uint8)


        return rgb

    @torch.no_grad()
    def build_avatar_once(self, initial_flame_params):
        assert self.model is not None
        assert self.source_image is not None
    
        initial_flame_params = {
            k: v.to(self.device, self.dtype)
            for k, v in initial_flame_params.items()
        }
    
        image = self.source_image
    
        query_points = None
    
        if self.model.latent_query_points_type.startswith("e2e_flame"):
            query_points, initial_flame_params = self.model.renderer.get_query_points(
                initial_flame_params,
                device=image.device,
            )
    
        latent_points, image_feats = self.model.forward_latent_points(
            image[:, 0],
            camera=None,
            query_points=query_points,
        )
    
        image_feats_bchw = rearrange(
            image_feats,
            "b (h w) c -> b c h w",
            h=int(math.sqrt(image_feats.shape[1])),
        )
    
        self.gs_model_list, self.query_points, _, _ = self.model.renderer.forward_gs(
            gs_hidden_features=latent_points,
            query_points=query_points,
            flame_data=initial_flame_params,
            additional_features={
                "image_feats": image_feats,
                "image": image[:, 0],
                "image_feats_bchw": image_feats_bchw,
            },
        )
    
        self.cached_ready = True
        print("LAM cached avatar ready")

    @torch.no_grad()
    def render_frame_cached(self, flame_params):
        assert self.cached_ready
        assert self.gs_model_list is not None
        assert self.query_points is not None
    
        flame_params = {
            k: v.to(self.device, self.dtype)
            for k, v in flame_params.items()
        }
    
        render_h = int(self.render_intrs[0, 0, 1, 2] * 2)
        render_w = int(self.render_intrs[0, 0, 0, 2] * 2)
    
        single_view_flame = self.model.renderer.get_single_view_smpl_data(
            flame_params,
            0,
        )
    
        render_res = self.model.renderer.forward_animate_gs(
            self.gs_model_list,
            self.query_points,
            single_view_flame,
            self.render_c2ws[:, 0:1],
            self.render_intrs[:, 0:1],
            render_h,
            render_w,
            self.render_bg_colors[:, 0:1],
        )
    
        rgb = render_res["comp_rgb"]
    
        # expected: [B, V, 3, H, W]
        if rgb.ndim == 5:
            rgb = rgb[0, 0].permute(1, 2, 0)
        elif rgb.ndim == 4:
            rgb = rgb[0].permute(1, 2, 0)
    
        rgb = rgb.detach().cpu().numpy()
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = (rgb * 255).astype(np.uint8)
    
        return rgb

    def prepare_camer_from_motion_dir(self, motion_dir):
        if motion_dir.endswith("/"):
            motion_dir = motion_dir[:-1]

        if os.path.basename(motion_dir) == "flame_param":
            base_dir = os.path.dirname(motion_dir)
        else:
            base_dir = motion_dir

        transforms_path = os.path.join(base_dir, "transforms.json")

        if not os.path.exists(transforms_path):
            raise FileNotFoundError(f"transforms.json not found in {base_dir}")

        with open(transforms_path, "r") as f:
            data = json.load(f)

        frame_info = sorted(data["frames"], key=lambda x : x.get("flame_param_path", ""))[0]

        c2w = np.array(frame_info["transform_matrix"], dtype=np.float32)
        c2w[:3, 1:3] *= -1
        c2w = torch.from_numpy(c2w).to(self.device, self.dtype)

        intrinsic = torch.eye(4, device=self.device, dtype=self.dtype)
        intrinsic[0, 0] = float(frame_info["fl_x"])
        intrinsic[1, 1] = float(frame_info["fl_y"])
        intrinsic[0, 2] = float(frame_info["cx"])
        intrinsic[1, 2] = float(frame_info["cy"])
    
        # Questo replica prepare_motion_seqs/head_utils.py
        render_scale = 0.5
        intrinsic[0] = intrinsic[0] * render_scale
        intrinsic[1] = intrinsic[1] * render_scale
    
        self.render_c2ws = c2w.view(1, 1, 4, 4)
        self.render_intrs = intrinsic.view(1, 1, 4, 4)
    
        # Per debug: nero. Poi si può tornare a bianco.
        self.render_bg_colors = torch.ones(1, 1, 3, device=self.device, dtype=self.dtype)
    
        render_h = int(self.render_intrs[0, 0, 1, 2] * 2)
        render_w = int(self.render_intrs[0, 0, 0, 2] * 2)
    
        # print("[LAM camera] loaded from:", transforms_path)
        # print("[LAM camera] render_h/render_w:", render_h, render_w)
        # print("[LAM camera] c2w:", self.render_c2ws.shape)
        # print("[LAM camera] intr:", self.render_intrs.shape)

