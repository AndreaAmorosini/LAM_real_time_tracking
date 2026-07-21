# carichi modello LAM
# - fai preprocess immagine sorgente
# - estrai `shape_param`
# - avvii webcam
# - per ogni frame:
#   - genera `flame_params`
#   - chiama `infer_single_view`
#   - mostra frame in GUI

class LAMLiveRenderer:
    def __init__(self, model, image, render_intrs, render_c2ws, bg_color, device="cuda"):
        self.model = model
        self.image = image
        self.device = device
        self.render_intrs = render_intrs
        self.render_c2ws = render_c2ws
        self.bg_color = bg_color

        self.gs_model_list = None
        self.query_points = None

    @torch.no_grad()
    def build_avatar_once(self, initial_flame_params):
        image = self.image.unsqueeze(0).to(self.device)

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

    @torch.no_grad()
    def render_one_frame(self, flame_params):
        render_h = int(self.render_intrs[0, 0, 1, 2] * 2)
        render_w = int(self.render_intrs[0, 0, 0, 2] * 2)

        render_res = self.model.renderer.forward_animate_gs(
            self.gs_model_list,
            self.query_points,
            self.model.renderer.get_single_view_smpl_data(flame_params, 0),
            self.render_c2ws[:, 0:1],
            self.render_intrs[:, 0:1],
            render_h,
            render_w,
            self.bg_color[:, 0:1],
        )

        rgb = render_res["comp_rgb"][0, 0]
        rgb = rgb.detach().permute(1, 2, 0).cpu().numpy()
        rgb = (rgb.clip(0, 1) * 255).astype("uint8")
        return rgb
