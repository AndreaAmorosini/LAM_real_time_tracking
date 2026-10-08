from pydantic import BaseSettings, Field


class LiveSettings(BaseSettings):
    # Server
    web_host: str = Field("127.0.0.1", env="LAM_WEB_HOST")
    web_port: int = Field(7861, env="LAM_WEB_PORT")
    web_reload: bool = Field(False, env="LAM_WEB_RELOAD")
    log_level: str = Field("info", env="LAM_LOG_LEVEL")

    # Paths
    output_root: str = Field("output", env="LAM_OUTPUT_ROOT")
    oac_output_root: str = Field("output/open_avatar_chat", env="LAM_OAC_OUTPUT_ROOT")
    upload_dir: str = Field("output/live_uploads", env="LAM_UPLOAD_DIR")
    tracking_output_dir: str = Field("tracking_output_live", env="LAM_TRACKING_OUTPUT_DIR")
    webgl_dist_dir: str = Field("webgl_frontend/dist", env="LAM_WEBGL_DIST_DIR")

    # Routes
    oac_assets_route: str = Field("/oac_assets", env="LAM_OAC_ASSETS_ROUTE")
    webgl_route: str = Field("/webgl", env="LAM_WEBGL_ROUTE")

    # LAM
    lam_config_path: str = Field("configs/inference/lam-20k-8gpu.yaml", env="LAM_CONFIG_PATH")
    lam_arkit_config_path: str = Field("configs/inference/live-arkit.yaml", env="LAM_ARKIT_CONFIG_PATH")
    lam_model_name: str = Field("./model_zoo/lam_models/releases/lam/lam-20k/step_045500/", env="LAM_MODEL_NAME")
    lam_device: str = Field("cuda", env="LAM_DEVICE")
    lam_render_size: int = Field(512, env="LAM_RENDER_SIZE")
    camera_motion_dir: str = Field("assets/sample_motion/export/Look_In_My_Eyes/", env="LAM_CAMERA_MOTION_DIR")

    # Blender
    blender_path: str = Field("blender", env="LAM_BLENDER_PATH")

    # Tracking
    camera_index: int = Field(0, env="LAM_CAMERA_INDEX")
    capture_width: int = Field(320, env="LAM_CAPTURE_WIDTH")
    capture_height: int = Field(240, env="LAM_CAPTURE_HEIGHT")
    capture_fps: int = Field(30, env="LAM_CAPTURE_FPS")
    default_ui_fps: float = Field(20.0, env="LAM_DEFAULT_UI_FPS")
    webgl_ui_fps: float = Field(30.0, env="LAM_WEBGL_UI_FPS")
    jpeg_quality: int = Field(80, env="LAM_JPEG_QUALITY")
    mapping_mode: str = Field("stable", env="LAM_MAPPING_MODE")

    # Derived landmarks
    derived_neutral_frames: int = Field(20, env="LAM_DERIVED_NEUTRAL_FRAMES")
    derived_alpha: float = Field(0.60, env="LAM_DERIVED_ALPHA")

    # Jaw tuning
    jaw_derived_weight: float = Field(0.40, env="LAM_JAW_DERIVED_WEIGHT")
    jaw_min: float = Field(0.04, env="LAM_JAW_MIN")
    jaw_cap: float = Field(0.62, env="LAM_JAW_CAP")
    stable_jaw_deadzone: float = Field(0.04, env="LAM_STABLE_JAW_DEADZONE")
    stable_jaw_gain: float = Field(0.70, env="LAM_STABLE_JAW_GAIN")

    # Speech tuning
    speech_open_gain: float = Field(2.35, env="LAM_SPEECH_OPEN_GAIN")
    speech_open_weight_to_jaw: float = Field(0.85, env="LAM_SPEECH_OPEN_WEIGHT_TO_JAW")
    big_jaw_gain: float = Field(1.75, env="LAM_BIG_JAW_GAIN")
    big_jaw_deadzone: float = Field(0.20, env="LAM_BIG_JAW_DEADZONE")
    big_jaw_power: float = Field(1.45, env="LAM_BIG_JAW_POWER")
    landmark_jaw_cap: float = Field(0.75, env="LAM_LANDMARK_JAW_CAP")

    # O / U vowel tuning
    mouth_narrow_gain: float = Field(4.2, env="LAM_MOUTH_NARROW_GAIN")
    vowel_o_funnel_gain: float = Field(0.90, env="LAM_VOWEL_O_FUNNEL_GAIN")
    vowel_o_pucker_gain: float = Field(0.35, env="LAM_VOWEL_O_PUCKER_GAIN")
    vowel_o_jaw_gain: float = Field(0.28, env="LAM_VOWEL_O_JAW_GAIN")
    vowel_u_pucker_gain: float = Field(0.95, env="LAM_VOWEL_U_PUCKER_GAIN")
    vowel_u_funnel_gain: float = Field(0.30, env="LAM_VOWEL_U_FUNNEL_GAIN")

    # Stable postprocess tuning
    stable_mouth_funnel_deadzone: float = Field(0.035, env="LAM_STABLE_MOUTH_FUNNEL_DEADZONE")
    stable_mouth_pucker_deadzone: float = Field(0.035, env="LAM_STABLE_MOUTH_PUCKER_DEADZONE")
    stable_mouth_lower_down_deadzone: float = Field(0.025, env="LAM_STABLE_MOUTH_LOWER_DOWN_DEADZONE")
    stable_mouth_stretch_deadzone: float = Field(0.025, env="LAM_STABLE_MOUTH_STRETCH_DEADZONE")

    stable_mouth_funnel_gain: float = Field(1.00, env="LAM_STABLE_MOUTH_FUNNEL_GAIN")
    stable_mouth_pucker_gain: float = Field(1.00, env="LAM_STABLE_MOUTH_PUCKER_GAIN")
    stable_mouth_lower_down_gain: float = Field(0.85, env="LAM_STABLE_MOUTH_LOWER_DOWN_GAIN")
    stable_mouth_stretch_gain: float = Field(1.05, env="LAM_STABLE_MOUTH_STRETCH_GAIN")

    # Mouth conflict tuning
    mouthclose_jaw_threshold: float = Field(0.06, env="LAM_MOUTHCLOSE_JAW_THRESHOLD")
    mouthclose_jaw_reduction_range: float = Field(0.12, env="LAM_MOUTHCLOSE_JAW_REDUCTION_RANGE")
    pucker_funnel_dominance_ratio: float = Field(1.35, env="LAM_PUCKER_FUNNEL_DOMINANCE_RATIO")
    pucker_funnel_weaker_scale: float = Field(0.75, env="LAM_PUCKER_FUNNEL_WEAKER_SCALE")
    rounding_mouthclose_threshold: float = Field(0.06, env="LAM_ROUNDING_MOUTHCLOSE_THRESHOLD")
    rounding_mouthclose_range: float = Field(0.35, env="LAM_ROUNDING_MOUTHCLOSE_RANGE")
    rounding_mouthclose_scale: float = Field(0.75, env="LAM_ROUNDING_MOUTHCLOSE_SCALE")

    # Multi-view refinement
    multiview_refine_iters: int = Field(200, env="LAM_MULTIVIEW_REFINE_ITERS")
    multiview_refine_lr_rgb: float = Field(3e-2, env="LAM_MULTIVIEW_REFINE_LR_RGB")
    multiview_refine_lr_opacity: float = Field(1e-2, env="LAM_MULTIVIEW_REFINE_LR_OPACITY")
    multiview_refine_lr_offset: float = Field(5e-3, env="LAM_MULTIVIEW_REFINE_LR_OFFSET")
    multiview_refine_lr_scale: float = Field(2e-3, env="LAM_MULTIVIEW_REFINE_LR_SCALE")
    multiview_refine_lambda_mask: float = Field(0.2, env="LAM_MULTIVIEW_REFINE_LAMBDA_MASK")
    multiview_refine_lambda_offset: float = Field(5.0, env="LAM_MULTIVIEW_REFINE_LAMBDA_OFFSET")
    multiview_refine_lambda_scale: float = Field(0.1, env="LAM_MULTIVIEW_REFINE_LAMBDA_SCALE")
    multiview_refine_lambda_opacity: float = Field(0.01, env="LAM_MULTIVIEW_REFINE_LAMBDA_OPACITY")
    multiview_refine_lr_rotation: float = Field(1e-4, env="LAM_MULTIVIEW_REFINE_LR_ROTATION")
    multiview_refine_max_offset_delta: float = Field(0.0, env="LAM_MULTIVIEW_REFINE_MAX_OFFSET_DELTA")
    multiview_refine_optimize_scale: bool = Field(False, env="LAM_MULTIVIEW_REFINE_OPTIMIZE_SCALE")
    multiview_refine_optimize_rotation: bool = Field(False, env="LAM_MULTIVIEW_REFINE_OPTIMIZE_ROTATION")
    multiview_refine_max_iters: int = Field(3000, env="LAM_MULTIVIEW_REFINE_MAX_ITERS")
    multiview_refine_lambda_rgb_prior: float = Field(
        10.0,
        env="LAM_MULTIVIEW_REFINE_LAMBDA_RGB_PRIOR",
    )
    
    multiview_weight_front: float = Field(1.0, env="LAM_MULTIVIEW_WEIGHT_FRONT")
    multiview_weight_left: float = Field(0.25, env="LAM_MULTIVIEW_WEIGHT_LEFT")
    multiview_weight_right: float = Field(0.25, env="LAM_MULTIVIEW_WEIGHT_RIGHT")
    multiview_weight_up: float = Field(0.15, env="LAM_MULTIVIEW_WEIGHT_UP")
    multiview_weight_down: float = Field(0.10, env="LAM_MULTIVIEW_WEIGHT_DOWN")


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = LiveSettings()
