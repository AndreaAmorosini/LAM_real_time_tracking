# leggere frame webcam
# - fare tracking
# - smooth blendshapes
# - convertire a FLAME params

from lam.live.camera import WebcamSource
from lam.live.trackers.mediapipe_face_tracker import MediaPipeFaceTracker
from lam.live.filters import BlendshapeSmoother
from lam.live.retargeting.mediapipe_to_flame import MediaPipeToFlameAdapter
import torch

def default_render_camera(render_size=512, device="cuda"):
    c2w = torch.eye(4, device=device).view(1, 1, 4, 4)

    intr = torch.eye(4, device=device).view(1, 1, 4, 4)
    intr[:, :, 0, 0] = render_size
    intr[:, :, 1, 1] = render_size
    intr[:, :, 0, 2] = render_size / 2
    intr[:, :, 1, 2] = render_size / 2

    bg = torch.ones(1, 1, 3, device=device)

    return c2w, intr, bg

class LiveMotionProvider:
    def __init__(
        self,
        camera_id=0,
        mediapipe_model_path="model_zoo/mediapipe/face_landmarker.task",
        device="cuda",
        width=640,
        height=480,
        fps=30,
        smoothing_alpha=0.45,
        expr_dim=100,
    ):
        self.camera = WebcamSource(camera_id=camera_id, width=width, height=height, fps=fps)
        self.tracker = MediaPipeFaceTracker(model_path=mediapipe_model_path)
        self.smoother = BlendshapeSmoother(alpha=smoothing_alpha)
        self.adapter = MediaPipeToFlameAdapter(device=device, expr_dim=expr_dim)

    def open(self):
        self.camera.open()
        self.tracker.open()

    def read(self, source_betas):
        frame_rgb = self.camera.read_rgb()
        if frame_rgb is None:
            return None, None, None, False

        tracking = self.tracker.track(frame_rgb)
        if not tracking.detected:
            return frame_rgb, None, tracking, False

        tracking.blendshapes = self.smoother.smooth(tracking.blendshapes)
        flame_params = self.adapter.to_flame_params(tracking, source_betas)

        return frame_rgb, flame_params, tracking, True

    def close(self):
        self.tracker.close()
        self.camera.close()

def main():
    provider = LiveMotionProvider(device="cpu")
    provider.open()
    
    source_betas = torch.zeros(10)
    
    for _ in range(100):
        frame, flame_params, tracking, ok = provider.read(source_betas)
        print(ok, None if flame_params is None else flame_params["expr"][0, 0].tolist())
    
    provider.close()

if __name__ == "__main__":
    main()