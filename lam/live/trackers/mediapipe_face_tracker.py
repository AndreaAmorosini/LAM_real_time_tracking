# carica `face_landmarker.task`
# - prende un frame RGB
# - restituisce:
#   - landmarks
#   - blendshapes
#   - facial transformation matrix


from dataclasses import dataclass
import time
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class MediaPipeFaceResult:
    timestamp_ms: int
    landmarks: object
    blendshapes: dict
    facial_matrix: np.ndarray | None
    detected: bool


class MediaPipeFaceTracker:
    def __init__(
        self,
        model_path="model_zoo/mediapipe/face_landmarker.task",
        running_mode="VIDEO",
        num_faces=1,
    ):
        self.model_path = model_path
        self.running_mode = running_mode
        self.num_faces = num_faces
        self.landmarker = None
        self.t0 = time.time()
        self.last_timestamp_ms = 0

    def open(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=self.num_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def track(self, frame_rgb: np.ndarray) -> MediaPipeFaceResult:
        timestamp_ms = int((time.time() - self.t0) * 1000)

        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
            
        self.last_timestamp_ms = timestamp_ms

        frame_rgb = np.ascontiguousarray(frame_rgb)
        
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return MediaPipeFaceResult(
                timestamp_ms=timestamp_ms,
                landmarks=None,
                blendshapes={},
                facial_matrix=None,
                detected=False,
            )

        blendshapes = {}
        if result.face_blendshapes:
            for category in result.face_blendshapes[0]:
                blendshapes[category.category_name] = float(category.score)

        facial_matrix = None
        if result.facial_transformation_matrixes:
            facial_matrix = np.array(result.facial_transformation_matrixes[0])

        return MediaPipeFaceResult(
            timestamp_ms=timestamp_ms,
            landmarks=result.face_landmarks[0],
            blendshapes=blendshapes,
            facial_matrix=facial_matrix,
            detected=True,
        )

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()


def main():
    from lam.live.camera import WebcamSource

    cam = WebcamSource()
    tracker = MediaPipeFaceTracker()

    cam.open()
    tracker.open()

    try:
        for i in range(100):
            frame = cam.read_rgb()
            out = tracker.track(frame)

            print(
                i,
                "detected:", out.detected,
                "blendshapes:", list(out.blendshapes.items())[:5],
                "matrix:", None if out.facial_matrix is None else out.facial_matrix.shape,
            )
    finally:
        tracker.close()
        cam.close()


if __name__ == "__main__":
    main()
