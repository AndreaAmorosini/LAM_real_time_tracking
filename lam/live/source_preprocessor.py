import hashlib
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

from tools.flame_tracking_single_image import FlameTrackingSingleImage


class LAMSourcePreprocessor:
    def __init__(
        self,
        output_dir="tracking_output_live",
        detect_iris_landmarks=True,
    ):
        self.output_dir = output_dir
        self.detect_iris_landmarks = detect_iris_landmarks
        self._tracker = None

    def _get_tracker(self):
        if self._tracker is not None:
            return self._tracker

        args = SimpleNamespace(
            output_dir=self.output_dir,
            config_name="alignment",
            blender_path=None,
        )

        self._tracker = FlameTrackingSingleImage(
            output_dir=self.output_dir,
            alignment_model_path="./model_zoo/flame_tracking_models/68_keypoints_model.pkl",
            vgghead_model_path="./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd",
            human_matting_path="./model_zoo/flame_tracking_models/matting/stylematte_synth.pt",
            facebox_model_path="./model_zoo/flame_tracking_models/FaceBoxesV2.pth",
            detect_iris_landmarks=self.detect_iris_landmarks,
            args=args,
        )

        return self._tracker

    def _stage_input(self, raw_image_path: str) -> str:
        raw_image_path = os.path.abspath(raw_image_path)

        if not os.path.exists(raw_image_path):
            raise FileNotFoundError(raw_image_path)

        src = Path(raw_image_path)
        stat = src.stat()

        key = f"{raw_image_path}:{stat.st_mtime_ns}:{stat.st_size}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]

        staged_dir = Path(self.output_dir) / "raw_inputs"
        staged_dir.mkdir(parents=True, exist_ok=True)

        staged_name = f"{src.stem}_{digest}{src.suffix}"
        staged_path = staged_dir / staged_name

        if not staged_path.exists():
            shutil.copy2(raw_image_path, staged_path)

        return str(staged_path)

    def _expected_export_image(self, staged_image_path: str) -> str:
        stem = Path(staged_image_path).stem
        return str(
            Path(self.output_dir)
            / "export"
            / stem
            / "images"
            / "00000_00.png"
        )

    def _expected_canonical_flame(self, staged_image_path: str) -> str:
        stem = Path(staged_image_path).stem
        return str(
            Path(self.output_dir)
            / "export"
            / stem
            / "canonical_flame_param.npz"
        )

    def preprocess(self, raw_image_path: str, force=False) -> str:
        staged_image_path = self._stage_input(raw_image_path)

        export_image = self._expected_export_image(staged_image_path)
        canonical_flame = self._expected_canonical_flame(staged_image_path)

        if (
            not force
            and os.path.exists(export_image)
            and os.path.exists(canonical_flame)
        ):
            print("[SOURCE PREPROCESS] using cached export:", export_image)
            return export_image

        tracker = self._get_tracker()

        print("[SOURCE PREPROCESS] preprocessing:", staged_image_path)
        ret = tracker.preprocess(staged_image_path)
        if ret != 0:
            raise RuntimeError(f"flame preprocess failed with code {ret}")

        print("[SOURCE PREPROCESS] optimizing...")
        ret = tracker.optimize()
        if ret != 0:
            raise RuntimeError(f"flame optimize failed with code {ret}")

        print("[SOURCE PREPROCESS] exporting...")
        ret, output_dir = tracker.export()
        if ret != 0:
            raise RuntimeError(f"flame export failed with code {ret}")

        export_image = os.path.join(output_dir, "images", "00000_00.png")

        if not os.path.exists(export_image):
            raise FileNotFoundError(f"export image not found: {export_image}")

        canonical_flame = os.path.join(output_dir, "canonical_flame_param.npz")
        if not os.path.exists(canonical_flame):
            raise FileNotFoundError(f"canonical flame not found: {canonical_flame}")

        print("[SOURCE PREPROCESS] done:", export_image)
        return export_image
