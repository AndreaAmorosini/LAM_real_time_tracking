import cv2
import numpy as np
import torch

ARKIT_52 = [
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
]


class MediaPipeToFlameAdapter:
    def __init__(
        self,
        device="cuda",
        dtype=torch.float32,
        enable_head_pose=True,
        calibrate_neutral_head=True,
        expr_dim=100
    ):
        self.device = device
        self.dtype = dtype
        self.enable_head_pose = enable_head_pose
        self.calibrate_neutral_head = calibrate_neutral_head
        self.expr_dim = expr_dim

        self.neutral_R = None

        # Gains principali
        self.jaw_open_gain = 0.55
        self.jaw_side_gain = 0.20

        self.eye_gaze_gain = 0.25

        self.head_rotation_gain = 0.75
        self.neck_rotation_gain = 0.25

        # Se la testa ruota al contrario, modifica questi segni.
        self.head_axis_sign = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def _b(self, blendshapes, name, default=0.0):
        return float(blendshapes.get(name, default))

    def _clamp01(self, x):
        return float(np.clip(x, 0.0, 1.0))

    def expression_arkit_52d(self, blendshapes):
        expr = np.zeros(52, dtype=np.float32)
    
        for i, name in enumerate(ARKIT_52):
            expr[i] = self._b(blendshapes, name, 0.0)
    
        return expr


    def expression_100d(self, blendshapes):
        """
        Mapping euristico MediaPipe/ARKit-like -> FLAME expression 100D.

        Nota: i coefficienti FLAME non sono semanticamente identici ai blendshape
        MediaPipe. Questa è una prima mappatura manuale da calibrare visivamente.
        """
        jaw_open = self._b(blendshapes, "jawOpen")

        smile_l = self._b(blendshapes, "mouthSmileLeft")
        smile_r = self._b(blendshapes, "mouthSmileRight")

        frown_l = self._b(blendshapes, "mouthFrownLeft")
        frown_r = self._b(blendshapes, "mouthFrownRight")

        pucker = self._b(blendshapes, "mouthPucker")
        funnel = self._b(blendshapes, "mouthFunnel")

        blink_l = self._b(blendshapes, "eyeBlinkLeft")
        blink_r = self._b(blendshapes, "eyeBlinkRight")

        squint_l = self._b(blendshapes, "eyeSquintLeft")
        squint_r = self._b(blendshapes, "eyeSquintRight")

        brow_up_l = self._b(blendshapes, "browOuterUpLeft")
        brow_up_r = self._b(blendshapes, "browOuterUpRight")

        brow_down_l = self._b(blendshapes, "browDownLeft")
        brow_down_r = self._b(blendshapes, "browDownRight")

        expr = np.zeros(100, dtype=np.float32)

        # Alta priorità: bocca
        expr[0] = jaw_open * 0.35
        expr[1] = 0.5 * (smile_l + smile_r) * 0.85
        expr[2] = 0.5 * (frown_l + frown_r) * 0.60
        expr[3] = pucker * 0.85
        expr[4] = funnel * 0.85

        # Alta priorità: blink/squint
        expr[5] = 0.5 * (blink_l + blink_r) * 1.00
        expr[6] = 0.5 * (squint_l + squint_r) * 0.65

        # Alta priorità: sopracciglia
        expr[7] = 0.5 * (brow_up_l + brow_up_r) * 0.75
        expr[8] = 0.5 * (brow_down_l + brow_down_r) * 0.55

        # Asimmetrie utili
        expr[9] = (smile_l - smile_r) * 0.50
        expr[10] = (blink_l - blink_r) * 0.50
        expr[11] = (brow_up_l - brow_up_r) * 0.40

        return expr

    def jaw_pose(self, blendshapes):
        jaw_open = self._b(blendshapes, "jawOpen")
        jaw_left = self._b(blendshapes, "jawLeft")
        jaw_right = self._b(blendshapes, "jawRight")

        jaw = np.zeros(3, dtype=np.float32)

        # Apertura mandibola.
        # Se la bocca si chiude invece di aprirsi, cambia segno.
        jaw[0] = jaw_open * self.jaw_open_gain

        # Movimento laterale mandibola.
        jaw[1] = (jaw_left - jaw_right) * self.jaw_side_gain

        jaw[2] = 0.0
        return jaw

    def eyes_pose(self, blendshapes):
        eyes = np.zeros(6, dtype=np.float32)

        look_up_l = self._b(blendshapes, "eyeLookUpLeft")
        look_down_l = self._b(blendshapes, "eyeLookDownLeft")
        look_in_l = self._b(blendshapes, "eyeLookInLeft")
        look_out_l = self._b(blendshapes, "eyeLookOutLeft")

        look_up_r = self._b(blendshapes, "eyeLookUpRight")
        look_down_r = self._b(blendshapes, "eyeLookDownRight")
        look_in_r = self._b(blendshapes, "eyeLookInRight")
        look_out_r = self._b(blendshapes, "eyeLookOutRight")

        g = self.eye_gaze_gain

        # Occhio sinistro
        eyes[0] = (look_down_l - look_up_l) * g
        eyes[1] = (look_in_l - look_out_l) * g
        eyes[2] = 0.0

        # Occhio destro
        eyes[3] = (look_down_r - look_up_r) * g
        eyes[4] = (look_out_r - look_in_r) * g
        eyes[5] = 0.0

        return eyes

    def rotation_from_matrix(self, facial_matrix):
        """
        MediaPipe facial_transformation_matrix -> axis-angle approssimato.

        Usa calibrazione neutra sul primo frame valido, così la posa iniziale
        diventa zero.
        """
        if not self.enable_head_pose:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        if facial_matrix is None:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        M = np.asarray(facial_matrix, dtype=np.float32)

        if M.shape != (4, 4):
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        R = M[:3, :3]

        if self.calibrate_neutral_head and self.neutral_R is None:
            self.neutral_R = R.copy()

        if self.neutral_R is not None:
            # Rotazione relativa rispetto alla posa neutra iniziale.
            R_rel = R @ self.neutral_R.T
        else:
            R_rel = R

        rvec, _ = cv2.Rodrigues(R_rel)
        rvec = rvec.reshape(3).astype(np.float32)

        rvec = rvec * self.head_axis_sign

        # Split tra root/head e neck.
        head_rot = rvec * self.head_rotation_gain
        neck_rot = rvec * self.neck_rotation_gain

        return head_rot, neck_rot

    def translation_from_matrix(self, facial_matrix):
        """
        Per ora disabilitata: la translation da MediaPipe è rumorosa e non
        direttamente coerente con lo spazio FLAME/LAM.
        """
        return np.zeros(3, dtype=np.float32)

    def to_flame_params(self, tracker_result, source_betas):
        blendshapes = tracker_result.blendshapes

        if self.expr_dim == 52:
            expr = self.expression_arkit_52d(blendshapes)
        else:
            expr = self.expression_100d(blendshapes)
        jaw = self.jaw_pose(blendshapes)
        eyes = self.eyes_pose(blendshapes)

        rotation, neck = self.rotation_from_matrix(tracker_result.facial_matrix)
        translation = self.translation_from_matrix(tracker_result.facial_matrix)

        def t(x):
            return torch.tensor(x, device=self.device, dtype=self.dtype)

        flame_params = {
            "expr": t(expr).view(1, 1, self.expr_dim),
            "rotation": t(rotation).view(1, 1, 3),
            "neck_pose": t(neck).view(1, 1, 3),
            "jaw_pose": t(jaw).view(1, 1, 3),
            "eyes_pose": t(eyes).view(1, 1, 6),
            "translation": t(translation).view(1, 1, 3),
            "betas": source_betas.to(self.device, self.dtype).view(1, -1),
        }

        return flame_params


def main():
    adapter = MediaPipeToFlameAdapter(device="cpu")

    fake = type("Fake", (), {})()
    fake.blendshapes = {
        "jawOpen": 0.8,
        "mouthSmileLeft": 0.5,
        "mouthSmileRight": 0.4,
        "eyeBlinkLeft": 0.2,
        "browOuterUpLeft": 0.6,
    }
    fake.facial_matrix = np.eye(4, dtype=np.float32)

    betas = torch.zeros(300)
    params = adapter.to_flame_params(fake, betas)

    for k, v in params.items():
        print(k, v.shape)


if __name__ == "__main__":
    main()
