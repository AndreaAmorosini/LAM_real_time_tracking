import math
import numpy as np


class HeadPoseReliability:
    """
    Computes reliability factors for landmark-derived blendshapes.

    jaw_eye:
      strict factor for jawOpen / eyeWide, because they are very sensitive
      to pose/projection.

    derived:
      softer factor for brows / nose / cheeks.
    """

    def __init__(self):
        self.prev_yaw = None
        self.prev_pitch = None
        self.prev_roll = None

    def _angles_from_matrix(self, facial_matrix):
        if facial_matrix is None:
            return 0.0, 0.0, 0.0

        M = np.asarray(facial_matrix, dtype=float)

        if M.shape != (4, 4):
            return 0.0, 0.0, 0.0

        yaw = math.atan2(M[0][2], M[2][2])
        pitch = math.atan2(
            -M[1][2],
            math.sqrt(M[1][0] ** 2 + M[1][1] ** 2),
        )
        roll = math.atan2(M[1][0], M[1][1])

        return yaw, pitch, roll

    def _linear_falloff(self, amount, low, high, min_value=0.0):
        if amount <= low:
            return 1.0
        if amount >= high:
            return min_value

        t = (amount - low) / (high - low)
        return 1.0 * (1.0 - t) + min_value * t

    def update(self, tracking):
        if tracking is None or tracking.facial_matrix is None:
            return {
                "jaw_eye": 1.0,
                "derived": 1.0,
                "pose_amount": 0.0,
                "motion_amount": 0.0,
            }

        yaw, pitch, roll = self._angles_from_matrix(tracking.facial_matrix)

        pose_amount = abs(yaw) + abs(pitch) * 0.8 + abs(roll) * 0.4

        if self.prev_yaw is None:
            motion_amount = 0.0
        else:
            motion_amount = (
                abs(yaw - self.prev_yaw)
                + abs(pitch - self.prev_pitch)
                + abs(roll - self.prev_roll) * 0.5
            )

        self.prev_yaw = yaw
        self.prev_pitch = pitch
        self.prev_roll = roll

        # Strict: jawOpen / eyeWide
        # Full under ~7 deg-ish, off after ~22 deg-ish.
        jaw_eye_pose = self._linear_falloff(
            pose_amount,
            low=0.12,
            high=0.38,
            min_value=0.0,
        )

        # Freeze/reduce when head moves quickly.
        jaw_eye_motion = self._linear_falloff(
            motion_amount,
            low=0.035,
            high=0.12,
            min_value=0.0,
        )

        jaw_eye = jaw_eye_pose * jaw_eye_motion

        # Softer: brows / nose / cheeks
        derived_pose = self._linear_falloff(
            pose_amount,
            low=0.18,
            high=0.60,
            min_value=0.25,
        )

        derived_motion = self._linear_falloff(
            motion_amount,
            low=0.06,
            high=0.18,
            min_value=0.35,
        )

        derived = derived_pose * derived_motion

        return {
            "jaw_eye": float(max(0.0, min(1.0, jaw_eye))),
            "derived": float(max(0.0, min(1.0, derived))),
            "pose_amount": float(pose_amount),
            "motion_amount": float(motion_amount),
        }
