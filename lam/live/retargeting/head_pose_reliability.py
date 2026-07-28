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
        self.frame_count = 0
        self.neutral_frames = 20
        self.neutral_pitch = 0.09

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

        self.frame_count += 1

        if self.frame_count <= self.neutral_frames:
            self.neutral_pitch = 0.90 * self.neutral_pitch + 0.10 * pitch

        centered_pitch = pitch - self.neutral_pitch

        # Brows are very sensitive to vertical head pitch.
        # If the user raises/lowers the head, landmark brow metrics drift.
        head_up_amount = max(0.0, -centered_pitch)
        head_down_amount = max(0.0, centered_pitch)
        
        brow_pose = self._linear_falloff(
            head_up_amount,
            low=0.10,
            high=0.34,
            min_value=0.08,
        )
        
        brow_yaw = self._linear_falloff(
            abs(yaw),
            low=0.20,
            high=0.55,
            min_value=0.45,
        )
        
        brows = brow_pose * brow_yaw

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

        mouth_pose = self._linear_falloff(
            abs(yaw) + abs(roll) * 0.35 + head_up_amount * 0.20,
            low=0.25,
            high=0.85,
            min_value=0.65
        )

        mouth_motion = self._linear_falloff(
            motion_amount,
            low=0.08,
            high=0.24,
            min_value=0.65
        )

        mouth = mouth_pose * mouth_motion

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
            "brows": float(max(0.0, min(1.0, brows))),
            "mouth": float(max(0.0, min(1.0, mouth))),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
            "centered_pitch": float(centered_pitch),
            "neutral_pitch": float(self.neutral_pitch),
            "head_up": float(head_up_amount),
            "head_down": float(head_down_amount)
        }
