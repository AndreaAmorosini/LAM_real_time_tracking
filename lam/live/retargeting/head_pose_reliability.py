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
                "mouth": 1.0,
                "brows": 1.0,
                "eyes": 1.0,
                "nose_cheek": 1.0,
                "raw": 1.0,
                "pose_amount": 0.0,
                "motion_amount": 0.0,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "centered_pitch": 0.0,
                "neutral_pitch": self.neutral_pitch,
                "head_up": 0.0,
                "head_down": 0.0,
            }
    
        yaw, pitch, roll = self._angles_from_matrix(tracking.facial_matrix)
    
        self.frame_count += 1
    
        if self.frame_count <= self.neutral_frames:
            self.neutral_pitch = 0.90 * self.neutral_pitch + 0.10 * pitch
    
        centered_pitch = pitch - self.neutral_pitch
    
        head_up_amount = max(0.0, -centered_pitch)
        head_down_amount = max(0.0, centered_pitch)
    
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
    
        # Static pose should use centered pitch, not absolute pitch.
        pose_amount = (
            abs(yaw) * 1.0
            + abs(centered_pitch) * 1.15
            + abs(roll) * 0.45
        )
    
        # Dynamic head motion gates.
        motion_strict = self._linear_falloff(
            motion_amount,
            low=0.015,
            high=0.070,
            min_value=0.03,
        )
    
        motion_soft = self._linear_falloff(
            motion_amount,
            low=0.025,
            high=0.120,
            min_value=0.20,
        )
    
        # Static pose gates.
        pose_strict = self._linear_falloff(
            pose_amount,
            low=0.10,
            high=0.36,
            min_value=0.05,
        )
    
        pose_soft = self._linear_falloff(
            pose_amount,
            low=0.16,
            high=0.62,
            min_value=0.20,
        )
    
        # Region-specific gates.
        jaw_eye = pose_strict * motion_strict
    
        eyes = self._linear_falloff(
            abs(yaw) + abs(centered_pitch) * 1.4 + abs(roll) * 0.5,
            low=0.08,
            high=0.32,
            min_value=0.05,
        ) * motion_strict
    
        mouth = self._linear_falloff(
            abs(yaw) + abs(centered_pitch) * 1.0 + abs(roll) * 0.35,
            low=0.12,
            high=0.48,
            min_value=0.12,
        ) * motion_soft
    
        brows = self._linear_falloff(
            abs(yaw) * 0.7 + head_up_amount * 1.5 + abs(roll) * 0.25,
            low=0.08,
            high=0.34,
            min_value=0.05,
        ) * motion_soft
    
        nose_cheek = self._linear_falloff(
            abs(yaw) + abs(centered_pitch) * 0.9 + abs(roll) * 0.4,
            low=0.12,
            high=0.48,
            min_value=0.08,
        ) * motion_soft
    
        derived = pose_soft * motion_soft
        raw = pose_soft * motion_soft
    
        return {
            "jaw_eye": float(max(0.0, min(1.0, jaw_eye))),
            "derived": float(max(0.0, min(1.0, derived))),
            "mouth": float(max(0.0, min(1.0, mouth))),
            "brows": float(max(0.0, min(1.0, brows))),
            "eyes": float(max(0.0, min(1.0, eyes))),
            "nose_cheek": float(max(0.0, min(1.0, nose_cheek))),
            "raw": float(max(0.0, min(1.0, raw))),
            "motion_strict": float(max(0.0, min(1.0, motion_strict))),
            "motion_soft": float(max(0.0, min(1.0, motion_soft))),
            "pose_strict": float(max(0.0, min(1.0, pose_strict))),
            "pose_soft": float(max(0.0, min(1.0, pose_soft))),
            "pose_amount": float(pose_amount),
            "motion_amount": float(motion_amount),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
            "centered_pitch": float(centered_pitch),
            "neutral_pitch": float(self.neutral_pitch),
            "head_up": float(head_up_amount),
            "head_down": float(head_down_amount),
        }
