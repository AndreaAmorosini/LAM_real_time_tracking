class LandmarkDerivedBlendshapes:
    def __init__(self, neutral_frames=20, alpha=0.55):
        self.neutral_frames = neutral_frames
        self.alpha = alpha
        self.frame_count = 0
        self.neutral = {}
        self.values = {}

    def _xy(self, landmarks, idx):
        lm = landmarks[idx]
        return float(lm.x), float(lm.y)

    def _mean_y(self, landmarks, indices):
        return sum(float(landmarks[i].y) for i in indices) / len(indices)

    def _dist_x(self, landmarks, a, b):
        ax, _ = self._xy(landmarks, a)
        bx, _ = self._xy(landmarks, b)
        return abs(ax - bx)

    def _smooth(self, key, value):
        prev = self.values.get(key, value)
        value = self.alpha * value + (1.0 - self.alpha) * prev
        self.values[key] = value
        return value

    def _smoothstep(self, edge0, edge1, x):
        if edge1 <= edge0:
            return 0.0
        t = (float(x) - edge0) / (edge1 - edge0)
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _calibrated_delta(self, key, current, positive_when_larger=True, gain=1.0):
        if self.frame_count < self.neutral_frames:
            old = self.neutral.get(key, current)
            self.neutral[key] = 0.9 * old + 0.1 * current
            return 0.0

        base = self.neutral.get(key, current)
        delta = current - base if positive_when_larger else base - current
        return max(0.0, min(1.0, delta * gain))

    def _dist(self, landmarks, a, b):
        ax, ay = self._xy(landmarks, a)
        bx, by = self._xy(landmarks, b)
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    def _safe_div(self, a, b, fallback=0.0):
        if abs(b) < 1e-6:
            return fallback
        return a / b

    def _speech_curve(self, x):
        """
        Soft-knee curve for speech mouth opening.
    
        Goal:
        - small speech movements become visible
        - medium openings remain controlled
        - large openings do not saturate immediately
        """
        x = max(0.0, min(1.0, float(x)))
    
        if x < 0.07:
            return 0.0
    
        if x < 0.25:
            t = (x - 0.07) / (0.25 - 0.07)
            return t * 0.18
    
        if x < 0.60:
            t = (x - 0.25) / (0.60 - 0.25)
            return 0.18 + t * 0.30
    
        return min(0.62, 0.48 + (x - 0.60) * 0.25)


    def _eye_width_left(self, landmarks):
        return max(self._dist(landmarks, 33, 133), 1e-6)

    def _eye_width_right(self, landmarks):
        return max(self._dist(landmarks, 362, 263), 1e-6)

    def _mouth_width(self, landmarks):
        return max(self._dist(landmarks, 61, 291), 1e-6)

    def _local_face_scale(self, landmarks):
        # More stable than full face width for local mouth/nose metrics.
        return max(self._dist(landmarks, 168, 2), 1e-6)

    def _mean_xy(self, landmarks, indices):
        x = sum(float(landmarks[i].x) for i in indices) / len(indices)
        y = sum(float(landmarks[i].y) for i in indices) / len(indices)
        return x, y

    def _smooth_asymmetric(self, key, value, alpha_open=0.55, alpha_close=0.90):
        prev = self.values.get(key, value)
    
        # If value is decreasing, close faster.
        alpha = alpha_close if value < prev else alpha_open
    
        value = alpha * value + (1.0 - alpha) * prev
        self.values[key] = value
        return value

    def _xyz(self, landmarks, idx):
        lm = landmarks[idx]
        return float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0))

    def _dist3(self, landmarks, a, b, z_weight=0.5):
        ax, ay, az = self._xyz(landmarks, a)
        bx, by, bz = self._xyz(landmarks, b)

        dx = ax - bx
        dy = ay - by
        dz = (az - bz) * z_weight

        return (dx * dx + dy * dy + dz * dz) ** 0.5


    def derive(self, tracking):
        if tracking is None or not tracking.detected or tracking.landmarks is None:
            return {}

        lms = tracking.landmarks
        self.frame_count += 1

        scale = self._dist_x(lms, 33, 263)
        if scale < 1e-6:
            scale = 1.0

        out = {}

        # ------------------------------------------------------------------
        # BROWS - local eye-normalized metrics
        # ------------------------------------------------------------------

        left_eye_w = self._eye_width_left(lms)
        right_eye_w = self._eye_width_right(lms)

        left_eye_y = self._mean_y(lms, [159, 145])
        right_eye_y = self._mean_y(lms, [386, 374])

        left_brow_outer_y = self._mean_y(lms, [70, 63])
        left_brow_mid_y = self._mean_y(lms, [105, 66])
        left_brow_inner_y = self._mean_y(lms, [107])

        right_brow_inner_y = self._mean_y(lms, [336])
        right_brow_mid_y = self._mean_y(lms, [296, 334])
        right_brow_outer_y = self._mean_y(lms, [293, 300])

        left_outer_metric = (left_eye_y - left_brow_outer_y) / left_eye_w
        right_outer_metric = (right_eye_y - right_brow_outer_y) / right_eye_w

        left_inner_metric = (left_eye_y - left_brow_inner_y) / left_eye_w
        right_inner_metric = (right_eye_y - right_brow_inner_y) / right_eye_w
        inner_metric = 0.5 * (left_inner_metric + right_inner_metric)

        left_mid_metric = (left_eye_y - left_brow_mid_y) / left_eye_w
        right_mid_metric = (right_eye_y - right_brow_mid_y) / right_eye_w

        def shape_brow_up(v):
            # Rimuove micro-drift idle.
            deadzone = 0.045
        
            if v < deadzone:
                return 0.0
        
            # Rimappa da deadzone..1 a 0..1.
            v = (v - deadzone) / (1.0 - deadzone)
        
            # Curva leggermente espansiva.
            # 1.15 mantiene responsive, ma non esplode come lineare + gain alto.
            return max(0.0, min(1.0, v ** 1.15))
        
        
        brow_outer_l = self._calibrated_delta(
            "browOuterUpLeft_lm",
            left_outer_metric,
            True,
            gain=1.95,
        )
        
        brow_outer_r = self._calibrated_delta(
            "browOuterUpRight_lm",
            right_outer_metric,
            True,
            gain=1.95,
        )
        
        brow_inner = self._calibrated_delta(
            "browInnerUp_lm",
            inner_metric,
            True,
            gain=2.05,
        )
        
        out["browOuterUpLeft"] = shape_brow_up(brow_outer_l)
        out["browOuterUpRight"] = shape_brow_up(brow_outer_r)
        out["browInnerUp"] = shape_brow_up(brow_inner)

        out["browDownLeft"] = self._calibrated_delta(
            "browDownLeft_lm", left_mid_metric, False, gain=2.2
        )
        out["browDownRight"] = self._calibrated_delta(
            "browDownRight_lm", right_mid_metric, False, gain=2.2
        )

        # ------------------------------------------------------------------
        # EYE WIDE - local 3D eye-opening normalized by local eye width
        # ------------------------------------------------------------------

        left_eye_wi = self._dist3(lms, 33, 133)
        right_eye_wi = self._dist3(lms, 362, 263)

        left_eye_open = self._dist3(lms, 159, 145) / max(left_eye_wi, 1e-6)
        right_eye_open = self._dist3(lms, 386, 374) / max(right_eye_wi, 1e-6)

        wide_l = self._calibrated_delta(
            "eyeWideLeft_lm",
            left_eye_open,
            True,
            gain=3.0,
        )
        wide_r = self._calibrated_delta(
            "eyeWideRight_lm",
            right_eye_open,
            True,
            gain=3.0,
        )

        wide_deadzone = 0.14

        if wide_l < wide_deadzone:
            wide_l = 0.0
        else:
            wide_l = (wide_l - wide_deadzone) / (1.0 - wide_deadzone)

        if wide_r < wide_deadzone:
            wide_r = 0.0
        else:
            wide_r = (wide_r - wide_deadzone) / (1.0 - wide_deadzone)

        out["eyeWideLeft"] = max(0.0, min(1.0, wide_l))
        out["eyeWideRight"] = max(0.0, min(1.0, wide_r))
        
        # ------------------------------------------------------------------
        # NOSE SNEER - local upper-lip-to-nose distance
        # ------------------------------------------------------------------

        local_face_scale = self._local_face_scale(lms)

        left_nose_y = self._mean_y(lms, [49, 98])
        right_nose_y = self._mean_y(lms, [279, 327])

        left_upper_lip_y = self._mean_y(lms, [40, 80, 81])
        right_upper_lip_y = self._mean_y(lms, [270, 310, 311])

        left_nose_lip_dist = (left_upper_lip_y - left_nose_y) / local_face_scale
        right_nose_lip_dist = (right_upper_lip_y - right_nose_y) / local_face_scale

        out["noseSneerLeft"] = self._calibrated_delta(
            "noseSneerLeft_lm", left_nose_lip_dist, False, gain=5.0
        )
        out["noseSneerRight"] = self._calibrated_delta(
            "noseSneerRight_lm", right_nose_lip_dist, False, gain=5.0
        )

        # ------------------------------------------------------------------
        # MOUTH / CHEEKS derived from landmarks
        # ------------------------------------------------------------------

        # Reference points
        left_mouth = 61
        right_mouth = 291
        upper_lip = 13
        lower_lip = 14
        nose_tip = 1

        mouth_width = self._dist(lms, left_mouth, right_mouth)
        if mouth_width < 1e-6:
            mouth_width = scale

        mouth_width_norm = mouth_width / scale

        left_corner_x, left_corner_y = self._xy(lms, left_mouth)
        right_corner_x, right_corner_y = self._xy(lms, right_mouth)

        mouth_center_x = (left_corner_x + right_corner_x) * 0.5
        mouth_center_y = (left_corner_y + right_corner_y) * 0.5

        # ------------------------------------------------------------------
        # MOUTH OPEN - local mouth-width-normalized
        # Use for logging/debug; avoid override if head pose causes artifacts.
        # ------------------------------------------------------------------

        mouth_w = self._dist3(lms, 61, 291)
        mouth_open = self._dist3(lms, 13, 14) / max(mouth_w, 1e-6)

        pitch = 0.0
        head_down_amount = 0.0

        if getattr(tracking, "facial_matrix", None) is not None:
            try:
                import math
                M = tracking.facial_matrix
                pitch = math.atan2(
                    -M[1][2],
                    math.sqrt(M[1][0] ** 2 + M[1][1] ** 2)
                )

                centered_pitch = pitch - 0.09
                head_down_amount = max(0.0, centered_pitch)
            except Exception:
                head_down_amount = 0.0
        
        # Speech-specific signal: sensitive to small lip openings,
        # but shaped by a soft-knee curve so it does not explode.
        speech_raw = self._calibrated_delta(
            "speechOpen_lm",
            mouth_open,
            True,
            gain=2.35,
        )
        
        speech_open = self._speech_curve(speech_raw)
        raw_jaw = 0.0
        try:
            raw_jaw = float(tracking.blendshapes.get("jawOpen", 0.0))
        except Exception:
            raw_jaw = 0.0
        
        # Quando la testa è chinata, il landmark-derived mouth_open tende a collassare.
        # Usiamo raw MediaPipe come assist, non come valore dominante.
        if head_down_amount > 0.10:
            head_down_t = max(0.0, min(1.0, (head_down_amount - 0.10) / 0.30))
        
            raw_assist = self._speech_curve(raw_jaw * (1.4 + head_down_t * 0.9))
        
            speech_open = max(
                speech_open,
                raw_assist * (0.45 + 0.35 * head_down_t),
            )
        
        # Bigger jaw signal: slower and more conservative.
        big_jaw_raw = self._calibrated_delta(
            "jawOpen_lm",
            mouth_open,
            True,
            gain=1.75,
        )
        
        big_jaw_deadzone = 0.20
        
        if big_jaw_raw < big_jaw_deadzone:
            big_jaw = 0.0
        else:
            big_jaw = (big_jaw_raw - big_jaw_deadzone) / (1.0 - big_jaw_deadzone)
            big_jaw = big_jaw ** 1.45
        
        jaw = max(
            speech_open * 0.85,
            big_jaw,
            raw_jaw * 0.35
        )
        
        out["speechOpen"] = max(0.0, min(1.0, speech_open))
        out["jawOpen"] = max(0.0, min(0.75, jaw))

        # Smile/frown: corners moving up/down relative to neutral.
        # y smaller = up, y larger = down.
        left_corner_up_metric = -left_corner_y / scale
        right_corner_up_metric = -right_corner_y / scale

        out["mouthSmileLeft"] = self._calibrated_delta(
            "mouthSmileLeft_lm",
            left_corner_up_metric,
            True,
            gain=4.5,
        )
        out["mouthSmileRight"] = self._calibrated_delta(
            "mouthSmileRight_lm",
            right_corner_up_metric,
            True,
            gain=4.5,
        )

        left_corner_down_metric = left_corner_y / scale
        right_corner_down_metric = right_corner_y / scale

        out["mouthFrownLeft"] = self._calibrated_delta(
            "mouthFrownLeft_lm",
            left_corner_down_metric,
            True,
            gain=4.0,
        )
        out["mouthFrownRight"] = self._calibrated_delta(
            "mouthFrownRight_lm",
            right_corner_down_metric,
            True,
            gain=4.0,
        )

        # Stretch: mouth width increasing.
        out["mouthStretchLeft"] = self._calibrated_delta(
            "mouthStretchLeft_lm",
            mouth_width_norm,
            True,
            gain=4.0,
        )
        out["mouthStretchRight"] = self._calibrated_delta(
            "mouthStretchRight_lm",
            mouth_width_norm,
            True,
            gain=4.0,
        )


        # ------------------------------------------------------------------
        # O / U vowel proxy
        # ------------------------------------------------------------------
        # O: mouth is rounded AND visibly open -> mostly mouthFunnel.
        # U: mouth is rounded/narrow but not very open -> mostly mouthPucker.
        #
        # Note: true lip protrusion is hard from webcam landmarks, so this uses
        # mouth width narrowing + mouth opening as an approximation.
        
        mouth_narrow = self._calibrated_delta(
            "mouthNarrow_lm",
            mouth_width_norm,
            False,
            gain=4.2,
        )
        
        mouth_narrow = max(0.0, min(1.0, mouth_narrow))
        
        # speech_open should already be computed in the jawOpen section.
        # If not available for any reason, fallback safely.
        speech = float(out.get("speechOpen", 0.0))
        
        open_gate_o = self._smoothstep(0.08, 0.32, speech)
        narrow_gate = self._smoothstep(0.05, 0.35, mouth_narrow)
        
        # O shape: rounded + open.
        vowel_o = (
            mouth_narrow * 0.65 +
            speech * 0.55
        ) * open_gate_o * narrow_gate
        
        # U shape: rounded/narrow, but with lower opening.
        u_open_suppression = 1.0 - self._smoothstep(0.20, 0.48, speech)
        vowel_u = mouth_narrow * u_open_suppression
        
        vowel_o = max(0.0, min(1.0, vowel_o))
        vowel_u = max(0.0, min(1.0, vowel_u))
        
        # ARKit mapping:
        # O -> mostly funnel, some pucker
        # U -> mostly pucker, small funnel
        out["mouthFunnel"] = max(
            out.get("mouthFunnel", 0.0),
            vowel_o * 0.90,
            vowel_u * 0.30,
        )
        
        out["mouthPucker"] = max(
            out.get("mouthPucker", 0.0),
            vowel_u * 0.95,
            vowel_o * 0.35,
        )
        
        # O needs some jaw/lower opening, U should not force much jaw.
        out["jawOpen"] = max(
            out.get("jawOpen", 0.0),
            vowel_o * 0.28,
        )
        
        # Debug-only derived values.
        out["mouthNarrow"] = mouth_narrow
        out["vowelO"] = vowel_o
        out["vowelU"] = vowel_u

        # Mouth left/right: center displacement.
        out["mouthLeft"] = self._calibrated_delta(
            "mouthLeft_lm",
            mouth_center_x / scale,
            False,
            gain=3.0,
        )
        out["mouthRight"] = self._calibrated_delta(
            "mouthRight_lm",
            mouth_center_x / scale,
            True,
            gain=3.0,
        )

        # Upper/lower lip movement
        left_upper_lip_metric = -left_upper_lip_y / scale
        right_upper_lip_metric = -right_upper_lip_y / scale

        out["mouthUpperUpLeft"] = self._calibrated_delta(
            "mouthUpperUpLeft_lm",
            left_upper_lip_metric,
            True,
            gain=4.5,
        )
        out["mouthUpperUpRight"] = self._calibrated_delta(
            "mouthUpperUpRight_lm",
            right_upper_lip_metric,
            True,
            gain=4.5,
        )

        left_lower_lip_y = self._mean_y(lms, [84, 85, 86])
        right_lower_lip_y = self._mean_y(lms, [314, 315, 316])

        lower_down_l = self._calibrated_delta(
            "mouthLowerDownLeft_lm",
            left_lower_lip_y / scale,
            True,
            gain=3.2,
        )
        
        lower_down_r = self._calibrated_delta(
            "mouthLowerDownRight_lm",
            right_lower_lip_y / scale,
            True,
            gain=3.2,
        )
        
        # Speech proxy: when the mouth opens for speech, pull lower lip slightly down.
        out["mouthLowerDownLeft"] = max(lower_down_l, speech_open * 0.35)
        out["mouthLowerDownRight"] = max(lower_down_r, speech_open * 0.35)

        # ------------------------------------------------------------------
        # CHEEK / ZYGOMA - local lower-eye-to-cheek compression
        # ------------------------------------------------------------------

        left_cheek_y = self._mean_y(lms, [50, 101, 118, 119])
        right_cheek_y = self._mean_y(lms, [280, 330, 347, 348])

        left_lower_eye_y = self._mean_y(lms, [145, 153, 154])
        right_lower_eye_y = self._mean_y(lms, [374, 380, 381])

        left_cheek_eye_dist = (left_lower_eye_y - left_cheek_y) / left_eye_w
        right_cheek_eye_dist = (right_lower_eye_y - right_cheek_y) / right_eye_w

        out["cheekSquintLeft"] = self._calibrated_delta(
            "cheekSquintLeft_lm",
            left_cheek_eye_dist,
            False,
            gain=2.5,
        )
        out["cheekSquintRight"] = self._calibrated_delta(
            "cheekSquintRight_lm",
            right_cheek_eye_dist,
            False,
            gain=2.5,
        )

        # Cheek puff proxy:
        # Real cheek puff is hard from landmarks. Use pucker + increased cheek width/rounding.
        left_cheek_width = self._dist(lms, 50, 205) / scale
        right_cheek_width = self._dist(lms, 280, 425) / scale
        cheek_width = (left_cheek_width + right_cheek_width) * 0.5

        puff_from_width = self._calibrated_delta(
            "cheekPuff_lm",
            cheek_width,
            True,
            gain=3.5,
        )

        out["cheekPuff"] = max(
            puff_from_width,
            out.get("mouthPucker", 0.0) * 0.25,
            out.get("mouthFunnel", 0.0) * 0.25,
        )

        speech_keys = {
            "speechOpen",
            "jawOpen",
            "mouthLowerDownLeft",
            "mouthLowerDownRight",
            "mouthPucker",
            "mouthFunnel",
            "mouthStretchLeft",
            "mouthStretchRight",
            "mouthNarrow",
            "vowelO",
            "vowelU",
        }
        
        for k in list(out.keys()):
            if k in speech_keys:
                out[k] = self._smooth_asymmetric(
                    k,
                    out[k],
                    alpha_open=0.70,
                    alpha_close=0.78,
                )
            else:
                out[k] = self._smooth(k, out[k])
                
        return out
