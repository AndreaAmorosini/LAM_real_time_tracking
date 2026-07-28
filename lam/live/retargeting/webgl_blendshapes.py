from lam.live.retargeting.webgl_tuning import (
    JAW_TUNING,
    STABLE_MOUTH_TUNING,
    MOUTH_CONFLICT_TUNING,
)


ARKIT_BLENDSHAPE_NAMES = [
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


SPEECH_DERIVED_OVERRIDE_NAMES = {
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPucker",
    "mouthFunnel",
    "mouthStretchLeft",
    "mouthStretchRight",
}


LANDMARK_DERIVED_OVERRIDE_NAMES = {
    "browOuterUpLeft",
    "browOuterUpRight",
    "browInnerUp",
    "browDownLeft",
    "browDownRight",

    "noseSneerLeft",
    "noseSneerRight",

    "cheekSquintLeft",
    "cheekSquintRight",
    "cheekPuff",

    "eyeWideLeft",
    "eyeWideRight",
    "jawOpen",

    *SPEECH_DERIVED_OVERRIDE_NAMES,
}


WEBGL_ALL_VALID_BLENDSHAPES = set(ARKIT_BLENDSHAPE_NAMES)


def _clamp01(v):
    return float(max(0.0, min(1.0, float(v))))


def _apply_keep(b, keep):
    for k in list(b.keys()):
        if k not in keep:
            b[k] = 0.0
    return b


def _apply_deadzones(b, deadzones):
    for k, dz in deadzones.items():
        if abs(b.get(k, 0.0)) < dz:
            b[k] = 0.0
    return b


def _apply_gains(b, gains):
    for k, g in gains.items():
        b[k] = b.get(k, 0.0) * g
    return b


def _clamp_all(b):
    for k in list(b.keys()):
        b[k] = _clamp01(b[k])
    return b


def _apply_common_conflicts(b):
    tuning = MOUTH_CONFLICT_TUNING

    jaw = b.get("jawOpen", 0.0)

    if jaw > tuning.mouthclose_jaw_threshold:
        reduction = max(
            0.0,
            min(
                1.0,
                (jaw - tuning.mouthclose_jaw_threshold)
                / tuning.mouthclose_jaw_reduction_range,
            ),
        )
        b["mouthClose"] *= 1.0 - reduction

    pucker = b.get("mouthPucker", 0.0)
    funnel = b.get("mouthFunnel", 0.0)
    ratio = tuning.pucker_funnel_dominance_ratio

    if pucker > funnel * ratio:
        b["mouthFunnel"] *= tuning.pucker_funnel_weaker_scale
    elif funnel > pucker * ratio:
        b["mouthPucker"] *= tuning.pucker_funnel_weaker_scale

    rounding = max(
        b.get("mouthPucker", 0.0),
        b.get("mouthFunnel", 0.0),
    )

    if rounding > tuning.rounding_mouthclose_threshold:
        reduction = max(
            0.0,
            min(1.0, rounding / tuning.rounding_mouthclose_range),
        )
        b["mouthClose"] *= 1.0 - reduction * tuning.rounding_mouthclose_scale

    b["mouthCheekPuff"] = max(
        b.get("mouthCheekPuff", 0.0),
        b.get("cheekPuff", 0.0),
    )

    nose_l = b.get("noseSneerLeft", 0.0)
    nose_r = b.get("noseSneerRight", 0.0)

    b["cheekSquintLeft"] = max(b.get("cheekSquintLeft", 0.0), nose_l * 0.8)
    b["cheekSquintRight"] = max(b.get("cheekSquintRight", 0.0), nose_r * 0.8)

    return b


def postprocess_webgl_blendshapes_raw_debug(b):
    _apply_keep(b, WEBGL_ALL_VALID_BLENDSHAPES)
    _apply_deadzones(b, {k: 0.01 for k in WEBGL_ALL_VALID_BLENDSHAPES})
    _apply_common_conflicts(b)
    _clamp_all(b)
    return b


def postprocess_webgl_blendshapes_stable_live(b):
    mouth = STABLE_MOUTH_TUNING

    _apply_keep(b, WEBGL_ALL_VALID_BLENDSHAPES)

    deadzones = {
        "browDownLeft": 0.035,
        "browDownRight": 0.035,
        "browInnerUp": 0.005,
        "browOuterUpLeft": 0.005,
        "browOuterUpRight": 0.005,

        "cheekPuff": 0.05,
        "cheekSquintLeft": 0.035,
        "cheekSquintRight": 0.035,
        "noseSneerLeft": 0.005,
        "noseSneerRight": 0.005,

        "eyeBlinkLeft": 0.025,
        "eyeBlinkRight": 0.025,
        "eyeLookDownLeft": 0.05,
        "eyeLookDownRight": 0.05,
        "eyeLookInLeft": 0.05,
        "eyeLookInRight": 0.05,
        "eyeLookOutLeft": 0.05,
        "eyeLookOutRight": 0.05,
        "eyeLookUpLeft": 0.05,
        "eyeLookUpRight": 0.05,
        "eyeSquintLeft": 0.035,
        "eyeSquintRight": 0.035,
        "eyeWideLeft": 0.015,
        "eyeWideRight": 0.015,

        "jawForward": 0.06,
        "jawLeft": 0.05,
        "jawRight": 0.05,
        "jawOpen": JAW_TUNING.stable_deadzone,

        "mouthClose": 0.05,
        "mouthDimpleLeft": 0.035,
        "mouthDimpleRight": 0.035,
        "mouthFrownLeft": 0.04,
        "mouthFrownRight": 0.04,
        "mouthFunnel": mouth.funnel_deadzone,
        "mouthLeft": 0.05,
        "mouthRight": 0.05,
        "mouthLowerDownLeft": mouth.lower_down_deadzone,
        "mouthLowerDownRight": mouth.lower_down_deadzone,
        "mouthPressLeft": 0.05,
        "mouthPressRight": 0.05,
        "mouthPucker": mouth.pucker_deadzone,
        "mouthRollLower": 0.05,
        "mouthRollUpper": 0.05,
        "mouthShrugLower": 0.05,
        "mouthShrugUpper": 0.05,
        "mouthSmileLeft": 0.025,
        "mouthSmileRight": 0.025,
        "mouthStretchLeft": mouth.stretch_deadzone,
        "mouthStretchRight": mouth.stretch_deadzone,
        "mouthUpperUpLeft": 0.04,
        "mouthUpperUpRight": 0.04,
    }

    gains = {
        "browDownLeft": 2.2,
        "browDownRight": 2.2,
        "browInnerUp": 3.0,
        "browOuterUpLeft": 3.0,
        "browOuterUpRight": 3.0,

        "cheekPuff": 0.7,
        "cheekSquintLeft": 2.0,
        "cheekSquintRight": 2.0,
        "noseSneerLeft": 3.0,
        "noseSneerRight": 3.0,

        "eyeBlinkLeft": 1.1,
        "eyeBlinkRight": 1.1,
        "eyeLookDownLeft": 0.35,
        "eyeLookDownRight": 0.35,
        "eyeLookInLeft": 0.35,
        "eyeLookInRight": 0.35,
        "eyeLookOutLeft": 0.35,
        "eyeLookOutRight": 0.35,
        "eyeLookUpLeft": 0.35,
        "eyeLookUpRight": 0.35,
        "eyeSquintLeft": 0.8,
        "eyeSquintRight": 0.8,
        "eyeWideLeft": 2.0,
        "eyeWideRight": 2.0,

        "jawForward": 0.30,
        "jawLeft": 0.45,
        "jawRight": 0.45,
        "jawOpen": JAW_TUNING.stable_gain,

        "mouthClose": 0.40,
        "mouthDimpleLeft": 1.0,
        "mouthDimpleRight": 1.0,
        "mouthFrownLeft": 0.9,
        "mouthFrownRight": 0.9,
        "mouthFunnel": mouth.funnel_gain,
        "mouthLeft": 0.65,
        "mouthRight": 0.65,
        "mouthLowerDownLeft": mouth.lower_down_gain,
        "mouthLowerDownRight": mouth.lower_down_gain,
        "mouthPressLeft": 0.55,
        "mouthPressRight": 0.55,
        "mouthPucker": mouth.pucker_gain,
        "mouthRollLower": 0.45,
        "mouthRollUpper": 0.45,
        "mouthShrugLower": 0.55,
        "mouthShrugUpper": 0.55,
        "mouthSmileLeft": 1.20,
        "mouthSmileRight": 1.20,
        "mouthStretchLeft": mouth.stretch_gain,
        "mouthStretchRight": mouth.stretch_gain,
        "mouthUpperUpLeft": 0.70,
        "mouthUpperUpRight": 0.70,
    }

    _apply_deadzones(b, deadzones)
    _apply_gains(b, gains)

    jaw = b.get("jawOpen", 0.0)
    if jaw < JAW_TUNING.min_value:
        b["jawOpen"] = 0.0

    b["jawOpen"] = min(b.get("jawOpen", 0.0), JAW_TUNING.cap)

    _apply_common_conflicts(b)
    _clamp_all(b)
    return b


def postprocess_webgl_blendshapes_expressive_live(b):
    # Per ora mantieni expressive come stable + valori meno repressivi,
    # oppure sposta qui la funzione expressive attuale invariata.
    return postprocess_webgl_blendshapes_stable_live(b)


def postprocess_webgl_blendshapes(b, mode="stable"):
    if mode == "raw":
        return postprocess_webgl_blendshapes_raw_debug(b)

    if mode == "expressive":
        return postprocess_webgl_blendshapes_expressive_live(b)

    return postprocess_webgl_blendshapes_stable_live(b)


def build_webgl_payload(
    tracking,
    status: str,
    fps_capture: float,
    fps_sent: float,
    mapping_mode: str = "stable",
    landmark_derived=None,
    head_reliability=None,
):
    blendshapes = {name: 0.0 for name in ARKIT_BLENDSHAPE_NAMES}

    if tracking is not None and tracking.detected:
        for name, value in tracking.blendshapes.items():
            if name in blendshapes:
                blendshapes[name] = float(value)

    reliability = {
        "jaw_eye": 1.0,
        "derived": 1.0,
        "pose_amount": 0.0,
        "motion_amount": 0.0,
    }

    if head_reliability is not None:
        reliability = head_reliability.update(tracking)

    derived_blendshapes = {}

    if landmark_derived is not None:
        derived_blendshapes = landmark_derived.derive(tracking)

        jaw_eye_factor = float(reliability.get("jaw_eye", 1.0))
        derived_factor = float(reliability.get("derived", 1.0))

        for name, value in derived_blendshapes.items():
            if name not in blendshapes:
                continue

            if name not in LANDMARK_DERIVED_OVERRIDE_NAMES:
                continue

            raw_value = float(blendshapes.get(name, 0.0))
            derived_value = float(value)

            if name == "jawOpen":
                factor = float(reliability.get("mouth", jaw_eye_factor))
                head_down = float(reliability.get("head_down", 0.0))
                head_down_boost = 1.0

                if head_down > 0.12:
                    head_down_boost = 1.0 + min(0.75, (head_down - 0.12) / 0.28 * 0.75)

                if factor <= 0.05:
                    jaw = raw_value
                else:
                    derived_shaped = max(0.0, min(1.0, derived_value)) ** 1.35
                    derived_weight = JAW_TUNING.derived_weight * factor
                    jaw = raw_value * (1.0 - derived_weight) + derived_shaped * derived_weight
                    jaw = max(raw_value * 0.95, jaw)

                jaw *= head_down_boost

                if jaw < JAW_TUNING.min_value:
                    jaw = 0.0

                jaw = min(jaw, JAW_TUNING.cap)
                blendshapes["jawOpen"] = jaw
                continue

            if name in {"eyeWideLeft", "eyeWideRight"}:
                factor = jaw_eye_factor

                if factor <= 0.05:
                    wide = raw_value
                else:
                    wide = raw_value * (1.0 - factor) + derived_value * factor

                if wide < 0.08:
                    wide = 0.0

                blendshapes[name] = min(wide, 0.85)
                continue

            if name in {
                "browInnerUp",
                "browOuterUpLeft",
                "browOuterUpRight",
                "browDownLeft",
                "browDownRight",
            }:
                brow_factor = float(reliability.get("brows", derived_factor))

                head_up_amount = float(reliability.get("head_up", 0.0))

                head_up_gate = 1.0
                if head_up_amount > 0.08:
                    head_up_gate = max(0.0, min(1.0, 1.0 - (head_up_amount - 0.08) / 0.22))

                brow_factor = min(brow_factor, head_up_gate)
            
                derived_brow = derived_value * brow_factor
            
                # Also soften raw MediaPipe brow-up when pose is bad,
                # because MediaPipe raw can drift with head pitch too.
                if name in {
                    "browInnerUp",
                    "browOuterUpLeft",
                    "browOuterUpRight",
                }:
                    raw_pose_scale = 0.20 + 0.80 * brow_factor
                    raw_brow = raw_value * raw_pose_scale
                else:
                    raw_brow = raw_value
            
                # Do not use plain max(raw, derived) for brows.
                if head_up_amount > 0.10:
                    value_out = max(
                        raw_brow * 0.70,
                        raw_brow * 0.65 + derived_brow * 0.35
                    )
                else:
                    value_out = max(
                        raw_brow,
                        raw_brow * 0.45 + derived_brow * 0.55,
                        derived_brow * 0.85
                    )
            
                # When pose is unreliable, cap false eyebrow raise.
                if head_up_amount > 0.10 and name in {
                    "browInnerUp",
                    "browOuterUpLeft",
                    "browOuterUpRight",
                }:
                    value_out = min(value_out, 0.08 + 0.20 * brow_factor)
            
                if head_up_amount > 0.10 and name in {
                    "browInnerUp",
                    "browOuterUpLeft",
                    "browOuterUpRight",
                }:
                    blendshapes[name] = min(value_out, 0.30)
                else:
                    blendshapes[name] = min(value_out, 0.65)
                    
                continue


            if name in SPEECH_DERIVED_OVERRIDE_NAMES:
                mouth_factor = float(
                    reliability.get(
                        "mouth",
                        max(0.0, min(1.0, 0.65 * jaw_eye_factor + 0.35 * derived_factor)),
                    )
                )

                derived_mouth = derived_value * mouth_factor

                head_down = float(reliability.get("head_down", 0.0))
                if head_down > 0.12:
                    speech_boost = 1.0 + min(0.60, (head_down - 0.12) / 0.28 * 0.60)
                    derived_mouth *= speech_boost

                if name in {"mouthLowerDownLeft", "mouthLowerDownRight"}:
                    value_out = max(
                        raw_value,
                        raw_value * 0.55 + derived_mouth * 0.45,
                        derived_mouth * 0.70,
                    )
                    blendshapes[name] = min(value_out, 0.45)
                    continue

                if name == "mouthPucker":
                    value_out = max(
                        raw_value,
                        raw_value * 0.45 + derived_mouth * 0.55,
                        derived_mouth * 0.90,
                    )
                    blendshapes[name] = min(value_out, 0.75)
                    continue

                if name == "mouthFunnel":
                    value_out = max(
                        raw_value,
                        raw_value * 0.45 + derived_mouth * 0.55,
                        derived_mouth * 0.90,
                    )
                    blendshapes[name] = min(value_out, 0.70)
                    continue

                if name in {"mouthStretchLeft", "mouthStretchRight"}:
                    value_out = max(raw_value, derived_mouth * 0.75)
                    blendshapes[name] = min(value_out, 0.55)
                    continue

            derived_value *= derived_factor

            if derived_factor <= 0.05:
                blendshapes[name] = raw_value
            else:
                blendshapes[name] = max(raw_value, derived_value)

    blendshapes["mouthCheekPuff"] = blendshapes.get("cheekPuff", 0.0)
    blendshapes = postprocess_webgl_blendshapes(blendshapes, mode=mapping_mode)

    facial_matrix = None
    if tracking is not None and tracking.facial_matrix is not None:
        facial_matrix = tracking.facial_matrix.tolist()

    return {
        "type": "webgl_frame",
        "detected": bool(tracking is not None and tracking.detected),
        "blendshapes": blendshapes,
        "facial_matrix": facial_matrix,
        "fps_capture": fps_capture,
        "fps_sent": fps_sent,
        "mapping_mode": mapping_mode,
        "status": status,
        "derived_blendshapes": derived_blendshapes,
        "reliability": reliability,
    }
