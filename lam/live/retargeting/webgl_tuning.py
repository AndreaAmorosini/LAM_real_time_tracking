from dataclasses import dataclass
from lam.live.settings import settings


@dataclass(frozen=True)
class JawTuning:
    derived_weight: float = settings.jaw_derived_weight
    min_value: float = settings.jaw_min
    cap: float = settings.jaw_cap
    stable_deadzone: float = settings.stable_jaw_deadzone
    stable_gain: float = settings.stable_jaw_gain


@dataclass(frozen=True)
class SpeechTuning:
    speech_open_gain: float = settings.speech_open_gain
    speech_open_weight_to_jaw: float = settings.speech_open_weight_to_jaw
    big_jaw_gain: float = settings.big_jaw_gain
    big_jaw_deadzone: float = settings.big_jaw_deadzone
    big_jaw_power: float = settings.big_jaw_power
    landmark_jaw_cap: float = settings.landmark_jaw_cap


@dataclass(frozen=True)
class VowelTuning:
    mouth_narrow_gain: float = settings.mouth_narrow_gain
    vowel_o_funnel_gain: float = settings.vowel_o_funnel_gain
    vowel_o_pucker_gain: float = settings.vowel_o_pucker_gain
    vowel_o_jaw_gain: float = settings.vowel_o_jaw_gain
    vowel_u_pucker_gain: float = settings.vowel_u_pucker_gain
    vowel_u_funnel_gain: float = settings.vowel_u_funnel_gain


@dataclass(frozen=True)
class StableMouthTuning:
    funnel_deadzone: float = settings.stable_mouth_funnel_deadzone
    pucker_deadzone: float = settings.stable_mouth_pucker_deadzone
    lower_down_deadzone: float = settings.stable_mouth_lower_down_deadzone
    stretch_deadzone: float = settings.stable_mouth_stretch_deadzone

    funnel_gain: float = settings.stable_mouth_funnel_gain
    pucker_gain: float = settings.stable_mouth_pucker_gain
    lower_down_gain: float = settings.stable_mouth_lower_down_gain
    stretch_gain: float = settings.stable_mouth_stretch_gain


@dataclass(frozen=True)
class MouthConflictTuning:
    mouthclose_jaw_threshold: float = settings.mouthclose_jaw_threshold
    mouthclose_jaw_reduction_range: float = settings.mouthclose_jaw_reduction_range
    pucker_funnel_dominance_ratio: float = settings.pucker_funnel_dominance_ratio
    pucker_funnel_weaker_scale: float = settings.pucker_funnel_weaker_scale
    rounding_mouthclose_threshold: float = settings.rounding_mouthclose_threshold
    rounding_mouthclose_range: float = settings.rounding_mouthclose_range
    rounding_mouthclose_scale: float = settings.rounding_mouthclose_scale


JAW_TUNING = JawTuning()
SPEECH_TUNING = SpeechTuning()
VOWEL_TUNING = VowelTuning()
STABLE_MOUTH_TUNING = StableMouthTuning()
MOUTH_CONFLICT_TUNING = MouthConflictTuning()
