export const expressionPresets = {
  SMILE: {
    mouthShrugLower: 0.279, mouthSmileLeft: 0.550, mouthSmileRight: 0.550,
    mouthUpperUpRight: 0.145, noseSneerLeft: 0.179, noseSneerRight: 0.179,
    mouthPressRight: 0.128, eyeWideLeft: 0.497, eyeWideRight: 0.497,
    jawOpen: 0.050, mouthDimpleLeft: 0.35, mouthDimpleRight: 0.35,
    browInnerUp: 0.503, browOuterUpLeft: 1.000, browOuterUpRight: 1.000,
    cheekSquintRight: 0.296,
  },
  PROHIBITED: {
    mouthShrugLower: 0.291, mouthSmileLeft: 0.240, mouthStretchLeft: 0.235,
    mouthStretchRight: 0.341, noseSneerLeft: 0.179, mouthFrownLeft: 0.313,
    mouthFunnel: 0.536, mouthLeft: 0.084, mouthPucker: 0.436,
    jawLeft: 0.179, jawRight: 0.285, eyeBlinkLeft: 0.274,
    eyeBlinkRight: 0.240, eyeSquintLeft: 0.151, browDownLeft: 0.682,
    browDownRight: 1.000, browOuterUpRight: 0.346,
  },
  CONCERNED: {
    mouthShrugUpper: 0.508, mouthRollLower: 0.123, eyeWideLeft: 1.000,
    eyeWideRight: 1.000, jawForward: 1.000, jawOpen: 0.073,
    mouthClose: 0.145, browInnerUp: 0.944, browOuterUpLeft: 1.000,
    browOuterUpRight: 1.000,
  },
  THOUGHTFUL: {
    mouthFrownRight: 0.648, mouthFunnel: 0.520, mouthLowerDownRight: 0.034,
    mouthRollLower: 0.223, eyeSquintRight: 0.592, jawForward: 0.179,
    eyeBlinkLeft: 0.112, browDownLeft: 0.888, cheekPuff: 0.413,
  },
  DISGUSTED: {
    mouthUpperUpRight: 0.492, noseSneerLeft: 0.447, noseSneerRight: 0.771,
    mouthFrownLeft: 0.196, mouthFrownRight: 0.358, eyeBlinkRight: 0.034,
    cheekSquintRight: 0.760,
  },
  WINK: {
    mouthSmileLeft: 0.358, mouthSmileRight: 0.749, noseSneerLeft: 0.492,
    eyeWideLeft: 0.123, mouthDimpleLeft: 0.028, mouthDimpleRight: 0.223,
    eyeBlinkLeft: 1.000, eyeLookDownLeft: 0.162, eyeLookInRight: 0.246,
    eyeLookUpRight: 0.263, eyeSquintLeft: 0.330, browOuterUpLeft: 0.475,
    browOuterUpRight: 0.804, cheekSquintRight: 0.726,
  },
} as const;
