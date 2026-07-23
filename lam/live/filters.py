class EMAFilter:
    def __init__(self, alpha=0.45):
        self.alpha = alpha
        self.value = None

    def __call__(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value


class BlendshapeSmoother:
    def __init__(self, alpha=0.45, deadzone=0.01):
        self.filters = {}
        self.default_alpha = alpha
        self.default_deadzone = deadzone

    def _alpha_for_key(self, k):
        if k.startswith("mouth") or k.startswith("jaw"):
            return 0.62
        if k.startswith("eyeBlink"):
            return 0.75
        if k.startswith("eyeLook"):
            return 0.30
        if k.startswith("eyeSquint") or k.startswith("eyeWide"):
            return 0.50
        if k.startswith("brow"):
            return 0.55
        if k.startswith("nose") or k.startswith("cheek"):
            return 0.55
        return self.default_alpha

    def _deadzone_for_key(self, k):
        if k.startswith("mouth") or k.startswith("jaw"):
            return 0.008
        if k.startswith("brow"):
            return 0.004
        if k.startswith("nose") or k.startswith("cheek"):
            return 0.004
        if k.startswith("eyeBlink"):
            return 0.01
        if k.startswith("eyeLook"):
            return 0.025
        return self.default_deadzone

    def smooth(self, blendshapes: dict) -> dict:
        out = {}

        for k, v in blendshapes.items():
            dz = self._deadzone_for_key(k)
            if abs(v) < dz:
                v = 0.0

            if k not in self.filters:
                self.filters[k] = EMAFilter(self._alpha_for_key(k))

            out[k] = float(self.filters[k](v))

        return out
        
def main():
    s = BlendshapeSmoother()
    print(s.smooth({"jawOpen": 0.1}))
    print(s.smooth({"jawOpen": 0.9}))

if __name__ == "__main__":
    main()