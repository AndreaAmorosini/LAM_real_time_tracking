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
    def __init__(self, alpha=0.45, deadzone=0.015):
        self.filters = {}
        self.alpha = alpha
        self.deadzone = deadzone

    def smooth(self, blendshapes: dict) -> dict:
        out = {}

        for k, v in blendshapes.items():
            if abs(v) < self.deadzone:
                v = 0.0

            if k not in self.filters:
                self.filters[k] = EMAFilter(self.alpha)

            out[k] = float(self.filters[k](v))

        return out

def main():
    s = BlendshapeSmoother()
    print(s.smooth({"jawOpen": 0.1}))
    print(s.smooth({"jawOpen": 0.9}))

if __name__ == "__main__":
    main()