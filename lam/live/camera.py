# apre webcam
# - legge frame
# - restituisce RGB frame
# - gestisce fallback se webcam non disponibile


import cv2
import time


class WebcamSource:
    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam {self.camera_id}")

    def read_rgb(self):
        ok, frame_bgr = self.cap.read()
        if not ok:
            return None
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def close(self):
        if self.cap is not None:
            self.cap.release()


def main():
    cam = WebcamSource()
    cam.open()

    n = 0
    t0 = time.time()

    try:
        while n < 100:
            frame = cam.read_rgb()
            assert frame is not None
            print("frame", n, frame.shape, frame.dtype)
            n += 1
    finally:
        cam.close()

    print("fps:", n / max(time.time() - t0, 1e-6))


if __name__ == "__main__":
    main()
