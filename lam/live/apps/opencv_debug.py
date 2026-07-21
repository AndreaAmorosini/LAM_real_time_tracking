import time
import cv2
import torch

from lam.live.live_motion import LiveMotionProvider
from lam.live.debug_draw import draw_face_landmarks, draw_blendshape_debug


def main():
    provider = LiveMotionProvider(
        device="cpu",
        # width=640,
        # height=480,
        # fps=30,
    )

    source_betas = torch.zeros(10)

    provider.open()

    frame_count = 0
    t0 = time.time()
    fps = 0.0

    try:
        while True:
            frame, flame_params, tracking, ok = provider.read(source_betas)

            if frame is None:
                continue

            debug = draw_face_landmarks(
                frame,
                tracking,
                draw_tesselation=False,
                draw_contours=True,
                draw_irises=True,
            )
            debug = draw_blendshape_debug(debug, tracking, max_items=8)

            frame_count += 1
            now = time.time()
            if now - t0 >= 1.0:
                fps = frame_count / (now - t0)
                frame_count = 0
                t0 = now

            cv2.putText(
                debug,
                f"FPS: {fps:.1f} detected={ok}",
                (20, debug.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # RGB -> BGR per OpenCV
            debug_bgr = cv2.cvtColor(debug, cv2.COLOR_RGB2BGR)
            cv2.imshow("LAM MediaPipe Live Debug", debug_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord("q")]:
                break

    finally:
        provider.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
