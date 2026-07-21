import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2


_FACE_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_TESSELATION
_FACE_CONTOURS = mp.solutions.face_mesh.FACEMESH_CONTOURS
_FACE_IRISES = mp.solutions.face_mesh.FACEMESH_IRISES

_DRAWING_UTILS = mp.solutions.drawing_utils
_DRAWING_STYLES = mp.solutions.drawing_styles


def _to_landmark_list(landmarks):
    landmark_list = landmark_pb2.NormalizedLandmarkList()

    for lm in landmarks:
        landmark = landmark_list.landmark.add()
        landmark.x = lm.x
        landmark.y = lm.y
        landmark.z = lm.z

    return landmark_list


def draw_face_landmarks(
    frame_rgb: np.ndarray,
    tracking_result,
    draw_tesselation=True,
    draw_contours=True,
    draw_irises=True,
):
    image = frame_rgb.copy()

    if tracking_result is None or not tracking_result.detected:
        cv2.putText(
            image,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return image

    landmark_list = _to_landmark_list(tracking_result.landmarks)

    if draw_tesselation:
        _DRAWING_UTILS.draw_landmarks(
            image=image,
            landmark_list=landmark_list,
            connections=_FACE_CONNECTIONS,
            landmark_drawing_spec=None,
            connection_drawing_spec=_DRAWING_STYLES.get_default_face_mesh_tesselation_style(),
        )

    if draw_contours:
        _DRAWING_UTILS.draw_landmarks(
            image=image,
            landmark_list=landmark_list,
            connections=_FACE_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=_DRAWING_STYLES.get_default_face_mesh_contours_style(),
        )

    if draw_irises:
        _DRAWING_UTILS.draw_landmarks(
            image=image,
            landmark_list=landmark_list,
            connections=_FACE_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=_DRAWING_STYLES.get_default_face_mesh_iris_connections_style(),
        )

    return image


def draw_blendshape_debug(frame_rgb: np.ndarray, tracking_result, max_items=12):
    image = frame_rgb.copy()

    if tracking_result is None or not tracking_result.detected:
        return image

    items = sorted(
        tracking_result.blendshapes.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )[:max_items]

    x = 20
    y = 30

    for name, value in items:
        text = f"{name}: {value:.3f}"

        cv2.putText(
            image,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        y += 22

    return image


def draw_full_debug(frame_rgb: np.ndarray, tracking_result):
    image = draw_face_landmarks(frame_rgb, tracking_result)
    image = draw_blendshape_debug(image, tracking_result)
    return image
