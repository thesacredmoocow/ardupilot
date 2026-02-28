import math

import numpy as np
import cv2


def tag_point_to_body(
    tvec_tag_in_body: np.ndarray,
    rvec_tag_in_body: np.ndarray,
    point_A_in_tag: np.ndarray,
) -> np.ndarray:
    """
    Express point A (given in tag frame) in body frame.

    Args:
        tvec_tag_in_body: Position of tag origin in body frame (3,) or (3, 1).
        rvec_tag_in_body: Rotation vector of tag in body frame (OpenCV convention).
        point_A_in_tag: Position of point A in tag frame (3,) or (3, 1).

    Returns:
        tvec_to_A_in_body: Position of point A in body frame, shape (3,).
    """
    t = np.asarray(tvec_tag_in_body, dtype=np.float64).reshape(3)
    rvec = np.asarray(rvec_tag_in_body, dtype=np.float64).reshape(3, 1)
    p_tag = np.asarray(point_A_in_tag, dtype=np.float64).reshape(3)
    R_tag_to_body, _ = cv2.Rodrigues(rvec)
    p_body = R_tag_to_body @ p_tag + t
    return p_body


def quaternion_from_euler(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        sr * sp * cy - cr * cp * sy
    ]

