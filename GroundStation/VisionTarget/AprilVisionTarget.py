from VisionTarget.VisionTarget import VisionTarget
from pupil_apriltags import Detector
from typing import Optional, Dict, Any
import numpy as np
import cv2

class AprilVisionTarget(VisionTarget):
    def __init__(self, tag_size: float, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        self.tag_size = tag_size
        self.detector = Detector()
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        half_size = self.tag_size / 2.0
        self.obj_points = np.array([
            [-half_size, half_size, 0],
            [half_size, half_size, 0],
            [half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)

    def get_position(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(frame_gray)
        if not detections:
            return None

        # take the first detection (extend if needed for multiple tags)
        det = detections[0]
        ok, rvec, tvec = cv2.solvePnP(self.obj_points, det.corners, self.camera_matrix, self.dist_coeffs)
        if not ok:
            return None

        return {
            "rvec": rvec,
            "tvec": tvec,
            "detection": det,
        }


    