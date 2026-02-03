from IRDetector import UprightTPoseEstimator
from VisionTarget.VisionTarget import VisionTarget
import numpy as np
from typing import Optional, Dict, Any

class IRVisionTarget(VisionTarget):
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.ir_estimator = UprightTPoseEstimator(self.camera_matrix, self.dist_coeffs)

    def get_position(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        success, rvec, tvec = self.ir_estimator.process_frame(frame_bgr, 200)
        if not success:
            return None

        return {
            "rvec": rvec,
            "tvec": tvec,
        }