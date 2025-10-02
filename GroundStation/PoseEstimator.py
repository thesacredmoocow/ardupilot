import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pupil_apriltags import Detector

BODY_TO_TAG = (0, 0.13, 0.02)
TAG_SIZE = 0.055

LINE_LENGTH = 5
CENTER_COLOR = (0, 255, 0)
CORNER_COLOR = (255, 0, 255)

DOCKED_CAMERA_OFFSET = (0, -0.127, -0.107)

class PoseEstimator:
    """
    Estimates camera pose relative to a body frame using an AprilTag observed in an OpenCV frame.
    """
    def plotPoint(image, center, color):
        center = (int(center[0]), int(center[1]))
        image = cv2.line(image,
                        (center[0] - LINE_LENGTH, center[1]),
                        (center[0] + LINE_LENGTH, center[1]),
                        color,
                        3)
        image = cv2.line(image,
                        (center[0], center[1] - LINE_LENGTH),
                        (center[0], center[1] + LINE_LENGTH),
                        color,
                        3)
        return image

    # plot a little text
    def plotText(image, center, color, text):
        center = (int(center[0]) + 4, int(center[1]) - 4)
        return cv2.putText(image, str(text), center, cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 3)

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> None:
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs
        self._detector = Detector()

        # precompute tag corner object points (tag frame)
        half_size = TAG_SIZE / 2.0
        self._obj_points = np.array([
            [-half_size, half_size, 0],
            [half_size, half_size, 0],
            [half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)

    def estimate(self, frame_bgr: np.ndarray, draw_on_frame: bool = False) -> Optional[Dict[str, Any]]:
        """
        Estimate camera pose from an OpenCV BGR frame.

        Returns a dict with keys:
          - success: bool
          - rvec_tag: (3,1) rotation vector of tag in camera frame (OpenCV convention)
          - tvec_tag: (3,1) translation vector of tag in camera frame (meters)
          - R_cam_tag: (3,3) rotation matrix of camera w.r.t. tag
          - cam_in_tag: (3,1) camera position expressed in tag frame (meters)
          - cam_in_body: (3,1) camera position expressed in body frame (meters)
        Returns None if no tag is detected.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        detections = self._detector.detect(gray)
        if not detections:
            return None

        # take the first detection (extend if needed for multiple tags)
        det = detections[0]

        if draw_on_frame:
            frame_bgr = PoseEstimator.plotPoint(frame_bgr, det.center, CENTER_COLOR)
            frame_bgr = PoseEstimator.plotText(frame_bgr, det.center, CENTER_COLOR, det.tag_id)
            for corner in det.corners:
                frame_bgr = PoseEstimator.plotPoint(frame_bgr, corner, CORNER_COLOR)

        # solve pnp: tag corners (2d) to object points (3d in tag frame)
        ok, rvec, tvec = cv2.solvePnP(self._obj_points, det.corners, self._camera_matrix, self._dist_coeffs)
        if not ok:
            return None

        # compute camera position in tag frame: p_c^tag = -R^T * t
        R_cam_tag, _ = cv2.Rodrigues(rvec)
        cam_in_tag = -R_cam_tag.T @ tvec

        # camera position in body frame
        body_offset_tag = np.array(BODY_TO_TAG, dtype=np.float32).reshape(3, 1)
        cam_in_body = cam_in_tag - body_offset_tag
        tvec_body = tvec + R_cam_tag @ body_offset_tag

        # extract xyz offsets in meters
        x_offset = tvec[0][0]  # X offset in meters
        y_offset = tvec[1][0]  # Y offset in meters  
        z_offset = tvec[2][0]  # Z offset in meters
        
        # calculate distance from camera to marker
        distance = np.sqrt(x_offset**2 + y_offset**2 + z_offset**2)
        
        if draw_on_frame:
            # display header
            cv2.putText(frame_bgr, "Position (meters):", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # display xyz offsets in meters with better formatting
            cv2.putText(frame_bgr, f"X: {x_offset:.3f}m", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Y: {y_offset:.3f}m", (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Z: {z_offset:.3f}m", (10, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Distance: {distance:.3f}m", (10, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # display rotation angles in degrees
            rvec_degrees = np.degrees(rvec.flatten())
            cv2.putText(frame_bgr, "Orientation (degrees):", (10, 220), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"Roll: {rvec_degrees[0]:.1f}deg", (10, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame_bgr, f"Pitch: {rvec_degrees[1]:.1f}deg", (10, 270), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame_bgr, f"Yaw: {rvec_degrees[2]:.1f}deg", (10, 290), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            docking_error = (cam_in_body - np.array(DOCKED_CAMERA_OFFSET, dtype=np.float32).reshape(3, 1))
            cb_x, cb_y, cb_z = docking_error.flatten().tolist()

            docking_error_BODY_FRD = np.array((cb_y, -cb_x, cb_z), dtype=np.float32).reshape(3, 1)

            # display camera position in body frame
            cv2.putText(frame_bgr, "Camera in body frame (m):", (10, 325), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"Right error: {cb_x:.3f}m", (10, 355), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.putText(frame_bgr, f"Forward error: {cb_y:.3f}m", (10, 385), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.putText(frame_bgr, f"Down error: {cb_z:.3f}m", (10, 415), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            # draw coordinate axes at the body origin
            cv2.drawFrameAxes(frame_bgr, self._camera_matrix, self._dist_coeffs, rvec, tvec_body, TAG_SIZE)

        return {
            "success": True,
            "rvec_tag": rvec,
            "tvec_tag": tvec,
            "R_cam_tag": R_cam_tag,
            "cam_in_tag": cam_in_tag,
            "cam_in_body": cam_in_body,
            "detection": det,
            "tvec_body": tvec_body,
            "docking_error": docking_error_BODY_FRD,
            "distance": distance,
            "attitude": rvec_degrees,
        }


