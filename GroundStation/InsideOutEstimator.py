import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from VisionTarget.VisionTarget import VisionTarget

# Tag size in meters
TAG_SIZE = 0.055

# Camera mount offset from body origin (in body frame: Forward, Right, Down)
# Adjust these values based on your camera mount position
CAMERA_MOUNT_OFFSET = (0.0, 0.0, 0.0)  # (forward, right, down) in meters

# Camera mount rotation (in radians) - typically camera points forward
# For a forward-facing camera: roll=0, pitch=0, yaw=0 (or small adjustments)
CAMERA_MOUNT_ROLL = 0.0   # radians
CAMERA_MOUNT_PITCH = 0.0  # radians  
CAMERA_MOUNT_YAW = 0.0    # radians

# Desired tag position when docked (in body frame: Forward, Right, Down)
# This is where we want the tag to be relative to the body origin
DESIRED_TAG_POSITION = (1.0, 0.0, 0.0)  # 1 meter forward, centered, at body height

LINE_LENGTH = 5
CENTER_COLOR = (0, 255, 0)
CORNER_COLOR = (255, 0, 255)


class InsideOutEstimator:
    """
    Estimates tag pose relative to drone body frame using a camera mounted on the drone pointing forward.
    This is an inside-out estimator: camera is on the drone, tag is on the landing pad/docking station.
    """
    
    @staticmethod
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

    @staticmethod
    def plotText(image, center, color, text):
        center = (int(center[0]) + 4, int(center[1]) - 4)
        return cv2.putText(image, str(text), center, cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 3)

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        vision_target: VisionTarget,
        camera_mount_offset: Tuple[float, float, float] = CAMERA_MOUNT_OFFSET,
        camera_mount_rotation: Tuple[float, float, float] = (CAMERA_MOUNT_ROLL, CAMERA_MOUNT_PITCH, CAMERA_MOUNT_YAW),
    ) -> None:
        """
        Initialize inside-out pose estimator.
        
        Args:
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            camera_mount_offset: Camera position offset from body origin (Forward, Right, Down) in meters
            camera_mount_rotation: Camera mount rotation (roll, pitch, yaw) in radians
        """
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs
        self._vision_target = vision_target
        
        self._camera_mount_offset = np.array(camera_mount_offset, dtype=np.float32).reshape(3, 1)
        
        # Base rotation from OpenCV camera frame to body FRD frame
        # OpenCV: X right, Y down, Z forward
        # Body FRD: X forward, Y right, Z down
        # Transformation: Camera Z -> Body X, Camera X -> Body Y, Camera Y -> Body Z
        R_base = np.array([
            [0, 0, 1],  # Camera Z (forward) -> Body X (forward)
            [1, 0, 0],  # Camera X (right) -> Body Y (right)
            [0, 1, 0]   # Camera Y (down) -> Body Z (down)
        ], dtype=np.float32)
        
        # Apply additional mount rotation if specified
        roll, pitch, yaw = camera_mount_rotation
        if roll != 0 or pitch != 0 or yaw != 0:
            R_roll = self._rotation_matrix_x(roll)
            R_pitch = self._rotation_matrix_y(pitch)
            R_yaw = self._rotation_matrix_z(yaw)
            R_mount = R_yaw @ R_pitch @ R_roll
            self._R_body_cam = R_base @ R_mount
        else:
            self._R_body_cam = R_base

    @staticmethod
    def _rotation_matrix_x(angle: float) -> np.ndarray:
        """Rotation matrix around X axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)

    @staticmethod
    def _rotation_matrix_y(angle: float) -> np.ndarray:
        """Rotation matrix around Y axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

    @staticmethod
    def _rotation_matrix_z(angle: float) -> np.ndarray:
        """Rotation matrix around Z axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

    def _camera_to_body_frame(self, point_cam: np.ndarray) -> np.ndarray:
        """
        Transform a point from camera frame to body frame.
        
        Camera frame (OpenCV): X right, Y down, Z forward
        Body frame (FRD): X forward, Y right, Z down
        
        Args:
            point_cam: Point in camera frame (3x1)
            
        Returns:
            Point in body frame (3x1)
        """
        # First apply rotation, then add mount offset
        point_body = self._R_body_cam @ point_cam + self._camera_mount_offset
        return point_body

    def estimate(self, frame_bgr: np.ndarray, draw_on_frame: bool = False) -> Optional[Dict[str, Any]]:
        """
        Estimate tag pose relative to drone body frame from an OpenCV BGR frame.

        Returns a dict with keys:
          - success: bool
          - rvec_tag: (3,1) rotation vector of tag in camera frame (OpenCV convention)
          - tvec_tag: (3,1) translation vector of tag in camera frame (meters)
          - R_cam_tag: (3,3) rotation matrix of tag w.r.t. camera
          - tag_in_cam: (3,1) tag position expressed in camera frame (meters)
          - tag_in_body: (3,1) tag position expressed in body frame (meters, FRD)
          - docking_error: (3,1) error from desired tag position in body frame (FRD)
        Returns None if no tag is detected.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        pose_data = self._vision_target.get_position(frame_bgr)
        if pose_data is None:
            return None

        rvec = pose_data["rvec"]
        tvec = pose_data["tvec"]
        det = pose_data.get("detection")

        if draw_on_frame and det is not None:
            frame_bgr = InsideOutEstimator.plotPoint(frame_bgr, det.center, CENTER_COLOR)
            frame_bgr = InsideOutEstimator.plotText(frame_bgr, det.center, CENTER_COLOR, det.tag_id)
            for corner in det.corners:
                frame_bgr = InsideOutEstimator.plotPoint(frame_bgr, corner, CORNER_COLOR)

        # tvec is the tag position in camera frame
        # Transform from camera frame to body frame
        tag_in_cam = tvec  # Tag position in camera frame
        tag_in_body = self._camera_to_body_frame(tag_in_cam)  # Tag position in body frame

        # Compute rotation matrix
        R_cam_tag, _ = cv2.Rodrigues(rvec)  # Rotation from tag frame to camera frame

        # Extract xyz offsets in camera frame (meters)
        x_offset_cam = tvec[0][0]  # Right in camera frame
        y_offset_cam = tvec[1][0]  # Down in camera frame
        z_offset_cam = tvec[2][0]  # Forward in camera frame

        # Extract xyz offsets in body frame (meters, FRD)
        x_offset_body = tag_in_body[0][0]  # Forward in body frame
        y_offset_body = tag_in_body[1][0]  # Right in body frame
        z_offset_body = tag_in_body[2][0]  # Down in body frame

        # Calculate distance from camera to tag
        distance = np.sqrt(x_offset_cam**2 + y_offset_cam**2 + z_offset_cam**2)

        # Compute docking error (difference from desired tag position)
        desired_tag_pos = np.array(DESIRED_TAG_POSITION, dtype=np.float32).reshape(3, 1)
        docking_error = tag_in_body - desired_tag_pos

        # Convert to BODY_FRD format for MAVLink (same as body frame for this case)
        docking_error_BODY_FRD = docking_error.copy()

        # Convert rotation vector to degrees for display
        rvec_degrees = np.degrees(rvec.flatten())

        if draw_on_frame:
            # Display header
            cv2.putText(frame_bgr, "Tag Position (camera frame, m):", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display xyz offsets in camera frame
            cv2.putText(frame_bgr, f"Right: {x_offset_cam:.3f}m", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Down: {y_offset_cam:.3f}m", (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Forward: {z_offset_cam:.3f}m", (10, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Distance: {distance:.3f}m", (10, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Display tag position in body frame
            cv2.putText(frame_bgr, "Tag Position (body FRD, m):", (10, 220), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"Forward: {x_offset_body:.3f}m", (10, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame_bgr, f"Right: {y_offset_body:.3f}m", (10, 280), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame_bgr, f"Down: {z_offset_body:.3f}m", (10, 310), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Display docking error
            fwd_error, right_error, down_error = docking_error_BODY_FRD.flatten().tolist()
            cv2.putText(frame_bgr, "Docking Error (body FRD, m):", (10, 340), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"Forward error: {fwd_error:.3f}m", (10, 370), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.putText(frame_bgr, f"Right error: {right_error:.3f}m", (10, 400), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.putText(frame_bgr, f"Down error: {down_error:.3f}m", (10, 430), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            # Display rotation angles in degrees
            cv2.putText(frame_bgr, "Tag Orientation (degrees):", (10, 460), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"Roll: {rvec_degrees[0]:.1f}deg", (10, 490), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame_bgr, f"Pitch: {rvec_degrees[1]:.1f}deg", (10, 510), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame_bgr, f"Yaw: {rvec_degrees[2]:.1f}deg", (10, 530), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Draw coordinate axes at the tag center
            tag_size = getattr(self._vision_target, "tag_size", TAG_SIZE)
            cv2.drawFrameAxes(frame_bgr, self._camera_matrix, self._dist_coeffs, rvec, tvec, tag_size)

        return {
            "success": True,
            "rvec_tag": rvec,
            "tvec_tag": tvec,
            "R_cam_tag": R_cam_tag,
            "tag_in_cam": tag_in_cam,
            "tag_in_body": tag_in_body,
            "detection": det,
            "docking_error": docking_error_BODY_FRD,
            "distance": distance,
            "attitude": rvec_degrees,
        }

