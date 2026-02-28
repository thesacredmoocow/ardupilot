# from VisionTarget.VisionTarget import VisionTarget
from pupil_apriltags import Detector
from typing import Optional, Dict, Any, Tuple
import numpy as np
import cv2
import json
import time
from pathlib import Path
from picamera2 import Picamera2

TAG_OFFSETS_FILENAME = "tag_offsets.json"

class TagConfig:
    def __init__(self, tag_size: float, tag_id: int, position: Tuple[float, float, float], orientation: Tuple[float, float, float]):
        self.tag_size = tag_size
        self.tag_id = tag_id
        self.position = position
        self.orientation = orientation
        half_size = tag_size / 2.0
        self.obj_points = np.array([
            [-half_size, half_size, 0],
            [half_size, half_size, 0],
            [half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)

def get_tag_config(tag_id, tag_configs):
    for config in tag_configs:
        if config.tag_id == tag_id:
            return config
    return None

class PiCameraFrameSource:
    def __init__(self, size=(2304, 1296), fps=30):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": size, "format": "BGR888"},
            controls={"FrameRate": fps},
        )
        self.picam2.configure(config)
        self.picam2.start()
        # self.picam2.set_controls({"ExposureTime": 1})
        time.sleep(0.5)

    def read(self):
        rgb = self.picam2.capture_array()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return True, bgr

    def release(self):
        self.picam2.stop()


class AprilVisionTarget():
    # Stored offset from tag 11 to tag 123 in tag 11's frame (updated when both are visible).
    # Used to compute camera→tag11 when only tag 123 is visible.
    _t_123_in_11: Optional[np.ndarray] = None  # (3,) position of tag 123 origin in tag 11 frame
    _R_123_in_11: Optional[np.ndarray] = None    # (3,3) rotation from tag 11 to tag 123

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        offsets_path: Optional[Path] = "/home/raspi/ardupilot/GroundStation/VisionTarget/tag_offsets.json",
    ):
        self.tag_configs = [
            TagConfig(tag_size=0.128, tag_id=11, position=(0, 0, 0), orientation=(0, 0, 0)),
            TagConfig(tag_size=0.055, tag_id=123, position=(0, 0, 0), orientation=(0, 0, 0)),
        ]
        self.detector = Detector()
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
        self._t_123_in_11 = None
        self._R_123_in_11 = None
        self._offsets_path = Path(offsets_path) if offsets_path is not None else Path(__file__).resolve().parent / TAG_OFFSETS_FILENAME
        self._load_offsets()

    def _load_offsets(self) -> None:
        """Load tag 11→123 offset from JSON if the file exists."""
        if not self._offsets_path.exists():
            return
        try:
            with open(self._offsets_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            t = data.get("t_123_in_11")
            R = data.get("R_123_in_11")
            if t is not None and len(t) == 3:
                self._t_123_in_11 = np.array(t, dtype=np.float64)
            if R is not None and len(R) == 3 and all(len(row) == 3 for row in R):
                self._R_123_in_11 = np.array(R, dtype=np.float64)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Could not load tag offsets from {self._offsets_path}: {e}")

    def _save_offsets(self) -> None:
        """Write current tag 11→123 offset to JSON."""
        if self._t_123_in_11 is None or self._R_123_in_11 is None:
            return
        try:
            data = {
                "t_123_in_11": self._t_123_in_11.tolist(),
                "R_123_in_11": self._R_123_in_11.tolist(),
            }
            with open(self._offsets_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Tag offsets written to {self._offsets_path}")
        except OSError as e:
            print(f"Could not write tag offsets to {self._offsets_path}: {e}")

    def _offset_differs_more_than_10_percent(
        self,
        t_new: np.ndarray,
        R_new: np.ndarray,
    ) -> bool:
        """True if new offset differs from stored by more than 10%, or no stored offset."""
        if self._t_123_in_11 is None or self._R_123_in_11 is None:
            return True
        t_old = self._t_123_in_11
        R_old = self._R_123_in_11
        # Position: relative change in magnitude (avoid div by zero)
        norm_old = np.linalg.norm(t_old) + 1e-9
        rel_pos = np.linalg.norm(t_new - t_old) / norm_old
        if rel_pos > 0.10:
            return True
        # Rotation: angle (rad) between rotations
        R_diff = R_new @ R_old.T
        trace = np.trace(R_diff)
        trace = np.clip(trace, -1.0, 3.0)
        angle_rad = np.arccos((trace - 1.0) / 2.0)
        if angle_rad > 0.10:
            return True
        return False

    def detect(self, frame_bgr: np.ndarray):
        """Return list of AprilTag detections for the frame."""
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return self.detector.detect(frame_gray)

    def _solve_tag_pose(self, tag_id: int, corners: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (rvec, tvec) for tag in camera frame, or None."""
        cfg = get_tag_config(tag_id, self.tag_configs)
        if cfg is None:
            return None
        img_pts = np.array(corners, dtype=np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            cfg.obj_points, img_pts,
            self.camera_matrix, self.dist_coeffs,
        )
        if not ok:
            return None
        return (rvec, tvec)

    def get_position(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Return tvec and rvec from camera frame to tag id 11.
        When both tag 11 and 123 are visible, updates and prints the stored offset
        so that when only tag 123 is visible, camera→tag11 can still be computed.
        """
        detections = self.detect(frame_bgr)
        found = {d.tag_id: d for d in detections}
        has_11 = 11 in found
        has_123 = 123 in found

        # Case 1: Tag 11 visible → use it directly and optionally update offset from 123
        if has_11:
            rvec1, tvec1 = self._solve_tag_pose(11, found[11].corners)
            if rvec1 is None:
                return None
            if has_123:
                rvec2, tvec2 = self._solve_tag_pose(123, found[123].corners)
                if rvec2 is not None:
                    R1, _ = cv2.Rodrigues(rvec1)
                    R2, _ = cv2.Rodrigues(rvec2)
                    t1 = tvec1.reshape(3)
                    t2 = tvec2.reshape(3)
                    R_new = R1.T @ R2
                    t_new = R1.T @ (t2 - t1)
                    self._R_123_in_11 = R_new
                    self._t_123_in_11 = t_new
                    if not self._offsets_path.exists() or self._offset_differs_more_than_10_percent(t_new, R_new):
                        self._save_offsets()
                    else:
                        print(
                            "Offset 11→123 (in-memory) — position (m):",
                            self._t_123_in_11.tolist(),
                        )
            return {
                "rvec": rvec1,
                "tvec": tvec1,
                "detection": found[11],
            }

        # Case 2: Only tag 123 visible → use stored offset to get camera→tag11
        if has_123 and self._R_123_in_11 is not None and self._t_123_in_11 is not None:
            rvec2, tvec2 = self._solve_tag_pose(123, found[123].corners)
            if rvec2 is None:
                return None
            R2, _ = cv2.Rodrigues(rvec2)
            t2 = tvec2.reshape(3)
            # Camera→tag11: R1 = R2 @ R_123_in_11.T, t1 = t2 - R2 @ (R_123_in_11.T @ t_123_in_11)
            R1 = R2 @ self._R_123_in_11.T
            t1 = t2 - R2 @ (self._R_123_in_11.T @ self._t_123_in_11)
            rvec1, _ = cv2.Rodrigues(R1)
            tvec1 = t1.reshape(3, 1)
            return {
                "rvec": rvec1,
                "tvec": tvec1,
                "detection": found[123],
            }

        return None

# def main():
#     with PiCameraFrameSource(size=(2304, 1296), fps=30) as cam:
#         axis_length_m = 0.064 * 0.5
#         cameraMatrix = cam.get_camera_matrix()
#         distCoeffs = cam.get_dist_coeffs()
#         vision_target = AprilVisionTarget(camera_matrix=cameraMatrix, dist_coeffs=distCoeffs)
        
#         output_image_path = here / "apriltag_pose_annotated.jpg"
#         saved_annotated = False

#         # Display size (resize for faster drawing; full res can be heavy)
#         display_size = (960, 540)
#         window_name = "Tag detections"

#         while True:
#             time.sleep(0.3)
#             ret, frame = cam.read()
#             if not ret:
#                 break

#             detections = vision_target.detect(frame)
#             display = frame.copy()

#             # Draw all tag detections (corners + tag id)
#             for det in detections:
#                 corners = np.array(det.corners, dtype=np.int32)
#                 cv2.polylines(display, [corners.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
#                 center = tuple(np.array(det.center, dtype=np.int32))
#                 cv2.putText(
#                     display,
#                     f"id={det.tag_id}",
#                     (center[0] - 25, center[1] - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (0, 255, 0),
#                     2,
#                 )
#                 cv2.circle(display, center, 5, (0, 0, 255), -1)

#             result = vision_target.get_position(frame)
#             if result is not None:
#                 rvec = result["rvec"]
#                 tvec = result["tvec"]
#                 det = result["detection"]
#                 corners = np.array(det.corners, dtype=np.int32)
#                 center = tuple(np.array(det.center, dtype=np.int32))

#                 tx, ty, tz = float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0])
#                 rx, ry, rz = float(rvec[0][0]), float(rvec[1][0]), float(rvec[2][0])
#                 print(
#                     f"tag_id={det.tag_id} | "
#                     f"tvec: x={tx:.4f} m, y={ty:.4f} m, z={tz:.4f} m | "
#                     f"rvec: x={rx:.4f} rad, y={ry:.4f} rad, z={rz:.4f} rad"
#                 )

#                 cv2.drawFrameAxes(display, K, D, rvec, tvec, axis_length_m, 2)
#                 cv2.imwrite(str(output_image_path), display)

#             # Show resized frame in window
#             display_resized = cv2.resize(display, display_size)
#             cv2.imshow(window_name, display_resized)

#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#             time.sleep(0.03)

#         cam.release()
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()