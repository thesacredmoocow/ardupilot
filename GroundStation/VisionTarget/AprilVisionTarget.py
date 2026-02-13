# from VisionTarget.VisionTarget import VisionTarget
from pupil_apriltags import Detector
from typing import Optional, Dict, Any
import numpy as np
import cv2
import json
import time
from pathlib import Path
from picamera2 import Picamera2

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
        # rgb = cv2.resize(rgb, (640, 480))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return True, bgr

    def release(self):
        self.picam2.stop()


class AprilVisionTarget():
    def __init__(self, tag_size: float, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        self.tag_size = tag_size
        self.detector = Detector()
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
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
        img_points = np.array(det.corners, dtype=np.float32)
        ok, rvec, tvec = cv2.solvePnP(self.obj_points, img_points, self.camera_matrix, self.dist_coeffs)
        if not ok:
            return None

        return {
            "rvec": rvec,
            "tvec": tvec,
            "detection": det,
        }


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    camera_params_path = here.parent / "camera_params.json"
    if not camera_params_path.exists():
        camera_params_path = here.parent / "Camera" / "camera_params.json"

    with open(camera_params_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        K = np.array(data["camera_matrix"], dtype=np.float32)
        D = np.array(data["dist_coeff"], dtype=np.float32)

    tag_size_m = 0.05
    axis_length_m = tag_size_m * 0.5
    vision_target = AprilVisionTarget(tag_size=tag_size_m, camera_matrix=K, dist_coeffs=D)
    cam = PiCameraFrameSource(size=(2304, 1296), fps=30)
    output_image_path = here / "apriltag_pose_annotated.jpg"
    saved_annotated = False

    while True:
        time.sleep(0.3)
        ret, frame = cam.read()
        if not ret:
            break

        result = vision_target.get_position(frame)
        if result is not None:
            rvec = result["rvec"]
            tvec = result["tvec"]
            det = result["detection"]
            corners = np.array(det.corners, dtype=np.int32)
            center = tuple(np.array(det.center, dtype=np.int32))

            tx, ty, tz = float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0])
            rx, ry, rz = float(rvec[0][0]), float(rvec[1][0]), float(rvec[2][0])
            print(
                f"tag_id={det.tag_id} | "
                f"tvec: x={tx:.4f} m, y={ty:.4f} m, z={tz:.4f} m | "
                f"rvec: x={rx:.4f} rad, y={ry:.4f} rad, z={rz:.4f} rad"
            )

            annotated = frame.copy()
            cv2.polylines(annotated, [corners.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            for i, p in enumerate(corners):
                cv2.circle(annotated, (int(p[0]), int(p[1])), 4, (255, 0, 0), -1)
                cv2.putText(
                    annotated,
                    str(i),
                    (int(p[0]) + 5, int(p[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 0),
                    1,
                )
            cv2.circle(annotated, center, 4, (0, 0, 255), -1)
            cv2.drawFrameAxes(annotated, K, D, rvec, tvec, axis_length_m, 2)

            cv2.imwrite(str(output_image_path), annotated)
            # if not saved_annotated:
            #     print(f"Saved annotated image: {output_image_path}")
            #     saved_annotated = True

            # cv2.imshow("AprilTag Pose", annotated)
        # else:
        #     cv2.imshow("AprilTag Pose", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.03)

    cam.release()
    cv2.destroyAllWindows()
