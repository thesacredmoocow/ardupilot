"""
Example usage of the CameraSource base class and WebcamCameraSource implementation.
Demonstrates real-time camera display and drawing of tag id 11 position (camera → tag 11).
"""

import cv2
import numpy as np
from Camera.PiCameraFrameSource import PiCameraFrameSource
from VisionTarget.AprilVisionTarget import AprilVisionTarget

# Axis length in metres for drawing tag 11 pose
AXIS_LENGTH_M = 0.06

def main():
    print("Press 'q' to quit. Tag 11 position (rvec/tvec) is drawn when visible.")

    with PiCameraFrameSource(size=(2304, 1296), fps=30) as camera:
        cameraMatrix = camera.get_camera_matrix()
        distCoeffs = camera.get_dist_coeffs()
        if cameraMatrix is None or distCoeffs is None:
            print("No camera params; cannot run AprilTag pose.")
            return
        vision_target = AprilVisionTarget(camera_matrix=cameraMatrix, dist_coeffs=distCoeffs)

        while True:
            if camera.is_new_frame_available():
                frame = camera.get_latest_frame()
                if frame is None:
                    continue
                display = frame.copy()

                result = vision_target.get_position(frame)
                if result is not None:
                    rvec = result["rvec"]
                    tvec = result["tvec"]
                    det = result["detection"]
                    # Draw 3D axes at tag 11 position (camera → tag 11)
                    cv2.drawFrameAxes(
                        display,
                        np.array(cameraMatrix, dtype=np.float32),
                        np.array(distCoeffs, dtype=np.float32),
                        rvec,
                        tvec,
                        AXIS_LENGTH_M,
                        2,
                    )
                    # Draw detection outline (tag 11 or tag 123 if inferred)
                    corners = np.array(det.corners, dtype=np.int32)
                    cv2.polylines(display, [corners.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                    center = tuple(np.array(det.center, dtype=np.int32))
                    cv2.putText(
                        display,
                        f"tag {det.tag_id} (→11)",
                        (center[0] - 40, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("Frame", cv2.resize(display, (960, 540)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
