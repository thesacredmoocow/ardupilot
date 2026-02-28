import cv2
import numpy as np
from Camera.PiCameraFrameSource import PiCameraFrameSource
from VisionTarget.AprilVisionTarget import AprilVisionTarget
from Utils import tag_point_to_body
from Mavlink_publisher import MavlinkPublisher

target_point = np.array([0.0, 0.0, -1.0])

mavlink = MavlinkPublisher()
def main():
    mavlink.connect()
    with PiCameraFrameSource(size=(2304, 1296), fps=30) as camera:
        cameraMatrix = camera.get_camera_matrix()
        distCoeffs = camera.get_dist_coeffs()
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
                    target_point_in_body = tag_point_to_body(tvec, rvec, target_point)
                    print(f"Target point in body frame (FRD offsets): Forward={target_point_in_body[2]:.3f} m, Right={target_point_in_body[0]:.3f} m, Down={target_point_in_body[1]:.3f} m")

                    mavlink.publish_tvec_rvec("BDY_TAG", tvec, rvec)
                    mavlink.publish_target_point(target_point_in_body)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
