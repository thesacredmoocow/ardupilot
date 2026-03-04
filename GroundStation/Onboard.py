import cv2
import os
import numpy as np
from Camera.PiCameraFrameSource import PiCameraFrameSource
from VisionTarget.AprilVisionTarget import AprilVisionTarget
from Utils import tag_point_to_body
from Mavlink_publisher import MavlinkPublisher
from pymavlink import mavutil
from picamera2 import Picamera2
from libcamera import controls
from Camera.SimplePiCamera import setup_camera, get_camera_params
import time
import math
target_point = np.array([0.01, 0.5, -0.547])
target_point_with_alt = np.array([0.0, 0.0, -1.0])

DEBUG_VECTORS = False
SEND_LANDING_TARGET = False

os.environ['MAVLINK20'] = '1'

json_path = "/home/raspi/ardupilot/GroundStation/Camera/PiCameraW/Calibration/camera_params.json"

mavlink = MavlinkPublisher()
def main():
    mavlink.connect()
    time.sleep(1)
    while not mavlink.request_local_position_ned_at_50hz():
        print("Requesting local position NED at 50 Hz")
        time.sleep(1)

    cameraMatrix, distCoeffs = get_camera_params(json_path)
    print(cameraMatrix)
    print(distCoeffs)
    
    vision_target = AprilVisionTarget(camera_matrix=cameraMatrix, dist_coeffs=distCoeffs)

    # picam2 = setup_camera(size=(2304, 1296), fps=30)
    picam2 = setup_camera(size=(1536, 864), fps=120)
    start_time = time.time()
    frame_count = 0
    fps_t_start = time.perf_counter()
    while True:
        allow_send_override = False
        target_yaw_rate = 0.0
        target_throttle_rate = 0.0
        frame = picam2.capture_array()

        if frame is None:
            continue
        frame_count += 1
        display = frame.copy()

        result = vision_target.get_position(frame)
        if result is not None:
            rvec = result["rvec"]
            tvec = result["tvec"]
            det = result["detection"]
            target_point_in_body = tag_point_to_body(tvec, rvec, target_point)

            alt_calc = tag_point_to_body(tvec, rvec, target_point_with_alt)
            yaw = math.atan2(tvec[0], tvec[2])
            if DEBUG_VECTORS:
                mavlink.publish_tvec_rvec("BDY_TAG", tvec, rvec)
                mavlink.publish_target_point(target_point_in_body)
            # print(tvec.flatten())
            mavlink.publish_landing_target(target_point_in_body, use_angles=False)
            target_yaw_rate = yaw * 0.5
            target_throttle_rate = max(-0.3, min(0.5, -alt_calc[1] * 2.0))

            if mavlink.is_rc7_high():
                allow_send_override = True

        elapsed = time.perf_counter() - fps_t_start
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            print(f"FPS: {fps:.1f}")
            frame_count = 0
            fps_t_start = time.perf_counter()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
