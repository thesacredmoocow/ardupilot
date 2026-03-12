import cv2
import os
import numpy as np
from Camera.PiCameraFrameSource import PiCameraFrameSource
from VisionTarget.AprilVisionTarget import AprilVisionTarget
from Utils import tag_point_to_body, process_imu_packets
from Mavlink_publisher import MavlinkPublisher
from pymavlink import mavutil
from picamera2 import Picamera2
from libcamera import controls
from Camera.SimplePiCamera import setup_camera, get_camera_params
import time
import math
import SLAM.Orbslam as Orbslam
# target_point = np.array([0.01, 0.5, -0.547])
# target_point_with_alt = np.array([0.0, 0.0, -1.0])

target_point = np.array([0.01, 0.2, -1.0])
LANDING_TARGET_MESSAGE_OFFSET = 1.0 # meters

DEBUG_VECTORS = False
SEND_LANDING_TARGET = False

os.environ['MAVLINK20'] = '1'

json_path = "/home/raspi/ardupilot/GroundStation/Camera/PiCameraW/Calibration/camera_params.json"

# Calibration data output: frames in cam0/, timestamps in timestamps.txt
CALIB_CAM0_DIR = "/home/raspi/ardupilot/GroundStation/SLAM/calibration_data/cam0"
CALIB_TIMESTAMPS_PATH = "/home/raspi/ardupilot/GroundStation/SLAM/calibration_data/cam0/timestamps.txt"

mavlink = MavlinkPublisher(save_calib_data=True)


def main():
    mavlink.connect()
    time.sleep(1)
    # while not mavlink.request_local_position_ned_at_50hz():
    #     print("Requesting local position NED at 50 Hz")
    #     time.sleep(1)

    cameraMatrix, distCoeffs = get_camera_params(json_path)

    picam2 = setup_camera(size=(1536, 864), fps=30)
    start_time = time.time()
    
    mavlink.start_polling_thread(start_time)

    os.makedirs(CALIB_CAM0_DIR, exist_ok=True)
    with open(CALIB_TIMESTAMPS_PATH, "w") as ts_file:
        pass  # create/truncate timestamps file

    frame_count = 0
    while True:
        frame = picam2.capture_array()
        if frame is None:
            continue
        # Picamera2 returns RGB/RGBA; convert to BGR so saved PNG displays correctly
        if len(frame.shape) == 3:
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_time = time.time() - start_time
        timestamp_ns = int(frame_time * 1e9)
        small_frame = cv2.resize(frame, (768, 432))

        frame_path = os.path.join(CALIB_CAM0_DIR, f"{timestamp_ns}.png")
        cv2.imwrite(frame_path, small_frame)

        with open(CALIB_TIMESTAMPS_PATH, "a") as ts_file:
            ts_file.write(f"{timestamp_ns}\n")

        frame_count += 1


if __name__ == "__main__":
    main()
