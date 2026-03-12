import cv2
import os
import subprocess
import numpy as np
from Utils import process_imu_packets
from Mavlink_publisher import MavlinkPublisher
from Camera.SimplePiCamera import setup_camera, get_camera_params
import time
import SLAM.Orbslam as Orbslam
import pyorbslam3

import socket
import imagezmq
from imageSender import init_image_sender, update_latest_frame
os.environ['MAVLINK20'] = '1'

json_path = "/home/raspi/ardupilot/GroundStation/Camera/PiCameraW/Calibration/camera_params.json"

mavlink = MavlinkPublisher()


def main():
    mavlink.connect()
    time.sleep(1)
    init_image_sender(name="slam", send_hz=10.0)

    cameraMatrix, distCoeffs = get_camera_params(json_path)

    picam2 = setup_camera(size=(1536, 864), fps=120)
    slam = Orbslam.Orbslam()
    time.sleep(7)
    start_time = time.time()
    mavlink.start_polling_thread(start_time)
    frame_count = 0
    fps_t_start = time.perf_counter()
    last_submitted_frame_time = 0
    while True:
        frame = picam2.capture_array()
        frame_time_ms = int((time.time() - start_time) * 1000)

        if frame is None:
            continue

        frame_count += 1

        packets = mavlink.get_last_raw_imu_packets()

        if slam.is_queue_empty():
            packets = mavlink.get_last_raw_imu_packets()

            idx_packets = [p for p in packets if (last_submitted_frame_time <= p["time_delta_sec"] < frame_time_ms / 1000.0)]
            processed_packets = process_imu_packets(idx_packets)
            if processed_packets is not None and len(processed_packets) >= 3:
                frame_ts = frame_time_ms / 1000.0
                smallest_frame = cv2.resize(frame, (768, 432))
                slam.submit_frame(smallest_frame, frame_ts, imu=processed_packets)
                # print(f"time delta: {frame_time_ms / 1000.0 - last_submitted_frame_time}")
                last_submitted_frame_time = frame_time_ms / 1000.0

        # result = slam.get_pose()
        # if result is not None:
        #     pose, timestamp = result
        #     xyz_rpy = Orbslam._pose_matrix_to_xyz_rpy(pose)
        #     if xyz_rpy is not None:
        #         x, y, z, roll, pitch, yaw = xyz_rpy
        #         print(f"SLAM: x: {x:.4f} y: {y:.4f} z: {z:.4f} roll: {roll:.2f} pitch: {pitch:.2f} yaw: {yaw:.2f} timestamp: {timestamp}")
        smallest_frame = cv2.resize(frame, (320, 180))
        smallest_frame = cv2.cvtColor(smallest_frame, cv2.COLOR_BGR2RGB)
        update_latest_frame(smallest_frame)
        # sender.send_image("air-pi", smallest_frame)

        elapsed = time.perf_counter() - fps_t_start
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            print(f"FPS: {fps:.1f}")
            frame_count = 0
            fps_t_start = time.perf_counter()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# SDP required:
# A description in SDP format is required to receive the RTP stream. Note that rtp:// URIs cannot work with dynamic RTP payload format (64).