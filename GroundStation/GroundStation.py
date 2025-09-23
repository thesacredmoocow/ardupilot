"""
Example usage of the CameraSource base class and WebcamCameraSource implementation.
Demonstrates real-time camera display using cv2.imshow until keypress.
"""

import cv2
import time
import numpy as np
# from Camera.WebcamCameraSource import WebcamCameraSource
from Camera.LucidCameraSource import LucidCameraSource
from PoseEstimator import PoseEstimator

import os

os.environ['MAVLINK20'] = '1'

from pymavlink import mavutil
from time import time_ns

MAVLINK_IP = "192.168.0.107"
MAVLINK_PORT = "5760"
SYSID = 1
COMPID = 192



def main():

    print("Press 'q' to quit, 's' to save current frame")

    with LucidCameraSource() as camera:
        cameraMatrix = camera.get_camera_matrix()
        distCoeffs = camera.get_dist_coeffs()

        pose_estimator = PoseEstimator(cameraMatrix, distCoeffs)

        mavlink = mavutil.mavlink_connection(f"tcp:{MAVLINK_IP}:{MAVLINK_PORT}", source_system=SYSID, source_component=COMPID)

        if cameraMatrix is None or distCoeffs is None:
            print("Camera matrix or dist coeffs is None")
            return

        if not camera.is_capturing():
            print("Failed to start camera")
            return
        
        last_frame_time = time.time()
        fps_lpf = 0

        last_heartbeat_time = time.time()
        
        while True:

            if time.time() - last_heartbeat_time > 1:
                mavlink.mav.heartbeat_send(
                    type            =mavutil.mavlink.MAV_TYPE_GENERIC,
                    autopilot       =mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
                    base_mode       =0,
                    system_status   =mavutil.mavlink.MAV_STATE_STANDBY,
                    custom_mode     =0,
                    mavlink_version =0,
                )
                last_heartbeat_time = time.time()
            if camera.is_new_frame_available():
                frame = camera.get_latest_frame()
                if frame is not None:
                    pose_estimate = pose_estimator.estimate(frame, draw_on_frame=True)
                    if pose_estimate is not None:
                        fwd_error, right_error, down_error = pose_estimate["docking_error"].flatten().tolist()
                        mavlink.mav.landing_target_send(
                            time_usec       =time_ns() // 1000,
                            target_num      =0,
                            frame           =mavutil.mavlink.MAV_FRAME_BODY_FRD,
                            angle_x         =0.0,
                            angle_y         =0.0,
                            distance        =0.0,
                            size_x          =0.0,
                            size_y          =0.0,
                            x               =fwd_error,
                            y               =right_error,
                            z               =down_error,
                            q               =[1.0, 0.0, 0.0, 0.0],
                            type            =mavutil.mavlink.LANDING_TARGET_TYPE_VISION_FIDUCIAL,
                            position_valid  =1,
                        )
                    

                    fps = 1.0 / (time.time() - last_frame_time) if time.time() > last_frame_time else 0
                    fps_lpf = 0.97 * fps_lpf + 0.03 * fps
                    cv2.putText(frame, f"FPS: {fps_lpf:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    last_frame_time = time.time()

                    cv2.imshow('Result', frame)

                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quit requested by user")
                        break
                    elif key == ord('s'):
                        # Save current frame
                        filename = f"captured_frame_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame)
            else:
                time.sleep(0.01)  # Small delay to prevent busy waiting
    
    # Clean up
    cv2.destroyAllWindows()
    mavlink.close()

if __name__ == "__main__":
    main()