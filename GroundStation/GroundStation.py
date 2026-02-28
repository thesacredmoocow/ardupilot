"""
Example usage of the CameraSource base class and WebcamCameraSource implementation.
Demonstrates real-time camera display using cv2.imshow until keypress.
"""

import cv2
import time
import numpy as np
import json
from pathlib import Path
from Camera.WebcamCameraSource import WebcamCameraSource
# from Camera.LucidCameraSource import LucidCameraSource
from Camera.PiCameraFrameSource import PiCameraFrameSource
from PoseEstimator import PoseEstimator
from InsideOutEstimator import InsideOutEstimator
from VisionTarget.AprilVisionTarget import AprilVisionTarget
# from VisionTarget.IRVisionTarget import IRVisionTarget
from Pose import Pose
import math
import os
from Docking import DockingManager
from Utils import quaternion_from_euler

os.environ['MAVLINK20'] = '1'

from pymavlink import mavutil
from time import time_ns

MAVLINK_IP = "127.0.0.1"
MAVLINK_PORT = "5760"
SYSID = 1
FC_COMPID = 1
THIS_COMPID = 191

SEND_MAVLINK = True

INSIDE_OUT_ESTIMATOR = True

def main():

    print("Press 'q' to quit, 's' to save current frame")

    with PiCameraFrameSource(size=(2304, 1296), fps=30) as camera:
    # with WebcamCameraSource(camera_index=0) as camera:
        cameraMatrix = camera.get_camera_matrix()
        distCoeffs = camera.get_dist_coeffs()

        vision_target = AprilVisionTarget(camera_matrix=cameraMatrix, dist_coeffs=distCoeffs)
        # vision_target = IRVisionTarget(camera_matrix=cameraMatrix, dist_coeffs=distCoeffs)
        pose_estimator = PoseEstimator(cameraMatrix, distCoeffs, vision_target)
        inside_out_estimator = InsideOutEstimator(cameraMatrix, distCoeffs, vision_target)

        if SEND_MAVLINK:
            mavlink = mavutil.mavlink_connection(f"tcp:{MAVLINK_IP}:{MAVLINK_PORT}", source_system=SYSID, source_component=THIS_COMPID)
        else:
            mavlink = None

        if cameraMatrix is None or distCoeffs is None:
            print("Camera matrix or dist coeffs is None")
            return

        if not camera.is_capturing():
            print("Failed to start camera")
            return
        
        last_frame_time = time.time()
        fps_lpf = 0

        last_heartbeat_time = time.time()

        pose = Pose()
        reset_counter = 0
        docking_manager = DockingManager()
        send_position_deltas = False
        start_time = time.time()
        while True:
            valid_sent = False
            if mavlink is not None:
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
                    timestamp = camera._frame_timestamp
                    # pose_estimate = pose_estimator.estimate(frame, draw_on_frame=True)
                    pose_estimate = inside_out_estimator.estimate(frame, draw_on_frame=True)
                    if pose_estimate is not None:
                        print(pose_estimate["tvec_tag"].flatten().tolist())
                        cb_x, cb_y, cb_z = pose_estimate["docking_error"].flatten().tolist()
                        docking_manager.update_position_error(cb_x, cb_y, cb_z, pose_estimate["success"])

                        if INSIDE_OUT_ESTIMATOR:
                            pose.update_pos(pose_estimate["tag_in_body"][0], pose_estimate["tag_in_body"][1], pose_estimate["tag_in_body"][2], timestamp*1000000)
                        else:
                            pose.update_pos(pose_estimate["cam_in_body"][0], pose_estimate["cam_in_body"][1], pose_estimate["cam_in_body"][2], timestamp*1000000)
                        
                        fwd_error, right_error, down_error = pose_estimate["docking_error"].flatten().tolist()
                        print(f"Docking error (fwd, right, down): {fwd_error:.3f}, {right_error:.3f}, {down_error:.3f}")
                        
                        if mavlink is not None:
                            mavlink.mav.landing_target_send(
                                time_usec       =int(timestamp * 1000000),
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
                            valid_sent = True

                            rotation_quaternion = quaternion_from_euler(pose_estimate["attitude"][0], pose_estimate["attitude"][1], pose_estimate["attitude"][2])
                            mavlink.mav.odometry_send(
                                time_usec       =int(timestamp * 1000000),
                                frame_id           =mavutil.mavlink.MAV_FRAME_BODY_FRD,
                                child_frame_id    =mavutil.mavlink.MAV_FRAME_BODY_FRD,
                                x               =pose.fpos_x,
                                y               =pose.fpos_y,
                                z               =pose.fpos_z,
                                q =rotation_quaternion,
                                vx              =pose.vel_x,
                                vy              =pose.vel_y,
                                vz              =pose.vel_z,
                                rollspeed       =0.0,
                                pitchspeed      =0.0,
                                yawspeed        =0.0,
                                pose_covariance = [float('nan')] * 21,
                                velocity_covariance =[0] * 21,
                                reset_counter       =reset_counter,
                                quality             =100,
                                estimator_type      =mavutil.mavlink.MAV_ESTIMATOR_TYPE_VIO,
                            )
                            # print(int(timestamp*1000))
                            target_vz = 0
                            if docking_manager.get_state() == docking_manager.DockingState.DOCKING_STATE_INSERTION:
                                target_vz = 0.0
                            if send_position_deltas:
                                position_error = docking_manager.get_position_error()
                                mavlink.mav.set_position_target_local_ned_send(
                                    time_boot_ms=int((timestamp - start_time) * 1000),
                                    target_system=SYSID,
                                    target_component=FC_COMPID,
                                    coordinate_frame=mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                                    type_mask= 0b111111111000,
                                    x = position_error[0],
                                    y = position_error[1],
                                    z = position_error[2],
                                    vx = 0,
                                    vy = 0,
                                    vz = target_vz,
                                    afx = 0,
                                    afy = 0,
                                    afz = 0,
                                    yaw = 0,
                                    yaw_rate = 0,
                                )
                    
                    if not valid_sent and mavlink is not None:
                        reset_counter += 1
                        reset_counter = reset_counter % 255
                        mavlink.mav.odometry_send(
                            time_usec       =int(timestamp * 1000000),
                            frame_id           =mavutil.mavlink.MAV_FRAME_BODY_FRD,
                            child_frame_id    =mavutil.mavlink.MAV_FRAME_BODY_FRD,
                            x               =0.0,
                            y               =0.0,
                            z               =0.0,
                            q =[1.0, 0.0, 0.0, 0.0], #todo
                            vx              =0.0,
                            vy              =0.0,
                            vz              =0.0,
                            rollspeed       =0.0,
                            pitchspeed      =0.0,
                            yawspeed        =0.0,
                            pose_covariance =[float('nan')] * 21,
                            velocity_covariance =[0] * 21,
                            reset_counter       =reset_counter,
                            quality             =-1,
                            estimator_type      =mavutil.mavlink.MAV_ESTIMATOR_TYPE_VIO,
                        )
                    fps = 1.0 / (time.time() - last_frame_time) if time.time() > last_frame_time else 0
                    fps_lpf = 0.97 * fps_lpf + 0.03 * fps
                    cv2.putText(frame, f"FPS: {fps_lpf:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Display docking state
                    docking_state = docking_manager.get_state()
                    state_names = {
                        docking_manager.DockingState.DOCKING_STATE_INITIAL: "INITIAL",
                        docking_manager.DockingState.DOCKING_STATE_ALIGNMENT: "ALIGNMENT", 
                        docking_manager.DockingState.DOCKING_STATE_INSERTION: "INSERTION",
                        docking_manager.DockingState.DOCKING_STATE_DOCKED: "DOCKED",
                        docking_manager.DockingState.DOCKING_STATE_ERROR: "ERROR"
                    }
                    state_name = state_names.get(docking_state, "UNKNOWN")
                    state_color = (0, 255, 0) if docking_state != docking_manager.DockingState.DOCKING_STATE_ERROR else (0, 0, 255)
                    cv2.putText(frame, f"Docking State: {state_name}", (300, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
                    last_frame_time = time.time()

                    cv2.imwrite('Result.jpg', frame)

                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quit requested by user")
                        
                        mavlink.mav.command_long_send(
                            target_system=SYSID,
                            target_component=FC_COMPID,
                            command=mavutil.mavlink.MAV_CMD_DO_FLIGHTTERMINATION ,
                            confirmation=0,
                            param1=1.0,
                            param2=0,
                            param3=0,
                            param4=0,
                            param5=0,
                            param6=0,
                            param7=0,
                        )
                        break
                    elif key == ord('s'):
                        # Save current frame
                        filename = f"captured_frame_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame)
                    elif key == ord('d'):
                        send_position_deltas = not send_position_deltas
                        print(f"Send position deltas: {send_position_deltas}")

                        if send_position_deltas:
                            # set mav mode to guided
                            mavlink.mav.command_long_send(
                                target_system=SYSID,
                                target_component=FC_COMPID,
                                command=mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                                confirmation=0,
                                param1=mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                                param2=4,
                                param3=0,
                                param4=0,
                                param5=0,
                                param6=0,
                                param7=0,
                            )
                        else:
                            # set mav mode to loiter
                            mavlink.mav.command_long_send(
                                target_system=SYSID,
                                target_component=FC_COMPID,
                                command=mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                                confirmation=0,
                                param1=mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                                param2=5,
                                param3=0,
                                param4=0,
                                param5=0,
                                param6=0,
                                param7=0,
                            )
                    elif key == ord('l'):
                        mavlink.mav.command_long_send(
                            target_system=SYSID,
                            target_component=FC_COMPID,
                            command=mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                            confirmation=0,
                            param1=mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                            param2=9,
                            param3=0,
                            param4=0,
                            param5=0,
                            param6=0,
                            param7=0,
                        )
                    elif key == ord('t'):
                        mavlink.mav.command_long_send(
                            target_system=SYSID,
                            target_component=FC_COMPID,
                            command=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                            confirmation=0,
                            param1=0,
                            param2=9,
                            param3=0,
                            param4=0,
                            param5=0,
                            param6=0,
                            param7=0.3,
                        )
            else:
                time.sleep(0.01)  # Small delay to prevent busy waiting
    
    # Clean up
    cv2.destroyAllWindows()
    if mavlink is not None:
        mavlink.close()

if __name__ == "__main__":
    main()
