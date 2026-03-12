import math

import numpy as np
import cv2
from typing import Optional, Tuple
import pyorbslam3

G = 9.81

def rpy_from_quaternion(q: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    """
    Convert quaternion (w, x, y, z) to roll, pitch, yaw.
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    roll = math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]))
    pitch = math.asin(2 * (q[0] * q[2] - q[3] * q[1]))
    yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]))
    return roll, pitch, yaw

def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Convert 3x3 rotation matrix to quaternion (w, x, y, z).

    Args:
        R: 3x3 rotation matrix.

    Returns:
        (w, x, y, z) quaternion.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    w = x = y = z = 0.0
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n > 1e-10:
        w, x, y, z = w / n, x / n, y / n, z / n
    return (float(w), float(x), float(y), float(z))


def body_pose_in_tag_frame(
    tvec_tag_in_body: np.ndarray,
    rvec_tag_in_body: np.ndarray,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """
    Express body pose in tag frame with tag as origin.
    Suitable for MAVLink ODOMETRY with frame_id = tag (local FRD).

    Body position and orientation are computed from tag-in-body (tvec, rvec).
    Position is returned in MAVLink LOCAL_FRD: x forward, y right, z down (tag frame).
    Quaternion is (w, x, y, z) from tag frame to body frame.

    Args:
        tvec_tag_in_body: Position of tag origin in body frame (3,) or (3, 1).
        rvec_tag_in_body: Rotation vector of tag in body frame (OpenCV convention).

    Returns:
        ( (x_frd, y_frd, z_frd), (qw, qx, qy, qz) )
    """
    t = np.asarray(tvec_tag_in_body, dtype=np.float64).reshape(3)
    rvec = np.asarray(rvec_tag_in_body, dtype=np.float64).reshape(3, 1)
    R_tag_to_body, _ = cv2.Rodrigues(rvec)
    # Body origin in tag frame (OpenCV: X right, Y down, Z forward)
    p_body_in_tag_cv = -R_tag_to_body.T @ t
    # MAVLink LOCAL_FRD: x forward, y right, z down  =>  x=z_cv, y=x_cv, z=y_cv
    x_frd = float(p_body_in_tag_cv[2])
    y_frd = float(p_body_in_tag_cv[0])
    z_frd = float(p_body_in_tag_cv[1])
    q = rotation_matrix_to_quaternion(R_tag_to_body)
    return p_body_in_tag_cv, q
    # return (x_frd, y_frd, z_frd), q


def tag_point_to_body(
    tvec_tag_in_body: np.ndarray,
    rvec_tag_in_body: np.ndarray,
    point_A_in_tag: np.ndarray,
) -> np.ndarray:
    """
    Express point A (given in tag frame) in body frame.

    Args:
        tvec_tag_in_body: Position of tag origin in body frame (3,) or (3, 1).
        rvec_tag_in_body: Rotation vector of tag in body frame (OpenCV convention).
        point_A_in_tag: Position of point A in tag frame (3,) or (3, 1).

    Returns:
        tvec_to_A_in_body: Position of point A in body frame, shape (3,).
    """
    t = np.asarray(tvec_tag_in_body, dtype=np.float64).reshape(3)
    rvec = np.asarray(rvec_tag_in_body, dtype=np.float64).reshape(3, 1)
    p_tag = np.asarray(point_A_in_tag, dtype=np.float64).reshape(3)
    R_tag_to_body, _ = cv2.Rodrigues(rvec)
    p_body = R_tag_to_body @ p_tag + t
    return p_body


def quaternion_from_euler(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        sr * sp * cy - cr * cp * sy
    ]

# Camera frame: z forward, x right, y down (right-handed).
# IMU frame:     x forward, y left, z up (right-handed).


def process_imu_packets(packets: list[dict]) -> Optional[np.ndarray]:
    """
    Convert MAVLink RAW_IMU packets into the ndarray format expected by ORBSlamPython.cpp
    convertImuFromNDArray: shape (N, 7), each row [accX, accY, accZ, gyroX, gyroY, gyroZ, timestamp].
    """
                    # y -> -z
                # x -> -y
                # z -> -x
    processed_packets = []
    for packet in packets:
        try:
            # Raw body NED: x fwd, y right, z down -> IMU: x fwd, y left, z up
            # ax = -float(packet["yacc"]) / 1000.0 * G   # forward
            # ay = -float(packet["zacc"]) / 1000.0 * G  # left = -right
            # az = -float(packet["xacc"]) / 1000.0 * G  # up = -down
            # gx = float(packet["ygyro"]) / 1000.0
            # gy = float(packet["zgyro"]) / 1000.0
            # gz = float(packet["xgyro"]) / 1000.0
            ax = float(packet["xacc"]) / 1000.0 * G
            ay = float(packet["yacc"]) / 1000.0 * G
            az = float(packet["zacc"]) / 1000.0 * G 
            gx = float(packet["xgyro"]) / 1000.0
            gy = float(packet["ygyro"]) / 1000.0
            gz = float(packet["zgyro"]) / 1000.0
            t = float(packet["time_delta_sec"])
            # data = pyorbslam3.IMUData(ax, ay, az, gx, gy, gz, t)
            processed_packets.append([ax, ay, az, gx, gy, gz, t])
        except (KeyError, TypeError, ValueError) as e:
            print("Error processing IMU packet: ", e)
    return processed_packets
