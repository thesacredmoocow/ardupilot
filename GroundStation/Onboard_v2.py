import cv2
import os
import numpy as np
from Camera.PiCameraFrameSource import PiCameraFrameSource
from VisionTarget.AprilVisionTarget import AprilVisionTarget, CAMERA_TO_BODY_OFFSET_M
from Utils import tag_point_to_body, process_imu_packets, body_pose_in_tag_frame
from Mavlink_publisher import MavlinkPublisher
from pymavlink import mavutil
from picamera2 import Picamera2
from libcamera import controls
from Camera.SimplePiCamera import setup_camera, get_camera_params
import time
import math
import SLAM.Orbslam as Orbslam
from Docking import DockingManager

from imageSender import init_image_sender, update_latest_frame
# target_point = np.array([0.01, 0.5, -0.547])
# target_point_with_alt = np.array([0.0, 0.0, -1.0])
#left down fwd
target_point = np.array([0.01, -0.2, -0.7])
LANDING_TARGET_MESSAGE_OFFSET = 1.0 # meters

DEBUG_VECTORS = False
SEND_LANDING_TARGET = False


os.environ['MAVLINK20'] = '1'

json_path = "/home/raspi/ardupilot/GroundStation/Camera/PiCameraW/Calibration/camera_params.json"
TAG_AXIS_LENGTH_M = {
    11: 0.064,   # half of 12.8 cm tag size
    123: 0.0275, # half of 5.5 cm tag size
}


def draw_in_tag_frame(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec_tag_to_cam: np.ndarray,
    tvec_tag_to_cam: np.ndarray,
    points_tag: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> None:
    """
    Project 3D points expressed in the tag frame into the image and draw lines
    from the tag origin to each point.

    points_tag: array-like of shape (N, 3) in tag frame coordinates.
    """
    if frame is None or points_tag is None:
        return
    pts = np.asarray(points_tag, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return

    origin = np.zeros((1, 3), dtype=np.float64)
    obj_pts = np.vstack([origin, pts])

    img_pts, _ = cv2.projectPoints(
        obj_pts,
        rvec_tag_to_cam,
        tvec_tag_to_cam,
        camera_matrix,
        dist_coeffs,
    )
    img_pts = np.squeeze(img_pts, axis=1)

    # Ensure we have at least origin + one point
    if img_pts.ndim != 2 or img_pts.shape[0] < 2 or img_pts.shape[1] < 2:
        return

    h, w = frame.shape[:2]

    def _to_px(p):
        """Convert 2D point to Python int pixel tuple."""
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        return x, y

    origin_px = _to_px(img_pts[0])

    for i in range(1, img_pts.shape[0]):
        pt_px = _to_px(img_pts[i])
        # Optional bounds check before drawing
        if not (0 <= pt_px[0] < w and 0 <= pt_px[1] < h):
            continue
        cv2.line(frame, origin_px, pt_px, color, thickness)


def tag_point_in_tag_to_body(
    tvec_tag_in_body: np.ndarray,
    rvec_tag_in_body: np.ndarray,
    point_in_tag: np.ndarray,
) -> np.ndarray:
    """
    Convert a point expressed in the tag frame to body frame, from first principles.

    tvec_tag_in_body: position of the tag origin in body frame (3,) or (3,1).
    rvec_tag_in_body: rotation vector of tag in body frame (Rodrigues, OpenCV).
    point_in_tag:     point coordinates in tag frame (3,) or (3,1).

    Returns:
        point_in_body: (3,) position of the point in body frame.
    """
    t = np.asarray(tvec_tag_in_body, dtype=np.float64).reshape(3)
    rvec = np.asarray(rvec_tag_in_body, dtype=np.float64).reshape(3, 1)
    p_tag = np.asarray(point_in_tag, dtype=np.float64).reshape(3)

    R_tag_to_body, _ = cv2.Rodrigues(rvec)
    p_body = R_tag_to_body @ p_tag + t
    return p_body.reshape(3)


def draw_body_offset_axes(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    point_in_body: np.ndarray,
    thickness: int = 2,
) -> None:
    """
    Draw axes lines from the body origin to indicate offsets of a point in body frame.

    Body frame: x right, y down, z forward (same orientation as camera, but translated).

    We visualize the offset (dx, dy, dz) = point_in_body as a stepped path:
      origin -> (dx, 0, 0)  [X component, red]
              -> (dx, dy, 0)  [Y component, green]
              -> (dx, dy, dz) [Z component, blue = full offset]
    """
    if frame is None or point_in_body is None:
        return
    p = np.asarray(point_in_body, dtype=np.float64).reshape(3)

    # Object frame = body. In camera frame, body origin is at -CAMERA_TO_BODY_OFFSET_M.
    rvec_body_to_cam = np.zeros((3, 1), dtype=np.float64)
    tvec_body_to_cam = -CAMERA_TO_BODY_OFFSET_M.reshape(3, 1)

    obj_pts = np.array(
        [
            [0.0, 0.0, 0.0],          # 0: body origin
            [p[0], 0.0, 0.0],         # 1: x-only
            [p[0], p[1], 0.0],        # 2: x + y
            [p[0], p[1], p[2]],       # 3: x + y + z (full offset)
        ],
        dtype=np.float64,
    )

    img_pts, _ = cv2.projectPoints(
        obj_pts,
        rvec_body_to_cam,
        tvec_body_to_cam,
        camera_matrix,
        dist_coeffs,
    )
    img_pts = np.squeeze(img_pts, axis=1)
    if img_pts.ndim != 2 or img_pts.shape[0] < 4 or img_pts.shape[1] < 2:
        return

    h, w = frame.shape[:2]

    def _to_px(p2):
        x = int(round(float(p2[0])))
        y = int(round(float(p2[1])))
        return x, y

    origin_px = _to_px(img_pts[0])
    x_px = _to_px(img_pts[1])
    xy_px = _to_px(img_pts[2])
    xyz_px = _to_px(img_pts[3])

    # Draw stepped axes that all meet at the final target point xyz_px.
    segments = [
        (origin_px, x_px, (0, 0, 255)),   # X: red
        (x_px, xy_px, (0, 255, 0)),       # Y: green
        (xy_px, xyz_px, (255, 0, 0)),     # Z: blue
    ]
    for p0, p1, col in segments:
        if 0 <= p0[0] < w and 0 <= p0[1] < h and 0 <= p1[0] < w and 0 <= p1[1] < h:
            cv2.line(frame, p0, p1, col, thickness)


mavlink = MavlinkPublisher(
    format="udpin", 
    address="127.0.0.1", 
    port=14542,
    source_system=1,
    source_component=192,
    target_system=1,
    target_component=1
)
def main():
    mavlink.connect()
    
    init_image_sender(name="onboard", send_hz=10.0)
    time.sleep(1)
    # while not mavlink.request_local_position_ned_at_50hz():
    #     print("Requesting local position NED at 50 Hz")
    #     time.sleep(1)

    cameraMatrix, distCoeffs = get_camera_params(json_path)
    cameraMatrix = np.asarray(cameraMatrix, dtype=np.float64).reshape(3, 3)
    distCoeffs = np.asarray(distCoeffs, dtype=np.float64).reshape(-1, 1)
    
    vision_target = AprilVisionTarget(camera_matrix=cameraMatrix, dist_coeffs=distCoeffs)
    docking_manager = DockingManager()

    picam2 = setup_camera(size=(1536, 864), fps=120)
    # slam = Orbslam.Orbslam()
    start_time = time.time()
    mavlink.start_polling_thread(start_time)
    frame_count = 0
    fps_t_start = time.perf_counter()
    while True:
        frame = picam2.capture_array()
        frame_time_ms = int((time.time() - start_time) * 1000)

        if frame is None:
            continue
        frame_count += 1


        result = vision_target.get_position(frame)
        if result is not None:
            rvec = result["rvec"]
            tvec = result["tvec"]
            det = result["detection"]
            # Odometry with tag frame as origin (MAVLink ODOMETRY)
            (x_tag, y_tag, z_tag), q_tag_to_body = body_pose_in_tag_frame(tvec, rvec)
            # print(f"x_tag: {x_tag:.2f}, y_tag: {y_tag:.2f}, z_tag: {z_tag:.2f}")

            # target_point_in_tag = docking_manager.get_target_positition(np.array([x_tag, y_tag, z_tag]))
            # DOCKED_POSITION = np.array([-0.04, -0.283, -0.548])
            # ALIGNMENT_ALT_OFFSET = 0.2
            # ALIGNMENT_POSITION = np.array([DOCKED_POSITION[0], DOCKED_POSITION[1] + ALIGNMENT_ALT_OFFSET, DOCKED_POSITION[2]])
            # target_point_in_tag =  ALIGNMENT_POSITION
            target_point_in_tag = docking_manager.get_target_positition(np.array([x_tag, y_tag, z_tag]))
            target_point_in_tag_on_ground = np.array([target_point_in_tag[0], target_point_in_tag[1] + 1.0, target_point_in_tag[2]])
            
            # rvec/tvec are tag-in-body; drawFrameAxes and draw_in_tag_frame need tag-in-camera.
            tvec_body = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
            rvec_body = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            tvec_cam = tvec_body - CAMERA_TO_BODY_OFFSET_M.reshape(3, 1)

            # Draw a line from tag origin to target_point_in_tag in the image
            draw_in_tag_frame(
                frame=frame,
                camera_matrix=cameraMatrix,
                dist_coeffs=distCoeffs,
                rvec_tag_to_cam=rvec_body,
                tvec_tag_to_cam=tvec_cam,
                points_tag=np.asarray(target_point_in_tag, dtype=np.float64).reshape(1, 3),
                color=(0, 255, 0),
                thickness=5,
            )

            # Convert target point from tag frame to body frame and draw body-frame offset axes
            target_point_in_body = tag_point_in_tag_to_body(
                tvec_tag_in_body=tvec,
                rvec_tag_in_body=rvec,
                point_in_tag=target_point_in_tag,
            )

            target_ground_point_in_body = tag_point_in_tag_to_body(
                tvec_tag_in_body=tvec,
                rvec_tag_in_body=rvec,
                point_in_tag=target_point_in_tag_on_ground,
            )

            # print(f"Target point in body: left: {-target_point_in_body[0]:.2f}, down: {target_point_in_body[1]:.2f}, fwd: {target_point_in_body[2]:.2f}")



            target_vertical_speed = -target_point_in_body[1] * 100
            # yaw = math.atan2(target_point_in_body[0], target_point_in_body[2])
            # INSERT_YOUR_CODE
            # Compute yaw such that the vehicle is pointing toward the tag's position (in body frame).
            # Here, yaw = atan2(left, forward) = atan2(x, z)
            yaw = math.atan2(tvec_body[0], tvec_body[2]) / 3




            mavlink.publish_landing_target(target_ground_point_in_body, use_angles=False, timestamp=frame_time_ms)

            print(f"vertical speed: {target_vertical_speed:.2f}, yaw: {yaw:.2f}")

            mavlink.publish_position_target(target_point_in_body, yaw, target_vertical_speed, frame_time_ms)

            # axis_len = TAG_AXIS_LENGTH_M.get(int(det.tag_id), 0.05)
            # if float(tvec_cam[2, 0]) > 0.0:
            #     cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec_body, tvec_cam, axis_len,10)
        


        smallest_frame = cv2.resize(frame, (320, 180))
        smallest_frame = cv2.cvtColor(smallest_frame, cv2.COLOR_BGR2GRAY)
        update_latest_frame(smallest_frame)

        elapsed = time.perf_counter() - fps_t_start
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            # print(f"FPS: {fps:.1f}")
            frame_count = 0
            fps_t_start = time.perf_counter()


if __name__ == "__main__":
    main()
