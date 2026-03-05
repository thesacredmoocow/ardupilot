# Import the necessary packages
import cv2
import numpy as np
import pyorbslam
import time
from picamera2 import Picamera2
import pathlib


def pose_matrix_to_xyz_rpy(pose):
    """Convert 4x4 pose matrix to xyz (m) and roll, pitch, yaw (deg)."""
    R = np.asarray(pose[:3, :3])
    t = np.asarray(pose[:3, 3]).flatten()
    x, y, z = t[0], t[1], t[2]
    # Rotation matrix to roll-pitch-yaw (XYZ Euler), in degrees
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = 0.0
    return x, y, z, roll, pitch, yaw

# Load video and create 3D path drawer
# cap = cv2.VideoCapture("video/path/here", 0)
# drawer = pyorbslam.TrajectoryDrawer()

settings_file = pathlib.Path("/home/raspi/ardupilot/GroundStation/SLAM/orbslam_settings.yaml")
vocab_file = pathlib.Path("/home/raspi/pyorbslam/src/ORB_SLAM3/Vocabulary/ORBvoc.txt")
# Create SLAM
slam = pyorbslam.MonoSLAM(settings_file) # Examples found in ``settings`` folder
slam.reset()
# 640x480 typically supports 120+ FPS on Pi camera modules
SIZE = (1536, 864)
FPS = 120

print(f"Opening camera at {SIZE[0]}x{SIZE[1]}, {FPS} FPS...")
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": SIZE},
    controls={"FrameRate": FPS},
    buffer_count=4,
)
picam2.configure(config)
picam2.start()
time.sleep(0.3)

# Runtime characterization: rolling window (ms) and FPS
TIMING_WINDOW = 30  # frames to average
timing_capture = []
timing_convert = []
timing_slam = []
timing_pose = []
timing_loop = []
last_loop_t = time.perf_counter()
print_every_n = 10  # print state + pose every N frames; timing every N frames

# Monocular init needs >100 keypoints per frame and ~100 matches with parallax.
# Move the camera slowly (sideways or forward) over a textured scene for a few seconds.
start_time = time.perf_counter()
frame_count = 0
while True:
    t_cap_start = time.perf_counter()
    frame = picam2.capture_array()
    t_cap_end = time.perf_counter()
    if frame is None:
        continue

    t_conv_start = time.perf_counter()
    # if array.ndim == 2:
    #     frame = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    # elif array.shape[2] == 4:
    #     frame = cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)
    # else:
    #     frame = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    # frame = np.ascontiguousarray(frame, dtype=np.uint8)
    frame = cv2.resize(frame, (768, 432))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t_conv_end = time.perf_counter()

    timestamp = t_cap_start - start_time
    t_slam_start = time.perf_counter()
    state = slam.process(frame, timestamp)
    t_slam_end = time.perf_counter()

    t_pose_start = time.perf_counter()
    pose_str = ""
    if state == pyorbslam.State.OK:
        pose = slam.get_pose_to_target()
        x, y, z, roll, pitch, yaw = pose_matrix_to_xyz_rpy(pose)
        pose_str = f"  xyz(m): {x:.4f} {y:.4f} {z:.4f}  |  rpy(deg): {roll:.2f} {pitch:.2f} {yaw:.2f}"
    t_pose_end = time.perf_counter()

    # Accumulate timings (ms)
    timing_capture.append((t_cap_end - t_cap_start) * 1000)
    timing_convert.append((t_conv_end - t_conv_start) * 1000)
    timing_slam.append((t_slam_end - t_slam_start) * 1000)
    timing_pose.append((t_pose_end - t_pose_start) * 1000)
    now = time.perf_counter()
    timing_loop.append((now - last_loop_t) * 1000)
    last_loop_t = now

    for buf in (timing_capture, timing_convert, timing_slam, timing_pose, timing_loop):
        if len(buf) > TIMING_WINDOW:
            buf.pop(0)

    frame_count += 1
    if frame_count % print_every_n != 0:
        continue

    # Report: state, pose, and timing/FPS
    print(state)
    if pose_str:
        print(pose_str)
    n = len(timing_loop)
    avg = lambda b: sum(b) / n if n else 0
    loop_ms = avg(timing_loop)
    fps_loop = 1000.0 / loop_ms if loop_ms > 0 else 0
    slam_ms = avg(timing_slam)
    fps_slam = 1000.0 / slam_ms if slam_ms > 0 else 0
    print(
        f"  [ms] capture: {avg(timing_capture):.2f}  convert: {avg(timing_convert):.2f}  slam: {slam_ms:.2f}  pose: {avg(timing_pose):.3f}  loop: {loop_ms:.2f}"
    )
    print(f"  [FPS] loop: {fps_loop:.1f}  slam: {fps_slam:.1f}")