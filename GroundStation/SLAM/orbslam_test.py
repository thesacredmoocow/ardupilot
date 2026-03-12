# Import the necessary packages
import cv2
import numpy as np
import pyorbslam3
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

vocabPath = "/home/raspi/ORB_SLAM3/Vocabulary/ORBvoc.txt"
settingsPath = "/home/raspi/ardupilot/GroundStation/SLAM/orbslam_settings.yaml"

# Create SLAM
slam = pyorbslam3.System(vocabPath, settingsPath, "Mono", True) # Examples found in ``settings`` folder
# slam.Reset()
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
# Monocular init needs >100 keypoints per frame and ~100 matches with parallax.
# Move the camera slowly (sideways or forward) over a textured scene for a few seconds.
frame_count = 0
start_time = time.time()
while True:
    frame = picam2.capture_array()
    timestamp = time.time() - start_time
    state = slam.processMonocular(frame, timestamp)
    print(f"state: {state}")
