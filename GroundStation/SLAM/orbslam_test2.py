# Import the necessary packages
import cv2
import pyorbslam
import time
from picamera2 import Picamera2
import pathlib

# Load video and create 3D path drawer
# cap = cv2.VideoCapture("video/path/here", 0)
# drawer = pyorbslam.TrajectoryDrawer()

# Timestamp information
timestamp = 0

settings_file = pathlib.Path("/home/raspi/ardupilot/GroundStation/SLAM/orbslam_settings.yaml")
vocab_file = pathlib.Path("/home/raspi/pyorbslam/src/ORB_SLAM3/Vocabulary/ORBvoc.txt")
# Create SLAM
slam = pyorbslam.MonoSLAM(settings_file) # Examples found in ``settings`` folder
slam.reset()