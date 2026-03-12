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

import threading
import time
import cv2
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import os
import json
import numpy as np

def setup_camera(size=(1536, 864), fps=120, idx=0):
    SIZE = (1536, 864)
    FPS = 120

    print(f"Opening camera at {SIZE[0]}x{SIZE[1]}, {FPS} FPS...")
    picam2 = Picamera2(idx)
    config = picam2.create_video_configuration(
        main={"size": SIZE},
        controls={"FrameRate": FPS},
        buffer_count=4,
    )
    picam2.configure(config)
    picam2.start()
    # Enable continuous autofocus (if the camera supports it)
    try:
        picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    except Exception as e:
        print(f"Autofocus not available (camera may not support it): {e}")
    return picam2

def get_camera_params(json_path):
    cameraMatrix = None
    distCoeffs = None
    if os.path.exists(json_path):
            # camera params file found next to this module
            try:
                with open(json_path, 'r') as f:
                    params = json.load(f)

                cameraMatrix = params.get('camera_matrix')
                distCoeffs = params.get('dist_coeff')

                # if cameraMatrix is not None:
                    # camera_matrix expected as a 3x3 nested list
                    # self._cameraMatrix = np.array(cameraMatrix, dtype=float)

                # if distCoeffs is not None:
                #     # dist_coeff may be a nested list (e.g. [[...]]) or flat list
                #     if isinstance(distCoeffs, list) and len(distCoeffs) == 1 and isinstance(distCoeffs[0], list):
                #         dc = distCoeffs[0]
                #     else:
                #         dc = distCoeffs
                #     # self._distCoeffs = np.array(dc, dtype=float)
            
            except Exception as e:
                print('failed')
                print(f"Failed to load camera params from {json_path}: {e}")
    return cameraMatrix, distCoeffs