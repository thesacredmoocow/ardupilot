import time
from typing import Optional, Tuple

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls

from Camera.CameraSource import CameraSource


class PiCameraFrameSource(CameraSource):
    """
    Camera source implementation backed by Picamera2.
    Provides frames in OpenCV BGR format via the shared CameraSource API.
    """

    def __init__(self, size: Tuple[int, int] = (2304, 1296), fps: int = 30):
        super().__init__(json_path="/home/raspi/ardupilot/GroundStation/Camera/PiCameraW/Calibration/camera_params.json")
        self.size = size
        self.fps = fps

    def _initialize_camera(self) -> bool:
        print("Starting PiCamera2 initialization")
        try:
            self._camera = Picamera2()
            config = self._camera.create_video_configuration(
                main={"size": self.size},#, "format": "RGB888"},
                controls={
                    "FrameRate": self.fps,
                    # "AfMode": controls.AfModeEnum.Continuous
                },
                buffer_count=4,
            )
            self._camera.configure(config)
            self._camera.start()
            # Let auto-exposure/awb settle briefly before capture loop starts.
            time.sleep(0.5)

            self._is_initialized = True
            print("PiCamera2 initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to initialize PiCamera2: {e}")
            return False

    def _capture_single_frame(self) -> Optional[np.ndarray]:
        if not self._camera:
            return None

        try:
            rgb = self._camera.capture_array()
            # ret = cv2.resize(rgb, (640, 480))
            ret = cv2.cvtColor(ret, cv2.COLOR_RGB2GRAY)
            return ret
        except Exception:
            return None

    def _cleanup_camera(self):
        if self._camera:
            try:
                self._camera.stop()
            except Exception:
                pass
            self._camera = None
        self._is_initialized = False

