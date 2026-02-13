import time
from typing import Tuple

import cv2
from picamera2 import Picamera2


class PiCameraFrameSource:
    """
    Lightweight PiCamera2 frame source with configurable camera index.
    """

    def __init__(
        self,
        camera_index: int = 0,
        size: Tuple[int, int] = (2304, 1296),
        fps: int = 30,
        output_size: Tuple[int, int] | None = (640, 480),
        exposure_time_us: int | None = 300,
    ):
        self._camera_index = camera_index
        self._output_size = output_size
        self._picam2 = self._create_camera(camera_index)

        config = self._picam2.create_video_configuration(
            main={"size": size, "format": "BGR888"},
            controls={"FrameRate": fps},
        )
        self._picam2.configure(config)
        self._picam2.start()

        if exposure_time_us is not None:
            self._picam2.set_controls({"ExposureTime": exposure_time_us})

        time.sleep(0.5)

    def _create_camera(self, camera_index: int) -> Picamera2:
        # Picamera2 constructor signature varies by version. Support both.
        try:
            return Picamera2(camera_num=camera_index)
        except TypeError:
            return Picamera2(camera_index)

    def read(self):
        frame = self._picam2.capture_array()
        if frame is None:
            return False, None
        if self._output_size is not None:
            frame = cv2.resize(frame, self._output_size)
        return True, frame

    def release(self):
        self._picam2.stop()
