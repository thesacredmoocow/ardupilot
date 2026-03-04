"""
Example usage of the CameraSource base class and WebcamCameraSource implementation.
Demonstrates real-time camera display and drawing of tag id 11 position (camera → tag 11).
"""

import cv2
import numpy as np
from picamera2 import Picamera2
from Camera.PiCameraFrameSource import PiCameraFrameSource
from VisionTarget.AprilVisionTarget import AprilVisionTarget

# Axis length in metres for drawing tag 11 pose
AXIS_LENGTH_M = 0.06


def print_camera_sensor_modes():
    """Print available sensor modes for the Pi camera."""
    cam = Picamera2()
    modes = cam.sensor_modes
    print(f"Available sensor modes ({len(modes)}):")
    for i, m in enumerate(modes):
        print(f"  [{i}] {m}")


def main():
    # print_camera_sensor_modes()
    print("\nPress 'q' to quit. Tag 11 position (rvec/tvec) is drawn when visible.")

    # with PiCameraFrameSource(size=(1536, 864), fps=120) as camera:
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

    while True:
        # if camera.is_new_frame_available():
            # frame = camera.get_latest_frame()
        frame = picam2.capture_array()
        if frame is None:
            continue
        # FPS calculation
        # FPS calculation
        import time
        if not hasattr(main, "_last_time"):
            main._last_time = time.time()
            main._frame_count = 0

        main._frame_count += 1
        now = time.time()
        if now - main._last_time >= 1.0:
            fps = main._frame_count / (now - main._last_time)
            print(f"FPS: {fps:.2f}")
            main._last_time = now
            main._frame_count = 0

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
