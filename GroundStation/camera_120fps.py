#!/usr/bin/env python3
"""
Standalone script to open the Raspberry Pi camera at 120 FPS.
Uses Picamera2 with a resolution that supports high frame rate (640x480).
Press 'q' to quit.
"""

import time
import cv2
from picamera2 import Picamera2


def main():
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

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            array = picam2.capture_array()
            frame = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

            frame_count += 1
            t_now = time.time()
            if t_now - t_start >= 1.0:
                fps = frame_count / (t_now - t_start)
                print(f"FPS: {fps:.1f}")
                frame_count = 0
                t_start = t_now

            cv2.imshow("Pi Camera 120 FPS", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                cv2.imwrite(f"/home/raspi/ardupilot/GroundStation/Camera/PiCameraW/Calibration/frame_{frame_count}.jpg", frame)
                print(f"Saved frame {frame_count}.jpg")
            elif key == ord("q"):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
