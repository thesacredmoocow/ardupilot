#!/usr/bin/env python3
"""
Sparse optical flow example using Picamera2 + OpenCV.

Captures frames from the same camera configuration as `test-camera.py`,
detects good features to track, computes Lucas-Kanade optical flow between
consecutive frames, draws tracks on the image, and saves annotated frames
to files named `dof-annotated-0001.jpg`, `dof-annotated-0002.jpg`, ...

Stop with Ctrl-C.
"""

import time
import sys
from pathlib import Path

from imageSender import init_image_sender, update_latest_frame
from Camera.SimplePiCamera import setup_camera, get_camera_params

try:
    from picamera2 import Picamera2
except Exception:
    print("Error: could not import picamera2. Install with: sudo apt install -y python3-picamera2", file=sys.stderr)
    raise

import cv2
import numpy as np

from Mavlink_publisher import MavlinkPublisher

import os

os.environ['MAVLINK20'] = '1'

mavlink = MavlinkPublisher(
    format="udpin", 
    address="127.0.0.1", 
    port=14541,
    source_system=1,
    source_component=193,
    target_system=1,
    target_component=1
)


def main():
    mavlink.connect()
    out_dir = Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "dof-annotated"

    width, height = 640, 480
    max_corners = 300
    quality_level = 0.01
    min_distance = 6

    picam2 = setup_camera(size=(1640, 1232), fps=40, idx=1)
    time.sleep(0.1)  # warm-up

    prev_gray = None
    prev_pts = None
    color = None
    idx = 1

    # FPS and per-frame timing
    last_ts = time.time()
    fps = 0.0
    prev_frame_ts: float = last_ts

    init_image_sender(name="flow", endpoint="tcp://192.168.0.116:5555", send_hz=10.0)


    start_time = time.time()
    mavlink.start_polling_thread(start_time)

    print("Starting DOF capture. Press Ctrl-C to stop.")
    try:
        while True:
            frame = picam2.capture_array()
            if frame is None:
                print("Warning: empty frame", file=sys.stderr)
                continue

            frame = cv2.resize(frame, (width, height))

            # Rotate the frame 90 degrees clockwise
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # INSERT_YOUR_CODE
            # Crop out the lower 200 pixels of the frame
            if frame.shape[0] > 100:
                frame = frame[:-100, :]

            # Ensure frame is RGB -> convert to BGR for OpenCV drawing
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                # initialize features
                prev_gray = gray
                prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=max_corners,
                                                   qualityLevel=quality_level, minDistance=min_distance)
                if prev_pts is None:
                    prev_pts = np.empty((0, 1, 2), dtype=np.float32)
                # random colors for drawing
                color = np.random.randint(0, 255, (max(1, len(prev_pts)), 3)).tolist()
                # show the starting frame
                # cv2.imshow("DOF", bgr)
                # k = cv2.waitKey(1) & 0xFF
                # if k == ord('q'):
                #     break
                    

                # update FPS timing
                now = time.time()
                dt = now - last_ts if now - last_ts > 0 else 1e-6
                fps_inst = 1.0 / dt
                fps = fps_inst if fps == 0.0 else (0.9 * fps + 0.1 * fps_inst)
                last_ts = now
                prev_frame_ts = now

                idx += 1
                time.sleep(0.1)
                continue

            if prev_pts is None or len(prev_pts) < 30:
                # try to detect again
                prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=max_corners,
                                                   qualityLevel=quality_level, minDistance=min_distance)
                if prev_pts is None:
                    prev_pts = np.empty((0, 1, 2), dtype=np.float32)

            if len(prev_pts) > 0:
                # calc optical flow (sparse)
                next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None,
                                                                 winSize=(15, 15), maxLevel=2,
                                                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

                # Select good points
                good_new = next_pts[status.flatten() == 1]
                good_old = prev_pts[status.flatten() == 1].reshape(-1, 2)

                # Compute per-point pixel/sec velocities using dt between frames
                now = time.time()
                dt_frame = now - prev_frame_ts
                if dt_frame <= 0:
                    dt_frame = 1e-6

                dx = []
                dy = []

                # if good_new.size and good_old.size:
                #     delta = good_new.astype(np.float64) - good_old.astype(np.float64)
                #     v = delta / float(dt_frame)  # (N,2) px/sec
                #     vx = v[:, 0]
                #     vy = v[:, 1]
                #     print(
                #         f"dt={dt_frame*1000.0:.2f} ms  "
                #         f"vx px/s mean={np.mean(vx):.2f} med={np.median(vx):.2f}  "
                #         f"vy px/s mean={np.mean(vy):.2f} med={np.median(vy):.2f}"
                #     )

                # Draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    col = (0, 0, 0)
                    cv2.line(bgr, (int(a), int(b)), (int(c), int(d)), col, 2)
                    cv2.circle(bgr, (int(a), int(b)), 3, col, -1)
                    dx.append((a - c) / dt_frame)
                    dy.append((b - d) / dt_frame)

                avg_dx = np.mean(dx) if dx else 0.0
                med_dx = np.median(dx) if dx else 0.0
                avg_dy = np.mean(dy) if dy else 0.0
                med_dy = np.median(dy) if dy else 0.0
                # print(
                #     f"dx px/s avg={avg_dx:.2f}  med={med_dx:.2f}   "
                #     f"dy px/s avg={avg_dy:.2f}  med={med_dy:.2f}"
                # )
                mavlink.publish_optical_flow(
                    time_usec=int(now * 1000000),
                    flow_x=med_dx,
                    flow_y=med_dy,
                    quality=min(255, len(prev_pts) * 5)
                )

                # Overlay FPS on the frame
                cv2.putText(bgr, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 2, cv2.LINE_AA)

                # Show annotated frame
                # smallest_frame = cv2.resize(bgr, (320, 180))
                smallest_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                update_latest_frame(smallest_frame)
                # update FPS timing
                dt = now - last_ts if now - last_ts > 0 else 1e-6
                fps_inst = 1.0 / dt
                fps = fps_inst if fps == 0.0 else (0.9 * fps + 0.1 * fps_inst)
                last_ts = now

                # For continuous tracking, use the good_new as prev_pts
                prev_pts = good_new.reshape(-1, 1, 2).astype(np.float32)
                prev_gray = gray.copy()

                prev_frame_ts = now
            # small delay
            # time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        picam2.stop()


if __name__ == "__main__":
    main()
