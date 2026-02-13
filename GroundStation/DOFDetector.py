import argparse
import json
import os
import threading
import time

import cv2
import numpy as np
from Camera.PiCameraFrameSource import PiCameraFrameSource
from pymavlink import mavutil


class DenseOpticalFlowEstimator:
    def __init__(self):
        self._farneback_kwargs = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )

    def estimate_translation(self, prev_gray: np.ndarray, gray: np.ndarray) -> tuple[float, float]:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **self._farneback_kwargs)
        flat_flow = flow.reshape(-1, 2)
        if flat_flow.size == 0:
            return 0.0, 0.0

        magnitudes = np.linalg.norm(flat_flow, axis=1)
        if magnitudes.size < 10:
            median_flow = np.median(flat_flow, axis=0)
            return float(median_flow[0]), float(median_flow[1])

        cutoff = np.percentile(magnitudes, 75.0)
        inliers = flat_flow[magnitudes <= cutoff]
        if inliers.shape[0] < 10:
            inliers = flat_flow

        median_flow = np.median(inliers, axis=0)
        return float(median_flow[0]), float(median_flow[1])


class DOFDetector:
    def __init__(
        self,
        camera_index: int = 1,
        frame_size: tuple[int, int] = (2304, 1296),
        fps: int = 30,
        output_size: tuple[int, int] = (640, 480),
        camera_params_path: str | None = None,
        send_mavlink: bool = True,
        mavlink_ip: str = "127.0.0.1",
        mavlink_port: int = 14545,
    ):
        self._camera = PiCameraFrameSource(
            camera_index=0,
            size=frame_size,
            fps=fps,
            output_size=output_size,
        )
        self._estimator = DenseOpticalFlowEstimator()
        self._send_mavlink = send_mavlink
        self._mavlink_ip = mavlink_ip
        self._mavlink_port = mavlink_port
        self._mavlink = None
        self._stop_event = threading.Event()
        self._thread = None
        self._camera_matrix = None
        self._dist_coeffs = None
        self._camera_params_path = camera_params_path or os.path.join(
            os.path.dirname(__file__), "camera_params.json"
        )
        self._load_camera_params()

    def _load_camera_params(self):
        if not os.path.exists(self._camera_params_path):
            print(f"[DOF] camera params not found: {self._camera_params_path}")
            return

        try:
            with open(self._camera_params_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            self._camera_matrix = np.array(params["camera_matrix"], dtype=np.float32)
            self._dist_coeffs = np.array(params["dist_coeff"], dtype=np.float32)
        except Exception as exc:
            print(f"[DOF] failed to load camera params: {exc}")

    def _init_mavlink(self):
        if not self._send_mavlink:
            return
        endpoint = f"udpout:{self._mavlink_ip}:{self._mavlink_port}"
        self._mavlink = mavutil.mavlink_connection(endpoint, source_system=1, source_component=197)
        print(f"[DOF] MAVLink connected on {endpoint}")

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, daemon=True, name="DOFDetectorThread")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None

    def _send_delta_mavlink(self, dx_px: float, dy_px: float, dt_s: float, now_s: float):
        if self._mavlink is None:
            return

        quality = int(np.clip(np.hypot(dx_px, dy_px) * 25.0, 0.0, 255.0))
        flow_x = int(np.clip(dx_px * 10.0, -32768, 32767))
        flow_y = int(np.clip(dy_px * 10.0, -32768, 32767))

        # OPTICAL_FLOW: flow_x/flow_y in pixel*10, compensated flow left as 0.0 when no gyro fusion.
        self._mavlink.mav.optical_flow_send(
            int(now_s * 1_000_000),  # time_usec
            0,                       # sensor_id
            flow_x,
            flow_y,
            0.0,                     # flow_comp_m_x
            0.0,                     # flow_comp_m_y
            quality,
            -1.0,                    # ground_distance unknown
        )

    def run(self):
        prev_gray = None
        prev_time = None
        last_heartbeat = 0.0

        try:
            self._init_mavlink()
            while not self._stop_event.is_set():
                ok, frame = self._camera.read()
                cv2.imwrite("saved_dof.png", frame)
                print('saved o,g')
                now = time.time()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                if self._camera_matrix is not None and self._dist_coeffs is not None:
                    frame = cv2.undistort(frame, self._camera_matrix, self._dist_coeffs)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is None:
                    prev_gray = gray
                    prev_time = now
                    continue

                dx_px, dy_px = self._estimator.estimate_translation(prev_gray, gray)
                dt_s = max(now - prev_time, 1e-6)
                print(f"[DOF] dx={dx_px:.3f} px, dy={dy_px:.3f} px, dt={dt_s * 1000.0:.1f} ms")

                if self._mavlink is not None:
                    if now - last_heartbeat > 1.0:
                        self._mavlink.mav.heartbeat_send(
                            type=mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                            autopilot=mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                            base_mode=0,
                            custom_mode=0,
                            system_status=mavutil.mavlink.MAV_STATE_ACTIVE,
                        )
                        last_heartbeat = now
                    self._send_delta_mavlink(dx_px, dy_px, dt_s, now)

                prev_gray = gray
                prev_time = now
        finally:
            self._camera.release()
            if self._mavlink is not None:
                self._mavlink.close()
                self._mavlink = None
            print("[DOF] detector stopped")


def main():
    parser = argparse.ArgumentParser(description="Dense optical flow DOF detector")
    parser.add_argument("--camera-index", type=int, default=1, help="PiCamera index to open")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera frame rate")
    parser.add_argument("--width", type=int, default=2304, help="Capture width")
    parser.add_argument("--height", type=int, default=1296, help="Capture height")
    parser.add_argument("--output-width", type=int, default=640, help="Processing width")
    parser.add_argument("--output-height", type=int, default=480, help="Processing height")
    parser.add_argument(
        "--camera-params-path",
        type=str,
        default=None,
        help="Path to camera_params.json (defaults to GroundStation/camera_params.json)",
    )
    parser.add_argument("--mavlink-ip", type=str, default="127.0.0.1", help="MAVLink destination IP")
    parser.add_argument("--mavlink-port", type=int, default=14545, help="MAVLink destination UDP port")
    parser.add_argument(
        "--disable-mavlink",
        action="store_true",
        help="Disable MAVLink output and print translations only",
    )
    args = parser.parse_args()

    detector = DOFDetector(
        camera_index=args.camera_index,
        frame_size=(args.width, args.height),
        fps=args.fps,
        output_size=(args.output_width, args.output_height),
        camera_params_path=args.camera_params_path,
        send_mavlink=not args.disable_mavlink,
        mavlink_ip=args.mavlink_ip,
        mavlink_port=args.mavlink_port,
    )

    detector.start()
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()


if __name__ == "__main__":
    main()
