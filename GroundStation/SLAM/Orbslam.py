"""
ORB-SLAM (mono + IMU) in a separate process. Submit frames and optional IMU with submit_frame();
get_pose() returns (pose, timestamp) only on the first call after each processed frame, then None.
IMU format: (N, 7) array, each row [AccX, AccY, AccZ, GyroX, GyroY, GyroZ, Timestamp] in SI units.
Use imu_from_raw() to build from MavlinkPublisher.get_raw_imu().
"""

import multiprocessing as mp
import pathlib
import time
from queue import Empty
from typing import Optional, Tuple, Union

import numpy as np


# IMU array format for ORB-SLAM3: each row [AccX, AccY, AccZ, GyroX, GyroY, GyroZ, Timestamp]
# in SI: acc in m/s^2, gyro in rad/s, timestamp in seconds.
IMU_ROW_ACC_X, IMU_ROW_ACC_Y, IMU_ROW_ACC_Z = 0, 1, 2
IMU_ROW_GYRO_X, IMU_ROW_GYRO_Y, IMU_ROW_GYRO_Z = 3, 4, 5
IMU_ROW_TIME = 6

# Minimum position (acc) or rotation (gyro) delta between consecutive samples; below this, drop the second sample.
IMU_MIN_DELTA = 1e-6


def _filter_imu_small_deltas(imu_arr: np.ndarray, min_delta: float = IMU_MIN_DELTA) -> np.ndarray:
    """
    Drop IMU samples whose position (acceleration) or rotation (gyro) delta from the previous
    kept sample is less than min_delta. Always keeps the first sample.
    """
    if imu_arr.shape[0] < 2:
        return imu_arr
    kept = [0]
    for i in range(1, imu_arr.shape[0]):
        acc_prev = imu_arr[kept[-1], IMU_ROW_ACC_X : IMU_ROW_ACC_Z + 1]
        gyro_prev = imu_arr[kept[-1], IMU_ROW_GYRO_X : IMU_ROW_GYRO_Z + 1]
        acc_cur = imu_arr[i, IMU_ROW_ACC_X : IMU_ROW_ACC_Z + 1]
        gyro_cur = imu_arr[i, IMU_ROW_GYRO_X : IMU_ROW_GYRO_Z + 1]
        pos_delta = np.linalg.norm(acc_cur.astype(np.float64) - acc_prev.astype(np.float64))
        rot_delta = np.linalg.norm(gyro_cur.astype(np.float64) - gyro_prev.astype(np.float64))
        if pos_delta >= min_delta and rot_delta >= min_delta:
            kept.append(i)
    return imu_arr[np.asarray(kept)]


def _pose_matrix_to_xyz_rpy(pose: Optional[np.ndarray]) -> Optional[Tuple[float, float, float, float, float, float]]:
    """Convert 4x4 pose matrix to xyz and roll, pitch, yaw (deg). Returns None if pose is None."""
    if pose is None:
        return None
    R = np.asarray(pose[:3, :3])
    t = np.asarray(pose[:3, 3]).flatten()
    x, y, z = t[0], t[1], t[2]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = 0.0
    return x, y, z, roll, pitch, yaw


def _slam_worker(
    settings_path: str,
    frame_queue: mp.Queue,
    pose_queue: mp.Queue,
) -> None:
    """Run in worker process: create SLAM (mono+IMU), read (frame, timestamp, imu) from queue, process, push pose."""
    import pyorbslam3  # import in worker so SLAM lives in this process
    vocabPath = "/home/raspi/ORB_SLAM3/Vocabulary/ORBvoc.txt"
    settingsPath = "/home/raspi/ardupilot/GroundStation/SLAM/orbslam_settings.yaml"

    slam = pyorbslam3.System(vocabPath, settingsPath, "MonoIMU", False)
    slam.Reset()

    while True:
        item = frame_queue.get()
        if item is None:
            print("frame queue is None")
            break
        frame, timestamp, imu = item
        if frame is None:
            print("frame is None")
            break
        # C++ convertImuFromNDArray expects shape (N, 7); do not wrap in extra dimension.
        # imu_arr = np.asarray(imu, dtype=np.float64)
        # if imu_arr.ndim == 1:
        #     imu_arr = imu_arr.reshape(1, -1)
        # if imu_arr.shape[0] < 2:
        #     # ORB_SLAM3 needs at least 2 IMU measurements; skip to avoid "Empty IMU" in C++.
        #     pose_queue.put((None, timestamp))
        #     continue
        reprocessed_packets = []
        for packet in imu:
            reprocessed_packets.append(pyorbslam3.IMUData(packet[0], packet[1], packet[2], packet[3], packet[4], packet[5], packet[6]))
        # print("sending frame to SLAM")
        # print(reprocessed_packets[0].acc_x, reprocessed_packets[0].acc_y, reprocessed_packets[0].acc_z, reprocessed_packets[0].ang_vel_x, reprocessed_packets[0].ang_vel_y, reprocessed_packets[0].ang_vel_z, reprocessed_packets[0].timestamp)
        state = slam.processMonocularIMU(frame, timestamp, reprocessed_packets)
        # print(len(reprocessed_packets))
        print(f"state: {slam.GetTrackingState()}")
        if slam.GetTrackingState() == 2:
            # pose = slam.get_pose_to_target()
            pose = state
            if pose is not None and pose.size > 0:
                pose_queue.put((np.asarray(pose, dtype=np.float64).copy(), timestamp))
                print(f"pose: {pose}")
            else:
                pose_queue.put((None, timestamp))
        # print(f"state: {state}")
        # if state == pyorbslam3.State.OK:
        #     pose = slam.get_pose_to_target()
        #     if pose is not None and pose.size > 0:
        #         pose_queue.put((np.asarray(pose, dtype=np.float64).copy(), timestamp))
        #     else:
        #         pose_queue.put((None, timestamp))
        # else:
        #     pose_queue.put((None, timestamp))
        #     print("state is: ", state)

    pose_queue.put((None, None))  # sentinel


class Orbslam:
    """
    ORB-SLAM running in a separate process. Submit frames with submit_frame();
    get_pose() returns (pose, timestamp) only on the first call after a frame is processed, then None until the next.
    """

    def __init__(
        self,
        settings_file: str = "/home/raspi/ardupilot/GroundStation/SLAM/orbslam_settings.yaml",
        frame_queue_maxsize: int = 2,
    ):
        settings_path = pathlib.Path(settings_file)

        self._frame_queue: mp.Queue = mp.Queue(maxsize=frame_queue_maxsize)
        self._pose_queue: mp.Queue = mp.Queue()

        self._process = mp.Process(
            target=_slam_worker,
            args=(settings_path, self._frame_queue, self._pose_queue),
            daemon=True,
        )
        self._process.start()

        self._latest_pose: Optional[np.ndarray] = None
        self._latest_timestamp: Optional[float] = None
        self._consumed = True  # no pose to return yet
        self._last_submit_timestamp: Optional[float] = None
        self._last_submit_imu: Optional[np.ndarray] = None  # (6,) last [ax, ay, az, gx, gy, gz]

    def submit_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        imu: Union[list, np.ndarray],
    ) -> None:
        """
        Queue a frame (and optional IMU) for processing only if the timestamp has
        incremented and the IMU readings (acc + gyro) differ from the previous submission.
        Frame: 3-channel RGB or grayscale (HxW or HxWx1).
        imu: optional (N, 7) array, each row [AccX, AccY, AccZ, GyroX, GyroY, GyroZ, Timestamp]
             in SI (m/s^2, rad/s, seconds). Use imu_from_raw() to build from MavlinkPublisher raw_imu.
        """
        # Ensure IMU readings have timestamps between last and current timestamps
        filtered_imu = None
        if imu is not None:
            imu_arr = np.asarray(imu)
            # Ensure we have at least 2 columns to index column 6 (timestamp)
            if imu_arr.ndim == 2 and imu_arr.shape[1] >= 7:
                # Accept only IMU rows with ts in (self._last_submit_timestamp, timestamp]
                # (if _last_submit_timestamp is None, include everything up to timestamp)
                if self._last_submit_timestamp is not None:
                    # Only rows with ts > last and ts <= current
                    ts_mask = (imu_arr[:,6] > self._last_submit_timestamp) & (imu_arr[:,6] <= timestamp)
                else:
                    ts_mask = imu_arr[:,6] <= timestamp
                filtered_imu = imu_arr[ts_mask]
                # If empty after filtering, pass empty array for consistent downstream
                if filtered_imu.size == 0:
                    filtered_imu = np.empty((0, imu_arr.shape[1]), dtype=imu_arr.dtype)
                else:
                    # Drop samples whose position (acc) or rotation (gyro) delta from previous kept sample is < 1e-6
                    filtered_imu = _filter_imu_small_deltas(filtered_imu)
            else:
                # If not well-shaped, pass along as-is
                filtered_imu = imu
        else:
            filtered_imu = None

        # print(filtered_imu, timestamp)

        imu_to_send = filtered_imu if filtered_imu is not None else imu
        self._last_submit_timestamp = timestamp
        # self._last_submit_imu = readings.copy()
        # print(imu)
        
        try:
            self._frame_queue.put((np.ascontiguousarray(frame), timestamp, imu_to_send), block=False)
        except Exception:
            try:
                self._frame_queue.put_nowait((np.ascontiguousarray(frame), timestamp, imu_to_send))
            except Exception:
                pass  # drop if full

    def get_pose(self) -> Optional[Tuple[Optional[np.ndarray], float]]:
        """
        Return (pose, timestamp) for the most recently processed frame on the first call
        after that frame was processed; return None on subsequent calls until another frame
        has been processed. pose is the 4x4 matrix or None if tracking was lost; timestamp
        is the frame timestamp passed to submit_frame().
        """
        while True:
            try:
                item = self._pose_queue.get_nowait()
                pose, timestamp = item
                self._latest_pose = pose
                self._latest_timestamp = timestamp
                self._consumed = False
            except Empty:
                break
        if self._consumed:
            return None
        if self._latest_timestamp is None:
            self._consumed = True
            return None
        self._consumed = True
        return (self._latest_pose, self._latest_timestamp)

    def get_pose_xyz_rpy(
        self,
    ) -> Optional[Tuple[Optional[Tuple[float, float, float, float, float, float]], float]]:
        """
        Same as get_pose() but returns ((x, y, z, roll_deg, pitch_deg, yaw_deg), timestamp) or None.
        First element is None if tracking was lost. Consumes like get_pose() (one of get_pose / get_pose_xyz_rpy per frame).
        """
        result = self.get_pose()
        if result is None:
            return None
        pose, timestamp = result
        if pose is None:
            return (None, timestamp)
        return (_pose_matrix_to_xyz_rpy(pose), timestamp)

    def is_queue_empty(self) -> bool:
        """Return True if the frame queue has no frames waiting to be processed."""
        return self._frame_queue.empty()

    def stop(self) -> None:
        """Signal the worker to exit and join (blocks until process ends)."""
        try:
            self._frame_queue.put(None, block=False)
        except Exception:
            pass
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)

    def __enter__(self) -> "Orbslam":
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
