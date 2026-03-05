"""
ORB-SLAM in a separate process: submit frames for processing; get_pose() returns the pose
only on the first call after each processed frame, then None until the next.
Returns (pose, timestamp); pose can be None if tracking was lost.
"""

import multiprocessing as mp
import pathlib
import time
from queue import Empty
from typing import Optional, Tuple, Union

import numpy as np


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
    vocab_path: str,
    frame_queue: mp.Queue,
    pose_queue: mp.Queue,
) -> None:
    """Run in worker process: create SLAM, read frames from queue, process, push pose."""
    import pyorbslam  # import in worker so SLAM lives in this process

    slam = pyorbslam.MonoSLAM(settings_path)
    slam.reset()

    while True:
        item = frame_queue.get()
        if item is None:
            break
        frame, timestamp = item
        if frame is None:
            break
        state = slam.process(frame, timestamp)
        if state == pyorbslam.State.OK:
            pose = slam.get_pose_to_target()
            if pose is not None and pose.size > 0:
                pose_queue.put((np.asarray(pose, dtype=np.float64).copy(), timestamp))
            else:
                pose_queue.put((None, timestamp))
        else:
            pose_queue.put((None, timestamp))

    pose_queue.put((None, None))  # sentinel


class Orbslam:
    """
    ORB-SLAM running in a separate process. Submit frames with submit_frame();
    get_pose() returns (pose, timestamp) only on the first call after a frame is processed, then None until the next.
    """

    def __init__(
        self,
        settings_file: str = "/home/raspi/ardupilot/GroundStation/SLAM/orbslam_settings.yaml",
        vocab_file: Optional[Union[str, pathlib.Path]] = None,
        frame_queue_maxsize: int = 2,
    ):
        settings_path = pathlib.Path(settings_file)
        if vocab_file is None:
            try:
                import pyorbslam
                # pyorbslam package root -> repo root -> src/ORB_SLAM3/Vocabulary
                repo_root = pathlib.Path(pyorbslam.__file__).resolve().parent.parent.parent
                vocab_path = str(repo_root / "src" / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt")
            except Exception:
                vocab_path = str(
                    pathlib.Path(__file__).resolve().parent.parent.parent
                    / "pyorbslam" / "src" / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt"
                )
        else:
            vocab_path = str(pathlib.Path(vocab_file).resolve())

        self._frame_queue: mp.Queue = mp.Queue(maxsize=frame_queue_maxsize)
        self._pose_queue: mp.Queue = mp.Queue()

        self._process = mp.Process(
            target=_slam_worker,
            args=(settings_path, None, self._frame_queue, self._pose_queue),
            daemon=True,
        )
        self._process.start()

        self._latest_pose: Optional[np.ndarray] = None
        self._latest_timestamp: Optional[float] = None
        self._consumed = True  # no pose to return yet

    def submit_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """Queue a frame for processing. Frame should be 3-channel RGB or grayscale (HxW or HxWx1)."""
        try:
            self._frame_queue.put((np.ascontiguousarray(frame), timestamp), block=False)
        except Exception:
            try:
                self._frame_queue.put_nowait((np.ascontiguousarray(frame), timestamp))
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
