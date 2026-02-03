import threading
import time
import cv2
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import os
import json
import numpy as np


class CameraSource(ABC):
    """
    Base class for camera sources that implements threading for continuous image capture.
    Provides thread-safe access to camera frames and new frame detection.
    """
    
    def __init__(self):
        """
        Initialize the camera source.fr
        """

        print("camera intting")
        
        # Threading variables
        self._capture_thread = None
        self._running = False
        self._lock = threading.Lock()
        
        # Frame storage
        self._current_frame = None
        self._new_frame_available = False
        self._frame_timestamp = 0
        
        # Camera initialization
        self._camera = None
        self._is_initialized = False

        self._cameraMatrix = None
        self._distCoeffs = None

        # Prefer JSON camera parameters (produced by camera calibration).
        # Resolve the path relative to this source file so the code works
        # regardless of the current working directory when the program is run.
        base_dir = os.path.dirname(__file__)
        json_path = os.path.join(base_dir, 'camera_params.json')
        if os.path.exists(json_path):
            # camera params file found next to this module
            try:
                with open(json_path, 'r') as f:
                    params = json.load(f)

                cameraMatrix = params.get('camera_matrix')
                distCoeffs = params.get('dist_coeff')

                if cameraMatrix is not None:
                    # camera_matrix expected as a 3x3 nested list
                    self._cameraMatrix = np.array(cameraMatrix, dtype=float)

                if distCoeffs is not None:
                    # dist_coeff may be a nested list (e.g. [[...]]) or flat list
                    if isinstance(distCoeffs, list) and len(distCoeffs) == 1 and isinstance(distCoeffs[0], list):
                        dc = distCoeffs[0]
                    else:
                        dc = distCoeffs
                    self._distCoeffs = np.array(dc, dtype=float)
            
            except Exception as e:
                print('failed')
                print(f"Failed to load camera params from {json_path}: {e}")

        else:
            print('what is what')

    def get_camera_matrix(self) -> Optional[np.ndarray]:
        return self._cameraMatrix
    
    def get_dist_coeffs(self) -> Optional[np.ndarray]:
        return self._distCoeffs
        
    def start_capture(self) -> bool:
        """
        Start the camera capture thread.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self._running:
            return True
            
        if not self._initialize_camera():
            return False
            
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        return True
        
    def stop_capture(self):
        """Stop the camera capture thread."""
        self._running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        self._cleanup_camera()
        
    def is_new_frame_available(self) -> bool:
        """
        Check if a new frame is available.
        
        Returns:
            bool: True if a new frame is available, False otherwise
        """
        with self._lock:
            return self._new_frame_available
            
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest captured frame as OpenCV frame (BGR format).
        
        Returns:
            np.ndarray or None: The latest frame in BGR format if available, None otherwise
        """
        with self._lock:
            if self._current_frame is not None:
                self._new_frame_available = False
                return self._current_frame.copy()
            return None
            
    def get_frame_with_timestamp(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Get the latest frame along with its timestamp.
        
        Returns:
            Tuple[np.ndarray or None, float]: (BGR frame, timestamp) or (None, 0) if no frame
        """
        with self._lock:
            if self._current_frame is not None:
                self._new_frame_available = False
                return self._current_frame.copy(), self._frame_timestamp
            return None, 0
            
    def is_capturing(self) -> bool:
        """
        Check if the camera is currently capturing.
        
        Returns:
            bool: True if capturing, False otherwise
        """
        return self._running and self._capture_thread and self._capture_thread.is_alive()
        
    def get_frame_info(self) -> dict:
        """
        Get information about the current frame and capture status.
        
        Returns:
            dict: Dictionary containing frame info
        """
        with self._lock:
            frame_info = {
                'has_frame': self._current_frame is not None,
                'new_frame_available': self._new_frame_available,
                'frame_timestamp': self._frame_timestamp,
                'is_capturing': self.is_capturing(),
                'is_initialized': self._is_initialized
            }
            
            if self._current_frame is not None:
                frame_info.update({
                    'frame_shape': self._current_frame.shape,
                    'frame_dtype': str(self._current_frame.dtype),
                    'frame_channels': self._current_frame.shape[2] if len(self._current_frame.shape) == 3 else 1
                })
                
            return frame_info
    
    def save_frame(self, filename: str) -> bool:
        """
        Save the current frame to a file using OpenCV.
        
        Args:
            filename: Path where to save the frame
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        frame = self.get_latest_frame()
        if frame is not None:
            try:
                return cv2.imwrite(filename, frame)
            except Exception as e:
                print(f"Failed to save frame: {e}")
                return False
        return False

    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        last_capture_time = 0
        print("Starting capture loop")
        
        while self._running:
            current_time = time.time()
                
            # Capture frame
            frame = self._capture_single_frame()
            if frame is not None:
                with self._lock:
                    self._current_frame = frame
                    self._new_frame_available = True
                    self._frame_timestamp = current_time
                    
                last_capture_time = current_time
            else:
                # If capture failed, wait a bit before retrying
                time.sleep(0.01)
                print("Waiting for frame")
    
    @abstractmethod
    def _initialize_camera(self) -> bool:
        """
        Initialize the camera. Must be implemented by subclasses.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
        
    @abstractmethod
    def _capture_single_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera. Must be implemented by subclasses.
        
        Returns:
            np.ndarray or None: Captured frame in BGR format or None if failed
        """
        pass
        
    @abstractmethod
    def _cleanup_camera(self):
        """
        Clean up camera resources. Must be implemented by subclasses.
        """
        pass
        
    def __enter__(self):
        """Context manager entry."""
        self.start_capture()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_capture()