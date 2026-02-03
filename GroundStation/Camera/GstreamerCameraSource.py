import cv2
import os
import numpy as np
from typing import Optional, Tuple
from Camera.CameraSource import CameraSource

# Helps reduce buffering on Windows FFMPEG builds
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;0"


class GstreamerCameraSource(CameraSource):
    """
    Camera source implementation using FFMPEG backend for RTSP streams and other network sources.
    Accepts a connection string (RTSP URL or pipeline) as input.
    Provides full OpenCV integration with BGR frame format.
    """
    
    def __init__(self, pipeline: str):
        """
        Initialize camera source with FFMPEG backend.
        
        Args:
            pipeline: Connection string (e.g., RTSP URL like "rtsp://192.168.137.108:8554/stream")
        """
        super().__init__()
        self.pipeline = pipeline
        
    def _initialize_camera(self) -> bool:
        """Initialize camera with FFMPEG backend."""
        print("Starting camera initialization")
        print(f"Connection string: {self.pipeline}")
        try:
            # OpenCV VideoCapture with FFMPEG backend
            # The connection string (RTSP URL or pipeline) is passed directly to VideoCapture
            self._camera = cv2.VideoCapture(self.pipeline, cv2.CAP_FFMPEG)
            
            if not self._camera.isOpened():
                print("Failed to open camera connection")
                return False
            print("Camera opened")
                
            # Set additional OpenCV properties for better performance
            self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency
            
            self._is_initialized = True
            print("Camera initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False
            
    def _capture_single_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame using FFMPEG backend."""
        if not self._camera or not self._camera.isOpened():
            return None
            
        ret, frame = self._camera.read()
        if not ret:
            print("Frame grab failed")
            return None
        return frame
        
    def _cleanup_camera(self):
        """Clean up camera resources."""
        if self._camera:
            self._camera.release()
            self._camera = None
        self._is_initialized = False
    
    def get_camera_properties(self) -> dict:
        """
        Get current camera properties.
        
        Returns:
            dict: Dictionary containing camera properties
        """
        if not self._camera or not self._camera.isOpened():
            return {}
            
        return {
            'width': int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self._camera.get(cv2.CAP_PROP_FPS),
            'backend': 'FFMPEG',
            'connection_string': self.pipeline,
            'buffer_size': self._camera.get(cv2.CAP_PROP_BUFFERSIZE)
        }
    
    def set_camera_property(self, property_id: int, value: float) -> bool:
        """
        Set a camera property.
        
        Args:
            property_id: OpenCV property ID (e.g., cv2.CAP_PROP_BRIGHTNESS)
            value: Property value
            
        Returns:
            bool: True if set successfully, False otherwise
        """
        if not self._camera or not self._camera.isOpened():
            return False
        return self._camera.set(property_id, value)
    
    def get_camera_property(self, property_id: int) -> float:
        """
        Get a camera property value.
        
        Args:
            property_id: OpenCV property ID (e.g., cv2.CAP_PROP_FRAME_WIDTH)
            
        Returns:
            float: Property value or -1 if failed
        """
        if not self._camera or not self._camera.isOpened():
            return -1
        return self._camera.get(property_id)
    
    def get_pipeline(self) -> str:
        """
        Get the connection string (RTSP URL or pipeline).
        
        Returns:
            str: The connection string
        """
        return self.pipeline

