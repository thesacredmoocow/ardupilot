import cv2
import numpy as np
from typing import Optional, Tuple
from Camera.CameraSource import CameraSource

class WebcamCameraSource(CameraSource):
    """
    Camera source implementation using OpenCV VideoCapture.
    Provides full OpenCV integration with BGR frame format.
    """
    
    def __init__(self, camera_index: int = 0, resolution: Tuple[int, int] = None, 
                 backend: int = None, api_preference: int = None):
        """
        Initialize OpenCV camera source.
        
        Args:
            camera_index: Camera index (0 for default camera)
            resolution: Optional (width, height) tuple for resolution
            backend: OpenCV backend (e.g., cv2.CAP_DSHOW, cv2.CAP_V4L2)
            api_preference: OpenCV API preference (e.g., cv2.CAP_ANY)
        """
        super().__init__()
        self.camera_index = camera_index
        self.resolution = resolution
        self.backend = backend
        self.api_preference = api_preference
        
    def _initialize_camera(self) -> bool:
        """Initialize OpenCV camera with specified backend and properties."""
        print("Starting camera initialization")
        try:
            # Initialize camera with optional backend and API preference
            if self.backend is not None and self.api_preference is not None:
                self._camera = cv2.VideoCapture(self.camera_index, self.backend, self.api_preference)
            elif self.backend is not None:
                self._camera = cv2.VideoCapture(self.camera_index, self.backend)
            else:
                self._camera = cv2.VideoCapture(self.camera_index)

            print("Camera opened")
            if not self._camera.isOpened():
                return False
                
            # Set resolution if specified
            if self.resolution:
                self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                
            
            # Set additional OpenCV properties for better performance
            self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency

            self._camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
            
            self._is_initialized = True
            print("Camera initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False
            
    def _capture_single_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame using OpenCV."""
        if not self._camera or not self._camera.isOpened():
            return None
            
        ret, frame = self._camera.read()
        if ret:
            return frame
        return None
        
    def _cleanup_camera(self):
        """Clean up OpenCV camera resources."""
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
            'backend': self._camera.getBackendName(),
            'brightness': self._camera.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self._camera.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self._camera.get(cv2.CAP_PROP_SATURATION),
            'hue': self._camera.get(cv2.CAP_PROP_HUE),
            'gain': self._camera.get(cv2.CAP_PROP_GAIN),
            'exposure': self._camera.get(cv2.CAP_PROP_EXPOSURE),
            'auto_exposure': self._camera.get(cv2.CAP_PROP_AUTO_EXPOSURE),
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
            property_id: OpenCV property ID (e.g., cv2.CAP_PROP_BRIGHTNESS)
            
        Returns:
            float: Property value or -1 if failed
        """
        if not self._camera or not self._camera.isOpened():
            return -1
        return self._camera.get(property_id)
    
    def set_brightness(self, brightness: float) -> bool:
        """Set camera brightness (0-100)."""
        return self.set_camera_property(cv2.CAP_PROP_BRIGHTNESS, brightness)
    
    def set_contrast(self, contrast: float) -> bool:
        """Set camera contrast (0-100)."""
        return self.set_camera_property(cv2.CAP_PROP_CONTRAST, contrast)
    
    def set_saturation(self, saturation: float) -> bool:
        """Set camera saturation (0-100)."""
        return self.set_camera_property(cv2.CAP_PROP_SATURATION, saturation)
    
    def set_exposure(self, exposure: float) -> bool:
        """Set camera exposure (-13 to -1, lower = brighter)."""
        return self.set_camera_property(cv2.CAP_PROP_EXPOSURE, exposure)
    
    def set_auto_exposure(self, auto: bool) -> bool:
        """Enable/disable auto exposure."""
        return self.set_camera_property(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if auto else 0.25)
