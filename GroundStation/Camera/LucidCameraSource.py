import cv2
import numpy as np
from typing import Any, Optional, Tuple
from Camera.CameraSource import CameraSource

from arena_api import enums
from arena_api.buffer import BufferFactory
from arena_api.system import system

class LucidCameraSource(CameraSource):
    """
    Camera source implementation using OpenCV VideoCapture.
    Provides full OpenCV integration with BGR frame format.
    """
    
    def __init__(self):
        """
        Initialize OpenCV camera source.
        """
        super().__init__()

        self._camera = None
        self._is_mono = True
        
    def _initialize_camera(self) -> bool:
        """Initialize OpenCV camera with specified backend and properties."""
        print("Starting camera initialization")

        device_infos = None
        device_infos = system.device_infos

        if len(device_infos) == 0:
            print("No camera connected")
            return False

        try:
            self._camera = system.create_device(device_infos=device_infos[0])[0]

            self._camera.tl_stream_nodemap.get_node('StreamBufferHandlingMode').value = 'NewestOnly'
            self._camera.tl_stream_nodemap.get_node('StreamPacketResendEnable').value = True
            self._camera.tl_stream_nodemap.get_node('StreamAutoNegotiatePacketSize').value = True

            self.set_camera_property('PixelFormat', 'BayerRG8')
            self.set_camera_property("TargetBrightness", 48)
            self.set_camera_property('GainAuto', 'Once')
            self.set_camera_property('ExposureAuto', 'Once')
            # self.set_camera_property('Gain', 10.0)

            self.set_camera_property('Width', 1440)
            self.set_camera_property('Height', 1080)

            self.set_camera_property('AcquisitionFrameRateEnable', True)
            self.set_camera_property('AcquisitionFrameRate', 60.0)

            self._is_mono = False

            self._camera.start_stream()

            return True
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False
            
    def _capture_single_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame using OpenCV."""
        if not self._camera:
            return None
        
        try:
            image_buffer = self._camera.get_buffer()  # optional args
            nparray = np.ctypeslib.as_array(image_buffer.pdata,shape=(image_buffer.height, image_buffer.width, int(image_buffer.bits_per_pixel / 8))).reshape(image_buffer.height, image_buffer.width, int(image_buffer.bits_per_pixel / 8))
            
            if not self._is_mono:
                display_img = cv2.cvtColor(nparray, cv2.COLOR_BayerBG2BGR)
            else:
                display_img = nparray
            
            self._camera.requeue_buffer(image_buffer)
            return display_img
        except Exception as e:
            print(f"Failed to capture frame: {e}")
            return None

        return None
        q
    def _cleanup_camera(self):
        """Clean up OpenCV camera resources."""
        if self._camera:
            self._camera.stop_stream()
            system.destroy_device()
            self._camera = None
        self._is_initialized = False

    def set_camera_property(self, property_id: int, value: Any) -> bool:
        """
        Set a camera property.
        
        Args:
            property_id: OpenCV property ID (e.g., cv2.CAP_PROP_BRIGHTNESS)
            value: Property value
            
        Returns:
            bool: True if set successfully, False otherwise
        """
        if not self._camera:
            return False
        try:
            self._camera.nodemap.get_node(property_id).value = value
            return True
        except Exception as e:
            print(f"Failed to set camera property: {e}")
            return False
    
    def get_camera_property(self, property_id: int) -> Any:
        """
        Get a camera property value.
        
        Args:
            property_id: OpenCV property ID (e.g., cv2.CAP_PROP_BRIGHTNESS)
            
        Returns:
            float: Property value or -1 if failed
        """
        if not self._camera:
            return "NA"
        return self._camera.nodemap.get_node(property_id).value
