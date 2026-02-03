from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

class VisionTarget(ABC):
    @abstractmethod
    def get_position(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Returns a dict with at least:
          - rvec: (3,1) rotation vector of target in camera frame
          - tvec: (3,1) translation vector of target in camera frame (meters)
        Optional keys may include detection metadata used for drawing overlays.
        """
        raise NotImplementedError
