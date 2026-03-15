from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

# Bounding box convention (IMPORTANT – research explicit):
# (x1, y1, x2, y2) with x2, y2 being EXCLUSIVE coordinates.
# Safe for NumPy slicing: img[y1:y2, x1:x2]
BBox = Tuple[int, int, int, int]


class FaceDetector(ABC):
    """
    Abstract base class for face detectors.

    Contract (research-critical):
    - Input: BGR image as np.ndarray of shape (H, W, 3), dtype uint8.
    - Output: Either
        * None (no valid face detected), or
        * BBox = (x1, y1, x2, y2) with x2/y2 EXCLUSIVE.
    - Returned bbox must be clipped to image bounds.
    - If multiple faces are detected, implementation must define
      how the "best" face is selected (e.g., largest area).

    This interface ensures detector interchangeability in experiments.
    """

    @abstractmethod
    def detect(self, bgr_frame: np.ndarray) -> Optional[BBox]:
        """
        Detect the primary face in a frame.

        Args:
            bgr_frame: OpenCV-style BGR image (H, W, 3).

        Returns:
            BBox or None.
        """
        raise NotImplementedError
