from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import FaceDetector, BBox


class HaarCascadeDetector(FaceDetector):
    """
    OpenCV Haar cascade face detector.

    Research notes:
    - Deterministic given the same OpenCV build and parameters.
    - Typically less robust than deep detectors; useful as a classical baseline.
    - Returns the largest detected face by area (common "primary subject" heuristic).

    BBox convention: (x1, y1, x2, y2) ints in pixel coordinates, clipped to image bounds.
    """

    def __init__(
        self,
        cascade_path: Optional[str] = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (60, 60),
    ):
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise ValueError(f"Failed to load Haar cascade from: {cascade_path}")

        if scale_factor <= 1.0:
            raise ValueError(f"scale_factor must be > 1.0, got {scale_factor}")
        if min_neighbors < 0:
            raise ValueError(f"min_neighbors must be >= 0, got {min_neighbors}")
        if min_size[0] <= 0 or min_size[1] <= 0:
            raise ValueError(f"min_size entries must be > 0, got {min_size}")

        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_size = (int(min_size[0]), int(min_size[1]))

    def detect(self, bgr_frame: np.ndarray) -> Optional[BBox]:
        if bgr_frame is None or bgr_frame.size == 0:
            return None

        if bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
            raise ValueError(f"Expected BGR image (H, W, 3), got shape={bgr_frame.shape}")

        h, w = bgr_frame.shape[:2]
        if h < 2 or w < 2:
            return None

        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        if faces is None or len(faces) == 0:
            return None

        # Select the largest face (by area)
        x, y, fw, fh = max(faces, key=lambda f: int(f[2]) * int(f[3]))

        x1, y1 = int(x), int(y)
        x2, y2 = int(x + fw), int(y + fh)

        # Clip to image bounds (x2/y2 are treated as exclusive here, but we'll clip safely)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # Return inclusive-style bbox if you standardize that elsewhere:
        # If the rest of your pipeline expects x2/y2 inclusive, change to (min(w-1,x2-1), min(h-1,y2-1)).
        return (x1, y1, x2, y2)
