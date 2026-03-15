from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from .base import FaceDetector, BBox


class RetinaFaceDetector(FaceDetector):
    """
    Deep face detector using InsightFace (RetinaFace under the hood) + ONNXRuntime.

    Research notes:
    - Uses CPUExecutionProvider by default for macOS reproducibility.
    - Returns the largest detected face (by area) that passes a score threshold.
    - BBox convention: (x1, y1, x2, y2) with x2/y2 as EXCLUSIVE coordinates
      (safe for NumPy slicing: img[y1:y2, x1:x2]).

    Requirements:
      pip install insightface onnxruntime
    """

    def __init__(
        self,
        det_size: Tuple[int, int] = (640, 640),
        threshold: float = 0.6,
        providers: Optional[list[str]] = None,
    ):
        if det_size[0] <= 0 or det_size[1] <= 0:
            raise ValueError(f"det_size must contain positive ints, got {det_size}")

        self.threshold = float(threshold)
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")

        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                "InsightFace is required for RetinaFaceDetector. "
                "Install with: pip install insightface onnxruntime"
            ) from e

        if providers is None:
            providers = ["CPUExecutionProvider"]

        # CPU provider is generally the most reliable baseline on macOS
        self.app = FaceAnalysis(providers=providers)
        self.app.prepare(ctx_id=-1, det_size=det_size)  # -1 is CPU-safe / GPU-independent

    def detect(self, bgr_frame: np.ndarray) -> Optional[BBox]:
        if bgr_frame is None or bgr_frame.size == 0:
            return None

        if bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
            raise ValueError(f"Expected BGR image (H, W, 3), got shape={bgr_frame.shape}")

        h, w = bgr_frame.shape[:2]
        if h < 2 or w < 2:
            return None

        faces = self.app.get(bgr_frame)
        if not faces:
            return None

        best: Optional[BBox] = None
        best_area = 0

        for f in faces:
            score = float(getattr(f, "det_score", 0.0))
            if score < self.threshold:
                continue

            # InsightFace bbox is typically (x1, y1, x2, y2) with float coords
            x1, y1, x2, y2 = [int(v) for v in f.bbox]

            # Clip to image bounds using EXCLUSIVE max (w/h), which matches slicing
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)

        return best
