from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class Smoother(ABC):
    """
    Interface for temporal smoothing of class probability vectors.

    Contract:
    - update(probs) takes a 1D array-like of shape (C,)
    - returns a 1D np.ndarray of shape (C,)
    - returned values should be finite; ideally sum to 1 (probability simplex)
    """

    @abstractmethod
    def update(self, probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError
