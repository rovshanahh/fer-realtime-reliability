from __future__ import annotations

from typing import Optional
import numpy as np

from .base import Smoother


class EMASmoother(Smoother):
    """
    Exponential moving average smoothing on probability vectors:
        prev <- alpha * probs + (1-alpha) * prev

    Notes (research-friendly):
    - Normalizes output to sum to 1.
    - Keeps internal state (prev). Call reset() between runs if needed.
    """

    def __init__(self, alpha: float = 0.6, eps: float = 1e-12):
        if not (0.0 < float(alpha) <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.prev = None

    def update(self, probs: np.ndarray) -> np.ndarray:
        p = np.asarray(probs, dtype=np.float64)

        if p.ndim != 1 or p.size == 0:
            raise ValueError(f"Expected 1D probs vector, got shape={p.shape}")

        if not np.all(np.isfinite(p)):
            raise ValueError("probs contains non-finite values (nan/inf)")

        # EMA update
        if self.prev is None:
            self.prev = p.copy()
        else:
            if self.prev.shape != p.shape:
                raise ValueError(f"Shape changed across updates: prev={self.prev.shape}, current={p.shape}")
            self.prev = self.alpha * p + (1.0 - self.alpha) * self.prev

        # Normalize to probability simplex
        s = float(self.prev.sum())
        if s <= self.eps:
            # fallback: uniform if everything collapsed to ~0
            out = np.full_like(self.prev, 1.0 / self.prev.size, dtype=np.float64)
        else:
            out = self.prev / (s + self.eps)

        return out.astype(np.float32)