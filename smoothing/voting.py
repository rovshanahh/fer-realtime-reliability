from __future__ import annotations

from collections import Counter, deque
from typing import Deque, Optional
import numpy as np

from .base import Smoother


class VotingSmoother(Smoother):
    """
    Majority-vote smoothing over the last `window` argmax labels.

    Output:
    - By default returns a one-hot probability vector (hard vote).
      This is intentional but can be changed by setting hard=False to
      return a soft distribution over the vote counts.
    """

    def __init__(self, window: int = 7, hard: bool = True):
        window = int(window)
        if window <= 0:
            raise ValueError(f"window must be > 0, got {window}")
        self.window = window
        self.hard = bool(hard)
        self.labels: Deque[int] = deque(maxlen=self.window)

    def reset(self) -> None:
        self.labels.clear()

    def update(self, probs: np.ndarray) -> np.ndarray:
        p = np.asarray(probs, dtype=np.float32)

        if p.ndim != 1 or p.size == 0:
            raise ValueError(f"Expected 1D probs vector, got shape={p.shape}")

        if not np.all(np.isfinite(p)):
            raise ValueError("probs contains non-finite values (nan/inf)")

        y = int(np.argmax(p))
        self.labels.append(y)

        counts = Counter(self.labels)
        voted = counts.most_common(1)[0][0]

        if self.hard:
            out = np.zeros_like(p, dtype=np.float32)
            out[voted] = 1.0
            return out

        # soft vote: normalize counts into a distribution
        out = np.zeros_like(p, dtype=np.float32)
        denom = float(sum(counts.values()))
        for cls, c in counts.items():
            if 0 <= cls < out.size:
                out[cls] = float(c) / max(1e-12, denom)
        return out