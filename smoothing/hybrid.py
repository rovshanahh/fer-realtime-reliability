from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Smoother
from .ema import EMASmoother
from .voting import VotingSmoother


class HybridSmoother(Smoother):
    """Hybrid smoother: EMA followed by majority voting."""

    def __init__(
        self,
        alpha: float = 0.6,
        window: int = 7,
        hard_vote: bool = True,
    ):
        self.ema = EMASmoother(alpha=alpha)
        self.vote = VotingSmoother(
            window=window,
            hard=hard_vote,
        )
        self.last_ema_probs: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.ema.reset()
        self.vote.reset()
        self.last_ema_probs = None

    def update(self, probs: np.ndarray) -> np.ndarray:
        ema_probs = np.asarray(
            self.ema.update(probs),
            dtype=np.float32,
        )

        self.last_ema_probs = ema_probs.copy()

        return self.vote.update(ema_probs)