from __future__ import annotations

import numpy as np

from .base import Smoother
from .ema import EMASmoother
from .voting import VotingSmoother


class HybridSmoother(Smoother):
    """
    Hybrid smoother: EMA -> Voting

    Rationale:
    - EMA reduces short-term probability noise.
    - Voting enforces temporal label stability.

    Output is controlled by VotingSmoother (hard one-hot by default).
    """

    def __init__(self, alpha: float = 0.6, window: int = 7, hard_vote: bool = True):
        self.ema = EMASmoother(alpha=alpha)
        self.vote = VotingSmoother(window=window, hard=hard_vote)

    def reset(self) -> None:
        self.ema.reset()
        self.vote.reset()

    def update(self, probs: np.ndarray) -> np.ndarray:
        ema_probs = self.ema.update(probs)
        voted = self.vote.update(ema_probs)
        return voted