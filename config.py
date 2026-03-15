'''from dataclasses import dataclass

@dataclass
class FERConfig:
    def __init__(self):
        # ----- model / dataset -----
        self.num_classes = 8
        self.class_names = [
            "neutral",
            "happy",
            "sad",
            "surprise",
            "fear",
            "disgust",
            "anger",
            "contempt"
        ]

        # ----- image preprocessing -----
        self.input_size = 224
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]

        # ----- runtime -----
        self.cam_index = 0

        # ----- smoothing params -----
        self.ema_alpha = 0.6
        self.voting_window = 7
        self.stable_k = 5'''


from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class FERConfig:
    # ----- model / dataset -----
    num_classes: int = 8
    class_names: List[str] = field(
        default_factory=lambda: [
            "neutral",
            "happy",
            "sad",
            "surprise",
            "fear",
            "disgust",
            "anger",
            "contempt",
        ]
    )

    # ----- image preprocessing -----
    input_size: int = 224
    imagenet_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    imagenet_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # ----- runtime -----
    cam_index: int = 0

    # ----- smoothing params -----
    ema_alpha: float = 0.6
    voting_window: int = 7
    stable_k: int = 5

    def __post_init__(self) -> None:
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {self.num_classes}")

        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"class_names length ({len(self.class_names)}) must equal num_classes ({self.num_classes})"
            )

        if self.input_size <= 0:
            raise ValueError(f"input_size must be > 0, got {self.input_size}")

        if len(self.imagenet_mean) != 3 or len(self.imagenet_std) != 3:
            raise ValueError("imagenet_mean and imagenet_std must be length 3 (RGB channels)")

        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {self.ema_alpha}")

        if self.voting_window <= 0:
            raise ValueError(f"voting_window must be > 0, got {self.voting_window}")

        if self.stable_k <= 0:
            raise ValueError(f"stable_k must be > 0, got {self.stable_k}")