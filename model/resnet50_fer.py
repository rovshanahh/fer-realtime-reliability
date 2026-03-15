from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResNet50FER(nn.Module):
    """
    ResNet-50 backbone adapted for Facial Expression Recognition (FER).

    Design choices (research-explicit):
    - Uses torchvision ResNet-50.
    - Optionally initialized with ImageNet weights.
    - Final fully connected layer replaced with a task-specific classifier.
    - Forward returns raw logits (softmax applied outside for flexibility).

    Assumptions:
    - Input tensor shape: (N, 3, H, W)
    - Input normalized with ImageNet mean/std.
    """

    def __init__(self, num_classes: int = 8, pretrained_imagenet: bool = True):
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {num_classes}")

        # Explicit weight selection for reproducibility
        weights = models.ResNet50_Weights.DEFAULT if pretrained_imagenet else None
        self.backbone = models.resnet50(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # Store for reference / logging
        self.num_classes = num_classes
        self.pretrained_imagenet = pretrained_imagenet

        self._init_classifier()

    def _init_classifier(self) -> None:
        """
        Initialize the classification head.
        ImageNet backbone weights are kept; FC layer is reinitialized.
        """
        nn.init.normal_(self.backbone.fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.backbone.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.size(1) != 3:
            raise ValueError(
                f"Expected input shape (N, 3, H, W), got {tuple(x.shape)}"
            )
        return self.backbone(x)
