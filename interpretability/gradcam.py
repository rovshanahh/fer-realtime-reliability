from __future__ import annotations

from typing import Optional, Callable
import numpy as np
import cv2
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM for CNN-style feature maps.

    Notes (research-friendly):
    - Works for classification models that output logits of shape (B, C).
    - target_layer must produce activations of shape (B, K, H, W).
    - Uses global-average-pooled gradients as weights.
    - Returns a normalized CAM in [0, 1] as a numpy array (H, W) for the first batch item.

    Important:
    - We DO NOT decorate generate() with @torch.no_grad() because it needs gradients.
    - This class does not change model mode; caller should set model.eval() if desired.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def fwd_hook(_module, _inputs, output):
            # output expected shape: (B, K, H, W)
            self.activations = output.detach()

        def bwd_hook(_module, grad_input, grad_output):
            # grad_output[0] expected shape: (B, K, H, W)
            self.gradients = grad_output[0].detach()

        self._hook_handles.append(self.target_layer.register_forward_hook(fwd_hook))
        # full backward hook is the recommended modern hook
        self._hook_handles.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def close(self) -> None:
        """Remove hooks to avoid memory leaks in long-running processes."""
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()

    @staticmethod
    def _normalize(cam: np.ndarray) -> np.ndarray:
        cam = cam.astype(np.float32, copy=False)
        mn = float(cam.min())
        mx = float(cam.max())
        cam = cam - mn
        denom = (mx - mn) + 1e-12
        cam = cam / denom
        cam = np.clip(cam, 0.0, 1.0)
        return cam

    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM for the first element of the batch.

        Args:
            x: input tensor of shape (B, 3, H, W)
            class_idx: class index to explain. If None, uses argmax(logits) for item 0.

        Returns:
            cam_np: numpy array (H', W') normalized to [0, 1] (spatial size of target_layer output).
        """
        if x.ndim != 4:
            raise ValueError(f"Expected input with shape (B, C, H, W), got {tuple(x.shape)}")

        # Clear any stale state
        self.gradients = None
        self.activations = None

        # Make sure gradients are enabled (in case caller is in no_grad)
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)

            logits = self.model(x)  # expected shape (B, num_classes)
            if logits.ndim != 2:
                raise ValueError(f"Expected model output shape (B, num_classes), got {tuple(logits.shape)}")

            if class_idx is None:
                class_idx = int(torch.argmax(logits[0]).item())
            else:
                class_idx = int(class_idx)
                if class_idx < 0 or class_idx >= logits.size(1):
                    raise ValueError(f"class_idx out of range: {class_idx} (num_classes={logits.size(1)})")

            score = logits[:, class_idx].sum()
            score.backward(retain_graph=False)

            if self.gradients is None or self.activations is None:
                raise RuntimeError(
                    "GradCAM hooks did not capture activations/gradients. "
                    "Check that target_layer is used in the forward pass and produces (B,K,H,W)."
                )

            grads = self.gradients      # (B, K, H, W)
            acts = self.activations     # (B, K, H, W)

            if grads.ndim != 4 or acts.ndim != 4:
                raise ValueError(f"Expected grads/acts to be 4D, got grads={grads.shape}, acts={acts.shape}")

            # weights: (B, K, 1, 1)
            weights = grads.mean(dim=(2, 3), keepdim=True)

            # cam: (B, H, W)
            cam = (weights * acts).sum(dim=1)
            cam = F.relu(cam)

            cam_np = cam[0].detach().cpu().numpy()
            return self._normalize(cam_np)


def overlay_cam_on_bgr(bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay a normalized CAM (H, W) in [0, 1] onto a BGR image.

    Args:
        bgr: image (H, W, 3) uint8
        cam: heatmap (h, w) float in [0, 1] (will be resized to image size)
        alpha: blending strength for heatmap

    Returns:
        blended BGR uint8 image
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image (H, W, 3), got {bgr.shape}")

    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    h, w = bgr.shape[:2]
    cam_resized = cv2.resize(cam.astype(np.float32, copy=False), (w, h), interpolation=cv2.INTER_LINEAR)
    cam_resized = np.clip(cam_resized, 0.0, 1.0)

    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(bgr, 1.0 - float(alpha), heatmap, float(alpha), 0)
    return out
