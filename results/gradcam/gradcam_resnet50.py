import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# Class names (MUST match training & reports)
# ============================================================
CLASS_NAMES = [
    "neutral", "happy", "sad", "surprise",
    "fear", "disgust", "anger", "contempt"
]


# ============================================================
# Model wrapper (matches checkpoint: backbone.*)
# ============================================================
class FERBackboneWrapper(torch.nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = torch.nn.Linear(
            self.backbone.fc.in_features, num_classes
        )

    def forward(self, x):
        return self.backbone(x)


# ============================================================
# Strict checkpoint loader (FAIL FAST, no silent bugs)
# ============================================================
def load_checkpoint_strict(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model"]:
            if k in ckpt:
                state_dict = ckpt[k]
                break
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    new_sd = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_sd[k] = v

    model.load_state_dict(new_sd, strict=True)
    print(f"[OK] Strictly loaded checkpoint: {ckpt_path}")
    return model


# ============================================================
# Grad-CAM implementation
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.hook_a = target_layer.register_forward_hook(self._forward_hook)
        self.hook_g = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove_hooks(self):
        self.hook_a.remove()
        self.hook_g.remove()

    def __call__(self, x, class_idx):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        score = logits[0, class_idx]
        score.backward()

        A = self.activations       # (1, C, h, w)
        dA = self.gradients        # (1, C, h, w)

        weights = dA.mean(dim=(2, 3), keepdim=True)
        cam = (weights * A).sum(dim=1)
        cam = F.relu(cam)[0]

        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam.cpu().numpy(), probs[0, class_idx].item()


# ============================================================
# Preprocessing (match ImageNet-trained ResNet)
# ============================================================
def get_preprocess(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])


# ============================================================
# Visualization utilities
# ============================================================
def overlay_cam(pil_img, cam, alpha=0.45):
    img = np.array(pil_img).astype(np.float32) / 255.0
    cam_img = Image.fromarray((cam * 255).astype(np.uint8))
    cam_img = cam_img.resize((img.shape[1], img.shape[0]))
    cam = np.array(cam_img).astype(np.float32) / 255.0

    heatmap = plt.get_cmap("jet")(cam)[..., :3]
    overlay = (1 - alpha) * img + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


def save_figure(pil_img, overlay, out_path, title):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(pil_img)
    plt.axis("off")
    plt.title("Input")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Grad-CAM")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--outdir", default="results/gradcam/out")
    parser.add_argument("--target", default="pred", help="pred or class index")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--alpha", type=float, default=0.45)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load image
    pil_img = Image.open(args.image).convert("RGB")
    x = get_preprocess(args.img_size)(pil_img).unsqueeze(0).to(device)

    # Load model
    model = FERBackboneWrapper(num_classes=len(CLASS_NAMES))
    model = load_checkpoint_strict(model, args.weights, device)
    model.to(device)
    model.eval()

    # Predict
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))

    class_idx = pred_idx if args.target == "pred" else int(args.target)

    # Grad-CAM
    cam_engine = GradCAM(model, model.backbone.layer4[-1])
    cam, cam_prob = cam_engine(x, class_idx)
    cam_engine.remove_hooks()

    overlay = overlay_cam(pil_img, cam, alpha=args.alpha)

    title = (
        f"Pred: {CLASS_NAMES[pred_idx]} (p={probs[pred_idx]:.3f}) | "
        f"Grad-CAM: {CLASS_NAMES[class_idx]} (p={cam_prob:.3f})"
    )

    out_path = outdir / f"gradcam_{Path(args.image).stem}_{CLASS_NAMES[class_idx]}.png"
    save_figure(pil_img, overlay, out_path, title)

    print("Saved:", out_path)
    print("Top-3 predictions:")
    for i in probs.argsort()[-3:][::-1]:
        print(f"  {i:02d} {CLASS_NAMES[i]:<9} p={probs[i]:.4f}")


if __name__ == "__main__":
    main()
