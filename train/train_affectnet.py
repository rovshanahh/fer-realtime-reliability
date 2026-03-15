from __future__ import annotations

import argparse
from pathlib import Path
import time
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from config import FERConfig
from model.resnet50_fer import ResNet50FER
from datasets.affectnet import AffectNetSubset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(cfg: FERConfig, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((cfg.input_size, cfg.input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg.input_size, cfg.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += float(loss.item()) * x.size(0)

        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

    return loss_sum / max(1, total), correct / max(1, total)


def set_trainable_params(model: ResNet50FER, ft_mode: str) -> None:
    """
    ft_mode:
      - "full": train all params
      - "fc": train only backbone.fc
      - "layer4_fc": train backbone.layer4 + backbone.fc
      - "layer3_layer4_fc": train backbone.layer3 + backbone.layer4 + backbone.fc
    """
    # Start by unfreezing everything (clean baseline)
    for p in model.parameters():
        p.requires_grad = True

    if ft_mode == "full":
        return

    # Freeze whole backbone
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Always unfreeze FC in these partial modes
    if not hasattr(model.backbone, "fc") or model.backbone.fc is None:
        raise RuntimeError("Expected model.backbone.fc to exist, but it doesn't.")
    for p in model.backbone.fc.parameters():
        p.requires_grad = True

    if ft_mode in ("layer4_fc", "layer3_layer4_fc"):
        if not hasattr(model.backbone, "layer4") or model.backbone.layer4 is None:
            raise RuntimeError("Expected model.backbone.layer4 to exist, but it doesn't.")
        for p in model.backbone.layer4.parameters():
            p.requires_grad = True

    if ft_mode == "layer3_layer4_fc":
        if not hasattr(model.backbone, "layer3") or model.backbone.layer3 is None:
            raise RuntimeError("Expected model.backbone.layer3 to exist, but it doesn't.")
        for p in model.backbone.layer3.parameters():
            p.requires_grad = True


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Root dir for images referenced by CSVs")
    ap.add_argument("--train_csv", required=True, help="e.g. data/affectnet_subset/train.csv")
    ap.add_argument("--val_csv", required=True, help="e.g. data/affectnet_subset/val.csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--resume", default=None, help="path to .pth to resume/fine-tune")
    ap.add_argument(
        "--ft_mode",
        choices=["full", "fc", "layer4_fc", "layer3_layer4_fc"],
        default="full"
    )
    ap.add_argument("--log_every", type=int, default=50, help="print progress every N batches")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = FERConfig()
    device = pick_device()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("device =", device)
    print("seed =", args.seed)
    print("cfg.num_classes =", cfg.num_classes)
    print("ft_mode =", args.ft_mode)

    tfm_train = build_transforms(cfg, train=True)
    tfm_val = build_transforms(cfg, train=False)

    ds_train = AffectNetSubset(args.data_root, args.train_csv, transform=tfm_train, num_classes=cfg.num_classes)
    ds_val = AffectNetSubset(args.data_root, args.val_csv, transform=tfm_val, num_classes=cfg.num_classes)

    print(f"Train samples: {len(ds_train)} (from {args.train_csv})")
    print(f"Val samples:   {len(ds_val)} (from {args.val_csv})")

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model = ResNet50FER(num_classes=cfg.num_classes, pretrained_imagenet=True).to(device)

    if args.resume:
        sd = torch.load(args.resume, map_location=device)
        model.load_state_dict(sd, strict=True)
        print(f"Loaded resume weights: {args.resume}")

    set_trainable_params(model, args.ft_mode)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found. Check ft_mode/model structure.")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_path = Path(args.out_dir) / f"best_{args.ft_mode}_affectnet_resnet50.pth"
    last_path = Path(args.out_dir) / f"last_{args.ft_mode}_affectnet_resnet50.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        seen = 0

        for batch_i, (x, y) in enumerate(dl_train, start=1):
            x, y = x.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            running += float(loss.item()) * x.size(0)
            seen += x.size(0)

            if args.log_every > 0 and batch_i % args.log_every == 0:
                print(
                    f"  batch {batch_i}/{len(dl_train)} | seen={seen}/{len(ds_train)} | loss={loss.item():.4f}",
                    flush=True
                )

        train_loss = running / max(1, seen)
        val_loss, val_acc = evaluate(model, dl_val, device)
        dt = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | {dt:.1f}s"
        )

        torch.save(model.state_dict(), last_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Saved best -> {best_path} (val_acc={best_val_acc:.4f})")

    print("\nDone.")
    print("Best checkpoint:", best_path)
    print("Last checkpoint:", last_path)


if __name__ == "__main__":
    main()