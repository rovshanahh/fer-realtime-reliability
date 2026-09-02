import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from config import FERConfig
from model.resnet50_fer import ResNet50FER
from data.affectnet_subset.affectnet_csv import AffectNetCSVDataset

FER2013_ROOT = "/Users/rovshanahaji-hasanli/Downloads/archive"
CSV_PATH = "fer2013_test.csv"
CKPT_PATH = "checkpoints/best_layer4_fc_affectnet_resnet50.pth"


def build_transform(cfg: FERConfig):
    return transforms.Compose([
        transforms.Resize((cfg.input_size, cfg.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])


def compute_metrics(y_true, y_pred, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    total = cm.sum()
    acc = float(np.trace(cm) / total) if total > 0 else 0.0

    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1 = 2 * prec * rec / (prec + rec)
            f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return acc, macro_f1


def main():
    device = torch.device("cpu")

    cfg = FERConfig()
    transform = build_transform(cfg)

    dataset = AffectNetCSVDataset(
        csv_path=CSV_PATH,
        img_root=FER2013_ROOT,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = ResNet50FER(num_classes=cfg.num_classes, pretrained_imagenet=False)
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            print(f"Batch {i}", flush=True)

            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)

            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())

    acc, macro_f1 = compute_metrics(all_true, all_pred, num_classes=cfg.num_classes)
    print(f"FER2013 test accuracy: {acc:.4f}")
    print(f"FER2013 test macro F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()