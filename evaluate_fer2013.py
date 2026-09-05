"""Cross-dataset evaluation of an AffectNet-trained ResNet-50 on FER2013.

This script replaces an earlier version that contained two defects. It loaded
the Layer4+FC checkpoint while the manuscript reports the Layer3+Layer4+FC
model, and it averaged Macro-F1 over all eight AffectNet classes even though
FER2013 contains only seven of them, so the unsupported contempt class
contributed a hard zero to the mean.

Label spaces
------------
AffectNet (model output, 8 classes)
    0 neutral  1 happy  2 sad  3 surprise  4 fear  5 disgust  6 anger  7 contempt
FER2013 (ground truth, 7 classes)
    0 neutral  1 happy  2 sad  3 surprise  4 fear  5 disgust  6 anger

The first seven indices coincide, so no remapping is needed. Contempt has no
FER2013 counterpart. Macro-F1 is therefore averaged over classes 0 to 6 only.

Two decision rules are reported from the same forward pass.
    unmasked  arg-max over all eight logits, so contempt can be predicted and
              every such prediction is necessarily wrong
    masked    the contempt logit is set to negative infinity before the
              arg-max, which isolates domain shift from the label-space
              mismatch

Usage
-----
    python evaluate_fer2013.py \
        --data-root /path/to/fer2013 \
        --csv fer2013_test.csv \
        --ckpt checkpoints/resnet50_layer3_finetuned_best.pt

The CSV is produced by datasets/make_fer2013_csv.py and holds a path column
relative to --data-root plus an integer label column.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from config import FERConfig
from model.resnet50_fer import ResNet50FER
from data.affectnet_subset.affectnet_csv import AffectNetCSVDataset

CONTEMPT_INDEX = 7
SHARED_CLASSES = list(range(7))


def build_transform(cfg: FERConfig):
    return transforms.Compose([
        transforms.Resize((cfg.input_size, cfg.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])


def macro_f1_over(y_true, y_pred, classes, num_classes):
    """Macro-F1 restricted to the given class indices.

    A class present in the reference labels but never predicted contributes a
    genuine zero. A class absent from the reference labels is excluded from
    the mean rather than contributing zero, which is the defect this script
    corrects.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    f1s = []
    for c in classes:
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        if tp + fn == 0:
            # no reference support for this class, exclude it from the mean
            continue
        if tp == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def evaluate(logits, labels, num_classes):
    labels = np.asarray(labels)

    pred_unmasked = logits.argmax(axis=1)

    masked = logits.copy()
    masked[:, CONTEMPT_INDEX] = -np.inf
    pred_masked = masked.argmax(axis=1)

    return {
        "n_images": int(len(labels)),
        "unmasked": {
            "accuracy": float((pred_unmasked == labels).mean()),
            "macro_f1": macro_f1_over(labels, pred_unmasked, SHARED_CLASSES, num_classes),
            "contempt_predictions": int((pred_unmasked == CONTEMPT_INDEX).sum()),
            "contempt_share": float((pred_unmasked == CONTEMPT_INDEX).mean()),
        },
        "masked": {
            "accuracy": float((pred_masked == labels).mean()),
            "macro_f1": macro_f1_over(labels, pred_masked, SHARED_CLASSES, num_classes),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True,
                    help="directory the CSV path column is relative to")
    ap.add_argument("--csv", default="fer2013_test.csv")
    ap.add_argument("--ckpt", default="checkpoints/resnet50_layer3_finetuned_best.pt",
                    help="Layer3+Layer4+FC checkpoint reported in the manuscript")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="fer2013_results.json")
    args = ap.parse_args()

    cfg = FERConfig()
    device = torch.device(args.device)

    dataset = AffectNetCSVDataset(
        csv_path=args.csv,
        img_root=args.data_root,
        transform=build_transform(cfg),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False)

    model = ResNet50FER(num_classes=cfg.num_classes, pretrained_imagenet=False)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device).eval()

    chunks_logits, chunks_labels = [], []
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            chunks_logits.append(model(images.to(device)).cpu().numpy())
            chunks_labels.append(labels.numpy())
            if i % 20 == 0:
                print(f"batch {i}/{len(loader)}", flush=True)

    logits = np.concatenate(chunks_logits)
    labels = np.concatenate(chunks_labels)

    res = evaluate(logits, labels, cfg.num_classes)
    res["checkpoint"] = args.ckpt

    u, m = res["unmasked"], res["masked"]
    print()
    print(f"images evaluated            {res['n_images']}")
    print(f"checkpoint                  {args.ckpt}")
    print()
    print("all eight logits retained")
    print(f"  accuracy                  {u['accuracy']:.4f}")
    print(f"  macro F1 (7 classes)      {u['macro_f1']:.4f}")
    print(f"  predicted contempt        {u['contempt_predictions']} "
          f"({u['contempt_share'] * 100:.2f}% of images, all necessarily wrong)")
    print()
    print("contempt logit masked")
    print(f"  accuracy                  {m['accuracy']:.4f}")
    print(f"  macro F1 (7 classes)      {m['macro_f1']:.4f}")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
