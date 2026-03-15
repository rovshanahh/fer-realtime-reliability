import argparse
from pathlib import Path
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from config import FERConfig
from model.resnet50_fer import ResNet50FER
from model.utils import set_finetune_resnet50
from datasets.affectnet import AffectNetSubset


def build_loaders(
    image_root_dir: str,
    train_csv: str,
    val_csv: str,
    input_size: int,
    batch_size: int,
    cfg: FERConfig,
):
    tfm = transforms.Compose([
        transforms.ToPILImage(),  # dataset returns RGB numpy (cv2)
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])

    train_csv = Path(train_csv)
    val_csv = Path(val_csv)

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train.csv at: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Missing val.csv at: {val_csv}")

    train_ds = AffectNetSubset(
        root_dir=image_root_dir,
        csv_path=str(train_csv),
        transform=tfm,
        num_classes=cfg.num_classes,
    )
    val_ds = AffectNetSubset(
        root_dir=image_root_dir,
        csv_path=str(val_csv),
        transform=tfm,
        num_classes=cfg.num_classes,
    )

    # macOS/MPS: keep pin_memory off; num_workers small & stable
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, topk=(1,)):
    model.eval()

    ys, yhats = [], []
    correct1, total = 0, 0
    topk_correct = {k: 0 for k in topk}

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        correct1 += (pred == y).sum().item()
        total += y.numel()

        ys.extend(y.detach().cpu().tolist())
        yhats.extend(pred.detach().cpu().tolist())

        if topk:
            maxk = max(topk)
            topk_preds = torch.topk(logits, k=maxk, dim=1).indices
            for k in topk:
                hit = (topk_preds[:, :k] == y.unsqueeze(1)).any(dim=1).sum().item()
                topk_correct[k] += hit

    acc = 100.0 * correct1 / max(total, 1)
    macro_f1 = f1_score(ys, yhats, average="macro")
    weighted_f1 = f1_score(ys, yhats, average="weighted")

    cm = confusion_matrix(ys, yhats, labels=list(range(num_classes)))
    report_dict = classification_report(
        ys, yhats,
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0
    )

    topk_acc = {k: 100.0 * topk_correct[k] / max(total, 1) for k in topk}

    metrics = {
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "topk_acc": topk_acc
    }

    return metrics, cm, report_dict


def save_eval_artifacts(out_dir: Path, tag: str, cfg: FERConfig, metrics: dict, cm: np.ndarray, report_dict: dict):
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"{tag}_confusion.npy", cm)

    payload = {
        "tag": tag,
        "class_names": cfg.class_names,
        "metrics": metrics,
        "classification_report": report_dict,
        "confusion_matrix_shape": list(cm.shape),
    }

    with (out_dir / f"{tag}_report.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with (out_dir / f"{tag}_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"TAG: {tag}\n")
        f.write(f"ACC: {metrics['acc']:.2f}%\n")
        f.write(f"MACRO_F1: {metrics['macro_f1']:.4f}\n")
        f.write(f"WEIGHTED_F1: {metrics['weighted_f1']:.4f}\n")
        if "topk_acc" in metrics:
            for k, v in metrics["topk_acc"].items():
                f.write(f"TOP{k}_ACC: {v:.2f}%\n")

        f.write("\nPer-class (precision / recall / f1):\n")
        for i, name in enumerate(cfg.class_names):
            row = report_dict.get(str(i), {})
            p = row.get("precision", 0.0)
            r = row.get("recall", 0.0)
            f1v = row.get("f1-score", 0.0)
            f.write(f"- {i:02d} {name}: P={p:.4f} R={r:.4f} F1={f1v:.4f}\n")


def train_one(model, train_loader, val_loader, device, epochs, lr, save_path: Path, cfg: FERConfig):
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")

        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=float(loss.item()))

        metrics, _, _ = evaluate(model, val_loader, device, num_classes=cfg.num_classes, topk=(1, 3))
        print(
            f"Val: acc={metrics['acc']:.2f}% "
            f"macroF1={metrics['macro_f1']:.4f} "
            f"weightedF1={metrics['weighted_f1']:.4f} "
            f"top3={metrics['topk_acc'].get(3, 0.0):.2f}%"
        )

        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            torch.save(model.state_dict(), str(save_path))
            print(f"Saved best model to: {save_path}")

    if save_path.exists():
        model.load_state_dict(torch.load(str(save_path), map_location=device))

    metrics, _, _ = evaluate(model, val_loader, device, num_classes=cfg.num_classes, topk=(1, 3))
    print(
        f"[BEST CHECKPOINT] acc={metrics['acc']:.2f}% "
        f"macroF1={metrics['macro_f1']:.4f} "
        f"weightedF1={metrics['weighted_f1']:.4f} "
        f"top3={metrics['topk_acc'].get(3, 0.0):.2f}%"
    )
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Image root dir (must contain Train/ and Test/ folders)")
    ap.add_argument("--train_csv", default="data/affectnet_subset/train.csv")
    ap.add_argument("--val_csv", default="data/affectnet_subset/val.csv")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mode", choices=["frozen", "finetune", "layer3"], required=True)
    ap.add_argument("--out", default="results/checkpoints")
    args = ap.parse_args()

    cfg = FERConfig()

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_loaders(
        image_root_dir=args.data_dir,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        input_size=cfg.input_size,
        batch_size=args.batch_size,
        cfg=cfg,
    )

    model = ResNet50FER(num_classes=cfg.num_classes, pretrained_imagenet=True).to(device)

    if args.mode == "frozen":
        set_finetune_resnet50(model, train_fc=True, unfreeze_layer3=False, unfreeze_layer4=False)
        save_path = out_dir / "resnet50_frozen_best.pt"
        tag = "frozen_resnet50"

    elif args.mode == "finetune":
        set_finetune_resnet50(model, train_fc=True, unfreeze_layer3=False, unfreeze_layer4=True)
        save_path = out_dir / "resnet50_finetuned_best.pt"
        tag = "finetune_resnet50"

    else:  # layer3
        set_finetune_resnet50(model, train_fc=True, unfreeze_layer3=True, unfreeze_layer4=True)
        save_path = out_dir / "resnet50_layer3_finetuned_best.pt"
        tag = "layer3_finetune_resnet50"

    _ = train_one(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_path=save_path,
        cfg=cfg,
    )

    reports_dir = Path("results/reports")
    model.load_state_dict(torch.load(str(save_path), map_location=device))
    metrics, cm, report_dict = evaluate(model, val_loader, device, num_classes=cfg.num_classes, topk=(1, 3))
    save_eval_artifacts(reports_dir, tag, cfg, metrics, cm, report_dict)

    print(
        f"[FINAL] mode={args.mode} "
        f"acc={metrics['acc']:.2f}% "
        f"macroF1={metrics['macro_f1']:.4f} "
        f"weightedF1={metrics['weighted_f1']:.4f}"
    )
    print(f"[SAVED] checkpoint: {save_path}")
    print(f"[SAVED] reports: {reports_dir} (tag={tag})")


if __name__ == "__main__":
    main()