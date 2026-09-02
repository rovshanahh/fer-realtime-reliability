import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from config import FERConfig
from datasets.affectnet import AffectNetSubset
from model.resnet50_fer import ResNet50FER


def build_loader(
    image_root_dir: str,
    val_csv: str,
    batch_size: int,
    cfg: FERConfig,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(
                (cfg.input_size, cfg.input_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.imagenet_mean,
                std=cfg.imagenet_std,
            ),
        ]
    )

    dataset = AffectNetSubset(
        root_dir=image_root_dir,
        csv_path=val_csv,
        transform=transform,
        num_classes=cfg.num_classes,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )


def compute_calibration(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    num_bins: int,
) -> Dict:
    bin_edges = np.linspace(
        0.0,
        1.0,
        num_bins + 1,
    )

    correctness = (
        predictions == labels
    ).astype(np.float64)

    total = len(labels)
    ece = 0.0
    mce = 0.0
    bins: List[Dict] = []

    for bin_index in range(num_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]

        if bin_index == num_bins - 1:
            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            mask = (
                (confidences >= lower)
                & (confidences < upper)
            )

        count = int(mask.sum())

        if count == 0:
            bins.append(
                {
                    "bin_index": bin_index,
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": 0,
                    "mean_confidence": None,
                    "accuracy": None,
                    "gap": None,
                }
            )
            continue

        mean_confidence = float(
            confidences[mask].mean()
        )

        bin_accuracy = float(
            correctness[mask].mean()
        )

        gap = abs(
            bin_accuracy - mean_confidence
        )

        weight = count / total
        ece += weight * gap
        mce = max(mce, gap)

        bins.append(
            {
                "bin_index": bin_index,
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": bin_accuracy,
                "gap": gap,
            }
        )

    return {
        "ece": float(ece),
        "mce": float(mce),
        "mean_confidence": float(
            confidences.mean()
        ),
        "accuracy": float(
            correctness.mean()
        ),
        "num_samples": int(total),
        "num_bins": int(num_bins),
        "bins": bins,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate confidence calibration on the "
            "AffectNet validation split."
        )
    )

    parser.add_argument(
        "--data_dir",
        required=True,
        help=(
            "AffectNet root directory containing "
            "Train and Test folders."
        ),
    )

    parser.add_argument(
        "--val_csv",
        default="data/affectnet_subset/val.csv",
    )

    parser.add_argument(
        "--weights",
        required=True,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num_bins",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--output",
        default=(
            "results/calibration/"
            "layer3_affectnet_calibration.json"
        ),
    )

    args = parser.parse_args()

    cfg = FERConfig()

    if torch.cuda.is_available():
        device = "cuda"
    elif (
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        device = "mps"
    else:
        device = "cpu"

    loader = build_loader(
        image_root_dir=args.data_dir,
        val_csv=args.val_csv,
        batch_size=args.batch_size,
        cfg=cfg,
    )

    model = ResNet50FER(
        num_classes=cfg.num_classes,
        pretrained_imagenet=False,
    ).to(device)

    weights_path = Path(args.weights)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights_path}"
        )

    state_dict = torch.load(
        str(weights_path),
        map_location=device,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    all_confidences: List[float] = []
    all_predictions: List[int] = []
    all_labels: List[int] = []
    all_logits: List[np.ndarray] = []

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(
            loader
        ):
            print(
                f"Batch {batch_index + 1}/"
                f"{len(loader)}"
            )

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            all_logits.extend(
                logits.cpu().numpy()
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1,
            )

            all_confidences.extend(
                confidence.cpu().numpy().tolist()
            )

            all_predictions.extend(
                prediction.cpu().numpy().tolist()
            )

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

    confidences_array = np.asarray(
        all_confidences,
        dtype=np.float64,
    )

    predictions_array = np.asarray(
        all_predictions,
        dtype=np.int64,
    )

    labels_array = np.asarray(
        all_labels,
        dtype=np.int64,
    )

    logits_array = np.asarray(
        all_logits,
        dtype=np.float64,
    )

    if logits_array.shape != (
        len(labels_array),
        cfg.num_classes,
    ):
        raise ValueError(
            "Unexpected logits shape: "
            f"{logits_array.shape}"
        )

    result = compute_calibration(
        confidences=confidences_array,
        predictions=predictions_array,
        labels=labels_array,
        num_bins=args.num_bins,
    )

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    predictions_path = output_path.with_name(
        output_path.stem
        + "_predictions.npz"
    )

    np.savez_compressed(
        predictions_path,
        labels=labels_array,
        predictions=predictions_array,
        confidences=confidences_array,
        logits=logits_array,
    )

    print()
    print(json.dumps(result, indent=2))
    print()

    print(
        f"Saved calibration results to: "
        f"{output_path}"
    )

    print(
        f"Saved per-image predictions and logits to: "
        f"{predictions_path}"
    )


if __name__ == "__main__":
    main()