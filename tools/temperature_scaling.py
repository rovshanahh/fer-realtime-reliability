import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(
        logits,
        axis=1,
        keepdims=True,
    )

    exp_values = np.exp(shifted)

    return exp_values / np.sum(
        exp_values,
        axis=1,
        keepdims=True,
    )


def compute_calibration(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int,
) -> Dict:
    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    confidences = np.max(
        probabilities,
        axis=1,
    )

    correctness = (
        predictions == labels
    ).astype(np.float64)

    bin_edges = np.linspace(
        0.0,
        1.0,
        num_bins + 1,
    )

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
            mean_confidence - bin_accuracy
        )

        ece += (
            count / total
        ) * gap

        mce = max(
            mce,
            gap,
        )

        bins.append(
            {
                "bin_index": bin_index,
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": bin_accuracy,
                "gap": float(gap),
            }
        )

    return {
        "accuracy": float(
            correctness.mean()
        ),
        "mean_confidence": float(
            confidences.mean()
        ),
        "ece": float(ece),
        "mce": float(mce),
        "num_samples": int(total),
        "bins": bins,
    }


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iterations: int,
) -> float:
    logits_tensor = torch.tensor(
        logits,
        dtype=torch.float64,
    )

    labels_tensor = torch.tensor(
        labels,
        dtype=torch.long,
    )

    log_temperature = torch.nn.Parameter(
        torch.zeros(
            1,
            dtype=torch.float64,
        )
    )

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=max_iterations,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()

        temperature = torch.exp(
            log_temperature
        )

        loss = F.cross_entropy(
            logits_tensor / temperature,
            labels_tensor,
        )

        loss.backward()

        return loss

    optimizer.step(
        closure
    )

    temperature = float(
        torch.exp(
            log_temperature
        ).detach().item()
    )

    return temperature


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit temperature scaling on one split and "
            "evaluate calibration on a held-out split."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--num_bins",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--max_iterations",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    parser.add_argument(
        "--output",
        default=(
            "results/calibration/"
            "temperature_scaling.json"
        ),
    )

    args = parser.parse_args()

    if not 0.0 < args.test_size < 1.0:
        raise ValueError(
            "--test_size must be between 0 and 1"
        )

    input_path = Path(
        args.input
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    data = np.load(
        input_path
    )

    labels = np.asarray(
        data["labels"],
        dtype=np.int64,
    )

    logits = np.asarray(
        data["logits"],
        dtype=np.float64,
    )

    indices = np.arange(
        len(labels)
    )

    calibration_indices, evaluation_indices = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    calibration_logits = logits[
        calibration_indices
    ]

    calibration_labels = labels[
        calibration_indices
    ]

    evaluation_logits = logits[
        evaluation_indices
    ]

    evaluation_labels = labels[
        evaluation_indices
    ]

    temperature = fit_temperature(
        logits=calibration_logits,
        labels=calibration_labels,
        max_iterations=args.max_iterations,
    )

    probabilities_before = softmax_numpy(
        evaluation_logits
    )

    probabilities_after = softmax_numpy(
        evaluation_logits / temperature
    )

    before = compute_calibration(
        probabilities=probabilities_before,
        labels=evaluation_labels,
        num_bins=args.num_bins,
    )

    after = compute_calibration(
        probabilities=probabilities_after,
        labels=evaluation_labels,
        num_bins=args.num_bins,
    )

    result = {
        "method": "temperature scaling",
        "temperature": float(
            temperature
        ),
        "seed": int(
            args.seed
        ),
        "num_bins": int(
            args.num_bins
        ),
        "calibration_samples": int(
            len(calibration_indices)
        ),
        "evaluation_samples": int(
            len(evaluation_indices)
        ),
        "before_scaling": before,
        "after_scaling": after,
        "ece_absolute_reduction": float(
            before["ece"] - after["ece"]
        ),
        "ece_relative_reduction": float(
            (
                before["ece"] - after["ece"]
            )
            / before["ece"]
        ),
        "mce_absolute_reduction": float(
            before["mce"] - after["mce"]
        ),
    }

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print()
    print(
        f"Saved temperature-scaling results to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()