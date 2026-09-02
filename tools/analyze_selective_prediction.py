import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import f1_score


def evaluate_threshold(
    labels: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    threshold: float,
) -> Dict:
    accepted_mask = confidences >= threshold

    accepted_count = int(
        accepted_mask.sum()
    )

    total_count = len(labels)

    coverage = (
        accepted_count / total_count
        if total_count > 0
        else 0.0
    )

    abstention_rate = (
        1.0 - coverage
    )

    if accepted_count == 0:
        selective_accuracy = None
        selective_macro_f1 = None
    else:
        accepted_labels = labels[
            accepted_mask
        ]

        accepted_predictions = predictions[
            accepted_mask
        ]

        selective_accuracy = float(
            np.mean(
                accepted_predictions
                == accepted_labels
            )
        )

        selective_macro_f1 = float(
            f1_score(
                accepted_labels,
                accepted_predictions,
                average="macro",
                labels=list(range(8)),
                zero_division=0,
            )
        )

    return {
        "threshold": float(threshold),
        "total_samples": int(total_count),
        "accepted_samples": accepted_count,
        "abstained_samples": int(
            total_count - accepted_count
        ),
        "coverage": float(coverage),
        "abstention_rate": float(
            abstention_rate
        ),
        "selective_accuracy": (
            selective_accuracy
        ),
        "selective_macro_f1": (
            selective_macro_f1
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate confidence-threshold selective "
            "prediction using saved offline predictions."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "NPZ file containing labels, predictions, "
            "and confidences."
        ),
    )

    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[
            0.0,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ],
    )

    parser.add_argument(
        "--output",
        default=(
            "results/selective_prediction/"
            "affectnet_threshold_analysis.json"
        ),
    )

    args = parser.parse_args()

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

    predictions = np.asarray(
        data["predictions"],
        dtype=np.int64,
    )

    confidences = np.asarray(
        data["confidences"],
        dtype=np.float64,
    )

    if not (
        len(labels)
        == len(predictions)
        == len(confidences)
    ):
        raise ValueError(
            "labels, predictions, and confidences "
            "must have equal lengths"
        )

    for threshold in args.thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "Every threshold must be between 0 and 1"
            )

    baseline_accuracy = float(
        np.mean(
            predictions == labels
        )
    )

    baseline_macro_f1 = float(
        f1_score(
            labels,
            predictions,
            average="macro",
            labels=list(range(8)),
            zero_division=0,
        )
    )

    results: List[Dict] = []

    for threshold in args.thresholds:
        result = evaluate_threshold(
            labels=labels,
            predictions=predictions,
            confidences=confidences,
            threshold=threshold,
        )

        results.append(
            result
        )

    payload = {
        "num_samples": int(
            len(labels)
        ),
        "baseline_accuracy": (
            baseline_accuracy
        ),
        "baseline_macro_f1": (
            baseline_macro_f1
        ),
        "results": results,
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
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )

    print()
    print(
        f"Saved threshold analysis to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()