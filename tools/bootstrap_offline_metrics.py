import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import f1_score


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    num_bins: int,
) -> float:
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
            continue

        mean_confidence = float(
            confidences[mask].mean()
        )

        bin_accuracy = float(
            correctness[mask].mean()
        )

        ece += (
            count / total
        ) * abs(
            bin_accuracy - mean_confidence
        )

    return float(ece)


def confidence_interval(
    values: np.ndarray,
    confidence_level: float,
) -> Tuple[float, float]:
    alpha = 1.0 - confidence_level

    lower = float(
        np.quantile(
            values,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            values,
            1.0 - alpha / 2.0,
        )
    )

    return lower, upper


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap confidence intervals for "
            "AffectNet offline evaluation metrics."
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
        "--iterations",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--confidence_level",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--num_bins",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    parser.add_argument(
        "--output",
        default=(
            "results/bootstrap/"
            "layer3_affectnet_bootstrap.json"
        ),
    )

    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError(
            "--iterations must be greater than 0"
        )

    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError(
            "--confidence_level must be between 0 and 1"
        )

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    data = np.load(input_path)

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

    sample_count = len(labels)

    point_accuracy = float(
        np.mean(predictions == labels)
    )

    point_macro_f1 = float(
        f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    point_ece = compute_ece(
        confidences=confidences,
        predictions=predictions,
        labels=labels,
        num_bins=args.num_bins,
    )

    rng = np.random.default_rng(
        args.seed
    )

    bootstrap_accuracy = np.empty(
        args.iterations,
        dtype=np.float64,
    )

    bootstrap_macro_f1 = np.empty(
        args.iterations,
        dtype=np.float64,
    )

    bootstrap_ece = np.empty(
        args.iterations,
        dtype=np.float64,
    )

    for iteration in range(
        args.iterations
    ):
        indices = rng.integers(
            low=0,
            high=sample_count,
            size=sample_count,
        )

        sampled_labels = labels[indices]
        sampled_predictions = predictions[indices]
        sampled_confidences = confidences[indices]

        bootstrap_accuracy[iteration] = np.mean(
            sampled_predictions
            == sampled_labels
        )

        bootstrap_macro_f1[iteration] = f1_score(
            sampled_labels,
            sampled_predictions,
            average="macro",
            zero_division=0,
        )

        bootstrap_ece[iteration] = compute_ece(
            confidences=sampled_confidences,
            predictions=sampled_predictions,
            labels=sampled_labels,
            num_bins=args.num_bins,
        )

        if (
            iteration == 0
            or (iteration + 1) % 100 == 0
            or iteration + 1 == args.iterations
        ):
            print(
                f"Bootstrap iteration "
                f"{iteration + 1}/"
                f"{args.iterations}"
            )

    accuracy_ci = confidence_interval(
        bootstrap_accuracy,
        args.confidence_level,
    )

    macro_f1_ci = confidence_interval(
        bootstrap_macro_f1,
        args.confidence_level,
    )

    ece_ci = confidence_interval(
        bootstrap_ece,
        args.confidence_level,
    )

    result: Dict = {
        "num_samples": int(sample_count),
        "bootstrap_iterations": int(
            args.iterations
        ),
        "confidence_level": float(
            args.confidence_level
        ),
        "seed": int(args.seed),
        "num_bins": int(args.num_bins),
        "accuracy": {
            "point_estimate": point_accuracy,
            "ci_lower": accuracy_ci[0],
            "ci_upper": accuracy_ci[1],
        },
        "macro_f1": {
            "point_estimate": point_macro_f1,
            "ci_lower": macro_f1_ci[0],
            "ci_upper": macro_f1_ci[1],
        },
        "ece": {
            "point_estimate": point_ece,
            "ci_lower": ece_ci[0],
            "ci_upper": ece_ci[1],
        },
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

    print()
    print(
        json.dumps(
            result,
            indent=2,
        )
    )
    print()
    print(
        f"Saved bootstrap results to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()