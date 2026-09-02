import argparse
import csv
from pathlib import Path
from typing import List, Optional


INVALID_LABEL = -1


def read_labels(path: Path, column: str) -> List[int]:
    labels: List[int] = []

    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(
                f"Column '{column}' not found in {path}"
            )

        for row in reader:
            labels.append(int(row[column]))

    return labels


def find_sustained_target(
    labels: List[int],
    target_label: int,
    *,
    start_frame: int = 0,
    stable_k: int = 5,
    lookahead: int = 30,
    minimum_ratio: float = 0.8,
) -> Optional[int]:
    """
    Find the first target-label run that is not merely temporary.

    Requirements:
    1. stable_k consecutive target predictions;
    2. target occupies at least minimum_ratio of the following
       lookahead frames.
    """

    if stable_k <= 0:
        raise ValueError("stable_k must be greater than 0")

    if lookahead <= 0:
        raise ValueError("lookahead must be greater than 0")

    if not 0.0 < minimum_ratio <= 1.0:
        raise ValueError(
            "minimum_ratio must be in the range (0, 1]"
        )

    start_frame = max(0, start_frame)

    for frame in range(
        start_frame,
        len(labels) - stable_k + 1,
    ):
        stable_window = labels[
            frame : frame + stable_k
        ]

        if not all(
            label == target_label
            for label in stable_window
        ):
            continue

        validation_end = min(
            len(labels),
            frame + stable_k + lookahead,
        )

        validation_window = labels[
            frame:validation_end
        ]

        valid_labels = [
            label
            for label in validation_window
            if label != INVALID_LABEL
        ]

        if not valid_labels:
            continue

        target_ratio = (
            sum(
                label == target_label
                for label in valid_labels
            )
            / len(valid_labels)
        )

        if target_ratio >= minimum_ratio:
            return frame

    return None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find a sustained target-expression onset "
            "from an FER prediction log."
        )
    )

    parser.add_argument(
        "--log",
        required=True,
        help="Path to an FER CSV log.",
    )

    parser.add_argument(
        "--target_label",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--column",
        choices=["y_raw", "y_smooth"],
        default="y_raw",
        help="Prediction column to inspect.",
    )

    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Ignore possible target predictions before this frame.",
    )

    parser.add_argument(
        "--stable_k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--lookahead",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--minimum_ratio",
        type=float,
        default=0.8,
    )

    args = parser.parse_args()

    log_path = Path(args.log)

    if not log_path.exists():
        raise FileNotFoundError(
            f"Log not found: {log_path}"
        )

    labels = read_labels(
        log_path,
        args.column,
    )

    onset = find_sustained_target(
        labels=labels,
        target_label=args.target_label,
        start_frame=args.start_frame,
        stable_k=args.stable_k,
        lookahead=args.lookahead,
        minimum_ratio=args.minimum_ratio,
    )

    print(f"Log: {log_path.name}")
    print(f"Column: {args.column}")
    print(f"Target label: {args.target_label}")

    if onset is None:
        print("Sustained target onset: not found")
    else:
        print(f"Sustained target onset: {onset}")


if __name__ == "__main__":
    main()