import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


INVALID_LABEL = -1
PROBABILITY_COLUMNS = [f"p{i}" for i in range(8)]


def read_log(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels: List[int] = []
    probabilities: List[List[float]] = []
    face_detected: List[int] = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                f"Missing CSV header: {path}"
            )

        required_columns = [
            "face_detected",
            "y_smooth",
            *PROBABILITY_COLUMNS,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in reader.fieldnames
        ]

        if missing_columns:
            raise ValueError(
                f"{path} is missing columns: "
                f"{missing_columns}"
            )

        for row in reader:
            labels.append(
                int(row["y_smooth"])
            )

            face_detected.append(
                int(row["face_detected"])
            )

            probabilities.append(
                [
                    float(row[column])
                    for column in PROBABILITY_COLUMNS
                ]
            )

    return (
        np.asarray(
            labels,
            dtype=np.int64,
        ),
        np.asarray(
            probabilities,
            dtype=np.float64,
        ),
        np.asarray(
            face_detected,
            dtype=np.int64,
        ),
    )


def full_sequence_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    face_detected: np.ndarray,
) -> Dict[str, float]:
    valid_mask = labels != INVALID_LABEL

    valid_labels = labels[valid_mask]
    valid_probabilities = probabilities[valid_mask]

    valid_count = len(valid_labels)

    if valid_count < 2:
        lfr = 0.0
        pj_l1 = 0.0
    else:
        lfr = float(
            np.mean(
                valid_labels[1:]
                != valid_labels[:-1]
            )
        )

        pj_l1 = float(
            np.mean(
                np.abs(
                    valid_probabilities[1:]
                    - valid_probabilities[:-1]
                ).sum(axis=1)
            )
        )

    if valid_count == 0:
        msl = 0.0
    else:
        flips = int(
            np.sum(
                valid_labels[1:]
                != valid_labels[:-1]
            )
        )

        segment_count = flips + 1
        msl = float(
            valid_count / segment_count
        )

    fddr = (
        float(
            1.0
            - np.mean(face_detected)
        )
        if len(face_detected) > 0
        else 0.0
    )

    return {
        "LFR": lfr,
        "MSL": msl,
        "PJ_L1": pj_l1,
        "FDDR": fddr,
    }


def sample_block_starts(
    sample_count: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    number_of_blocks = int(
        np.ceil(
            sample_count / block_size
        )
    )

    maximum_start = (
        sample_count - block_size
    )

    if maximum_start < 0:
        return np.zeros(
            number_of_blocks,
            dtype=np.int64,
        )

    return rng.integers(
        low=0,
        high=maximum_start + 1,
        size=number_of_blocks,
    )


def calculate_block_bootstrap_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    face_detected: np.ndarray,
    block_starts: np.ndarray,
    block_size: int,
    target_length: int,
) -> Dict[str, float]:
    total_valid_frames = 0
    total_valid_pairs = 0
    total_flips = 0
    total_probability_jitter = 0.0

    sampled_face_values: List[np.ndarray] = []

    frames_remaining = target_length

    for start in block_starts:
        if frames_remaining <= 0:
            break

        current_size = min(
            block_size,
            frames_remaining,
        )

        end = int(start) + current_size

        block_labels = labels[
            int(start):end
        ]

        block_probabilities = probabilities[
            int(start):end
        ]

        block_face_detected = face_detected[
            int(start):end
        ]

        sampled_face_values.append(
            block_face_detected
        )

        valid_mask = (
            block_labels != INVALID_LABEL
        )

        valid_labels = block_labels[
            valid_mask
        ]

        valid_probabilities = (
            block_probabilities[
                valid_mask
            ]
        )

        valid_count = len(
            valid_labels
        )

        total_valid_frames += valid_count

        if valid_count >= 2:
            pair_count = (
                valid_count - 1
            )

            flips = int(
                np.sum(
                    valid_labels[1:]
                    != valid_labels[:-1]
                )
            )

            jitter_sum = float(
                np.abs(
                    valid_probabilities[1:]
                    - valid_probabilities[:-1]
                )
                .sum(axis=1)
                .sum()
            )

            total_valid_pairs += pair_count
            total_flips += flips
            total_probability_jitter += (
                jitter_sum
            )

        frames_remaining -= current_size

    if total_valid_pairs > 0:
        lfr = float(
            total_flips
            / total_valid_pairs
        )

        pj_l1 = float(
            total_probability_jitter
            / total_valid_pairs
        )
    else:
        lfr = 0.0
        pj_l1 = 0.0

    if total_valid_frames > 0:
        estimated_segments = (
            1.0
            + lfr
            * (
                total_valid_frames - 1
            )
        )

        msl = float(
            total_valid_frames
            / estimated_segments
        )
    else:
        msl = 0.0

    if sampled_face_values:
        sampled_face_detected = (
            np.concatenate(
                sampled_face_values
            )[:target_length]
        )

        fddr = float(
            1.0
            - np.mean(
                sampled_face_detected
            )
        )
    else:
        fddr = 0.0

    return {
        "LFR": lfr,
        "MSL": msl,
        "PJ_L1": pj_l1,
        "FDDR": fddr,
    }


def percentile_interval(
    values: np.ndarray,
    confidence_level: float,
) -> Tuple[float, float]:
    alpha = (
        1.0 - confidence_level
    )

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


def bootstrap_log(
    path: Path,
    iterations: int,
    block_size: int,
    confidence_level: float,
    seed: int,
) -> Dict:
    (
        labels,
        probabilities,
        face_detected,
    ) = read_log(path)

    sample_count = len(labels)

    if sample_count == 0:
        raise ValueError(
            f"No frames found in: {path}"
        )

    if block_size <= 1:
        raise ValueError(
            "--block_size must be greater than 1"
        )

    if block_size > sample_count:
        raise ValueError(
            "--block_size cannot exceed "
            "the number of frames"
        )

    point_metrics = full_sequence_metrics(
        labels=labels,
        probabilities=probabilities,
        face_detected=face_detected,
    )

    bootstrap_values = {
        "LFR": np.empty(
            iterations,
            dtype=np.float64,
        ),
        "MSL": np.empty(
            iterations,
            dtype=np.float64,
        ),
        "PJ_L1": np.empty(
            iterations,
            dtype=np.float64,
        ),
        "FDDR": np.empty(
            iterations,
            dtype=np.float64,
        ),
    }

    rng = np.random.default_rng(
        seed
    )

    for iteration in range(
        iterations
    ):
        block_starts = (
            sample_block_starts(
                sample_count=sample_count,
                block_size=block_size,
                rng=rng,
            )
        )

        metrics = (
            calculate_block_bootstrap_metrics(
                labels=labels,
                probabilities=probabilities,
                face_detected=face_detected,
                block_starts=block_starts,
                block_size=block_size,
                target_length=sample_count,
            )
        )

        for metric_name in (
            bootstrap_values
        ):
            bootstrap_values[
                metric_name
            ][iteration] = metrics[
                metric_name
            ]

        if (
            iteration == 0
            or (iteration + 1) % 500 == 0
            or iteration + 1 == iterations
        ):
            print(
                f"{path.name}: "
                f"{iteration + 1}/"
                f"{iterations}"
            )

    result = {
        "log_file": str(path),
        "num_frames": int(
            sample_count
        ),
        "bootstrap_iterations": int(
            iterations
        ),
        "confidence_level": float(
            confidence_level
        ),
        "block_size_frames": int(
            block_size
        ),
        "seed": int(seed),
        "boundary_handling": (
            "Temporal differences are calculated "
            "within sampled blocks only. Transitions "
            "between independently sampled blocks "
            "are excluded."
        ),
        "metrics": {},
    }

    for metric_name, values in (
        bootstrap_values.items()
    ):
        lower, upper = (
            percentile_interval(
                values,
                confidence_level,
            )
        )

        result["metrics"][
            metric_name
        ] = {
            "point_estimate": float(
                point_metrics[
                    metric_name
                ]
            ),
            "bootstrap_mean": float(
                np.mean(values)
            ),
            "ci_lower": lower,
            "ci_upper": upper,
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Boundary-aware moving-block bootstrap "
            "confidence intervals for real-time "
            "FER temporal metrics."
        )
    )

    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--block_size",
        type=int,
        default=30,
        help=(
            "Length of each contiguous block in "
            "frames. At 30 FPS, 30 frames "
            "corresponds to one second."
        ),
    )

    parser.add_argument(
        "--confidence_level",
        type=float,
        default=0.95,
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
            "realtime_hybrid_block_bootstrap.json"
        ),
    )

    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError(
            "--iterations must be greater than 0"
        )

    if not (
        0.0
        < args.confidence_level
        < 1.0
    ):
        raise ValueError(
            "--confidence_level must be "
            "between 0 and 1"
        )

    results = []

    for index, log_name in enumerate(
        args.logs
    ):
        path = Path(
            log_name
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Log not found: {path}"
            )

        result = bootstrap_log(
            path=path,
            iterations=args.iterations,
            block_size=args.block_size,
            confidence_level=(
                args.confidence_level
            ),
            seed=args.seed + index,
        )

        results.append(
            result
        )

    payload = {
        "method": (
            "boundary-aware moving-block bootstrap"
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

    print()
    print(
        json.dumps(
            payload,
            indent=2,
        )
    )

    print()
    print(
        f"Saved results to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()