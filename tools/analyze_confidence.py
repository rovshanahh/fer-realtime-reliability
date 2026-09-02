import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def read_log(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze confidence and selective prediction metrics."
    )

    parser.add_argument(
        "--log",
        required=True,
        help="CSV log containing confidence_smooth.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Confidence threshold used for abstention.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path.",
    )

    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(
            "--threshold must be between 0 and 1"
        )

    log_path = Path(args.log)

    if not log_path.exists():
        raise FileNotFoundError(
            f"Log not found: {log_path}"
        )

    rows = read_log(log_path)

    valid_rows = [
        row
        for row in rows
        if int(row["face_detected"]) == 1
        and int(row["y_smooth"]) != -1
    ]

    if not valid_rows:
        raise RuntimeError(
            "No valid prediction rows were found."
        )

    confidences = [
        float(row["confidence_smooth"])
        for row in valid_rows
    ]

    accepted = [
        confidence >= args.threshold
        for confidence in confidences
    ]

    total = len(confidences)
    accepted_count = sum(accepted)
    abstained_count = total - accepted_count

    mean_confidence = sum(confidences) / total
    coverage = accepted_count / total
    abstention_rate = abstained_count / total
    low_confidence_rate = abstention_rate

    result = {
        "threshold": args.threshold,
        "valid_frames": total,
        "mean_confidence": mean_confidence,
        "coverage": coverage,
        "abstention_rate": abstention_rate,
        "low_confidence_rate": low_confidence_rate,
    }

    print(json.dumps(result, indent=2))

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()