import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    bins = [
        b for b in data["bins"]
        if b["count"] > 0
    ]

    confidences = [b["mean_confidence"] for b in bins]
    accuracies = [b["accuracy"] for b in bins]

    plt.figure(figsize=(6, 6))

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        linewidth=1.5,
        label="Perfect calibration",
    )

    plt.plot(
        confidences,
        accuracies,
        "o-",
        linewidth=2,
        label="Model",
    )

    plt.xlabel("Mean confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(True)
    plt.legend()

    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)

    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()