import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cm", required=True, help="Path to confusion matrix .npy file")
    parser.add_argument("--tag", required=True, help="Tag for output file name (e.g., frozen, layer4, layer3)")
    parser.add_argument("--out_dir", default="results/figures", help="Output directory")
    args = parser.parse_args()

    class_names = [
        "neutral", "happy", "sad", "surprise",
        "fear", "disgust", "anger", "contempt"
    ]

    os.makedirs(args.out_dir, exist_ok=True)

    # Load confusion matrix
    cm = np.load(args.cm)

    # Normalize row-wise
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Plot
    plt.figure(figsize=(6.5, 6))
    im = plt.imshow(cm_norm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    title_map = {
        "frozen": "Confusion Matrix (Frozen ResNet-50)",
        "layer4": "Confusion Matrix (Layer4+FC Fine-tuned ResNet-50)",
        "layer3": "Confusion Matrix (Layer3+Layer4+FC Fine-tuned ResNet-50)"
    }

    plt.title(title_map.get(args.tag, "Confusion Matrix"))

    plt.tight_layout()

    out_path = os.path.join(args.out_dir, f"cm_{args.tag}_resnet50.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {out_path}")

if __name__ == "__main__":
    main()