from pathlib import Path
import csv
import cv2
import torch
from torch.utils.data import Dataset


class AffectNetSubset(Dataset):
    """
    CSV format:
        path,label
        train/happy/img_001.jpg,3

    - path is relative to root_dir
    - label is int in [0, num_classes-1]
    """

    def __init__(self, root_dir, csv_path, transform=None, num_classes=8):
        self.root = Path(root_dir)
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.num_classes = num_classes
        self.samples = []

        missing = 0

        with self.csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise ValueError(
                    f"CSV must contain columns ['path', 'label'], got {reader.fieldnames}"
                )

            for row in reader:
                rel_path = row["path"].strip()
                label = int(row["label"])

                if label < 0 or label >= num_classes:
                    continue

                img_path = self.root / rel_path
                if not img_path.exists():
                    missing += 1
                    continue

                self.samples.append((rel_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {self.csv_path}")

        print(
            f"[AffectNetSubset] {self.csv_path.name}: "
            f"loaded={len(self.samples)} skipped_missing={missing}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        img_path = self.root / rel_path

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
