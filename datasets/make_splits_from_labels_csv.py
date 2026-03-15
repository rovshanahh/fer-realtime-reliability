import argparse
from pathlib import Path
import pandas as pd

AFFECTNET8 = ["neutral","happy","sad","surprise","fear","disgust","anger","contempt"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--affectnet_root", required=True)
    ap.add_argument("--labels_csv", default="labels.csv")
    ap.add_argument("--out_root", default="data/affectnet_subset")
    ap.add_argument("--path_col", required=True)
    ap.add_argument("--label_col", required=True)
    args = ap.parse_args()

    root = Path(args.affectnet_root)
    labels_path = root / args.labels_csv
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labels_path)

    df[args.path_col] = df[args.path_col].astype(str).str.replace("\\", "/", regex=False)

    train_dir = root / "train"
    test_dir = root / "test"

    train_rows = []
    val_rows = []

    for _, row in df.iterrows():
        rel = row[args.path_col].lstrip("/")
        label_str = str(row[args.label_col]).lower().strip()

        if label_str not in AFFECTNET8:
            continue  
        label_int = AFFECTNET8.index(label_str)

        if (train_dir / rel).exists():
            train_rows.append((f"train/{rel}", label_int))
        elif (test_dir / rel).exists():
            val_rows.append((f"test/{rel}", label_int))



    if not train_rows or not val_rows:
        raise RuntimeError("No train/val samples found. Check paths in labels.csv.")

    train_out = pd.DataFrame(train_rows, columns=["path", "label"])
    val_out = pd.DataFrame(val_rows, columns=["path", "label"])

    train_csv = out_root / "train.csv"
    val_csv = out_root / "val.csv"

    train_out.to_csv(train_csv, index=False)
    val_out.to_csv(val_csv, index=False)

    print("Wrote:", train_csv, "rows:", len(train_out))
    print("Wrote:", val_csv, "rows:", len(val_out))
    print("Class order:", AFFECTNET8)

if __name__ == "__main__":
    main()
