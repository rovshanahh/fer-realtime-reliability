import os
import csv

FER2013_ROOT = "/Users/rovshanahaji-hasanli/Downloads/archive"

LABEL_MAP = {
    "neutral":  0,
    "happy":    1,
    "sad":      2,
    "surprise": 3,
    "fear":     4,
    "disgust":  5,
    "angry":    6,
}

split = "test"
rows = []

for emotion_folder, label_int in LABEL_MAP.items():
    folder_path = os.path.join(FER2013_ROOT, split, emotion_folder)
    for fname in os.listdir(folder_path):
        if fname.endswith(".jpg") or fname.endswith(".png"):
            rel_path = os.path.join(split, emotion_folder, fname)
            rows.append({"path": rel_path, "label": label_int})

with open("fer2013_test.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Done. {len(rows)} images written to fer2013_test.csv")