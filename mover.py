import random
import shutil
from pathlib import Path

# Config
BASE_DIR = Path("clean_dataset")
CLASSES = ["positive_cases", "negative_cases"]
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

def prepare_dirs():
    for split in ["train", "test"]:
        for cls in CLASSES:
            (BASE_DIR / split / cls).mkdir(parents=True, exist_ok=True)

def split_class(cls: str):
    src_dir = BASE_DIR / cls

    # collect samples by stem (jpeg + json)
    samples = {}
    for jpeg in src_dir.glob("*.jpeg"):
        stem = jpeg.stem
        json_file = jpeg.with_suffix(".json")
        if json_file.exists():
            samples[stem] = (jpeg, json_file)

    items = list(samples.values())
    random.shuffle(items)

    split_idx = int(len(items) * TRAIN_RATIO)
    train_items = items[:split_idx]
    test_items = items[split_idx:]

    for split_name, split_items in [("train", train_items), ("test", test_items)]:
        dest_dir = BASE_DIR / split_name / cls
        for jpeg, json_file in split_items:
            shutil.move(jpeg, dest_dir / jpeg.name)
            shutil.move(json_file, dest_dir / json_file.name)

def main():
    prepare_dirs()
    for cls in CLASSES:
        split_class(cls)

if __name__ == "__main__":
    main()