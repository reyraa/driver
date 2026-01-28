import os
import json
from pathlib import Path
import random
import shutil
from pathlib import Path

from tqdm import tqdm
from datasets import load_dataset

# ---- CONFIG ----
PARQUET_DIR = Path("/Users/reyraa/Projects/natix/hydra/roadwork-dataset/data")
OUTPUT_DIR = Path("./clean_dataset")
MAX_TOTAL = None  # Set to an integer for quick tests (e.g., 1000)

# ---- OUTPUT STRUCTURE ----
POS_DIR = OUTPUT_DIR / "positive_cases"
NEG_DIR = OUTPUT_DIR / "negative_cases"
UNSURE_DIR = OUTPUT_DIR / "unsure_cases"
CLASSES = ["positive_cases", "negative_cases"]
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

for d in [POS_DIR, NEG_DIR, UNSURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---- FUNCTION: Process with Hugging Face dataset loader ----
def extract_and_sort(dataset_path: str):
    ds = load_dataset("parquet", data_files=dataset_path, split="train")
    print(f"Loaded {len(ds)} rows from {dataset_path}")

    for idx, row in tqdm(enumerate(ds), total=len(ds), desc=f"Processing {Path(dataset_path).name}"):
        try:
            image = row["image"]  # This is a PIL.Image
            metadata = {k: v for k, v in row.items() if k != "image"}
            label = metadata.get("label")
            has_description = bool(metadata.get("scene_description"))
        except Exception as e:
            print(f"Skipping row {idx}: {e}")
            continue

        out_name = f"{Path(dataset_path).stem}__image_{idx}"

        if label == 1 and has_description:
            target_dir = POS_DIR
        elif label == 0 and not has_description:
            target_dir = NEG_DIR
        else:
            target_dir = UNSURE_DIR

        try:
            image.convert("RGB").save(target_dir / f"{out_name}.jpeg")
            with open(target_dir / f"{out_name}.json", "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Failed to save {out_name}: {e}")

        if MAX_TOTAL and idx >= MAX_TOTAL:
            break

def prepare_dirs():
    for split in ["train", "test"]:
        for cls in CLASSES:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)



def split_class(cls: str):
    src_dir = OUTPUT_DIR / cls

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
        dest_dir = OUTPUT_DIR / split_name / cls
        for jpeg, json_file in split_items:
            shutil.move(jpeg, dest_dir / jpeg.name)
            shutil.move(json_file, dest_dir / json_file.name)


# ---- MAIN ----
all_parquets = sorted(PARQUET_DIR.glob("*.parquet"))

for parquet_file in all_parquets:
    extract_and_sort(str(parquet_file))
    prepare_dirs()
    for cls in CLASSES:
        split_class(cls)

print("\n✅ Done: Clean dataset written to:", OUTPUT_DIR.resolve())
print(f" - {len(list(POS_DIR.glob('*.jpeg')))} positive cases")
print(f" - {len(list(NEG_DIR.glob('*.jpeg')))} negative cases")
print(f" - {len(list(UNSURE_DIR.glob('*.jpeg')))} unsure cases")