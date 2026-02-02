import os
import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datasets import load_dataset

HF_DATASET_REPO = "natix-network-org/roadwork"

OUTPUT_DIR = Path("./clean_dataset2")
MAX_SAMPLES = None  # Set to e.g., 1000 for testing, None for full run
NUM_THREADS = 8     # Adjust based on your CPU cores

# ---- OUTPUT STRUCTURE ----
POS_DIR = OUTPUT_DIR / "positive_cases"
NEG_DIR = OUTPUT_DIR / "negative_cases"
UNSURE_DIR = OUTPUT_DIR / "unsure_cases"

# Ensure clean directories
for d in [POS_DIR, NEG_DIR, UNSURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def process_single_row(args):
    """
    Helper function to process and save a single row.
    Designed to be run in parallel.
    """
    idx, row = args

    try:
        image = row["image"] # PIL Image
        # Extract metadata (everything except the image binary)
        metadata = {k: v for k, v in row.items() if k != "image"}

        label = metadata.get("label")
        # Handle cases where scene_description might be None or empty string
        scene_desc = metadata.get("scene_description")
        has_description = bool(scene_desc and str(scene_desc).strip())

        # Logic Mapping
        if label == 1 and has_description:
            target_dir = POS_DIR
        elif label == 0 and not has_description:
            target_dir = NEG_DIR
        else:
            target_dir = UNSURE_DIR

        # Create filename
        out_name = f"image_{idx}"
        image_path = target_dir / f"{out_name}.jpeg"
        json_path = target_dir / f"{out_name}.json"

        # Save Image (Convert to RGB to handle PNG/RGBA issues)
        image.convert("RGB").save(image_path, "JPEG", quality=90)

        # Save Metadata
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return True

    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        return False


def main():
    print(f"⬇️  Downloading/Loading dataset from '{HF_DATASET_REPO}'...")

    # Load dataset from Hub (Streaming mode is faster if you don't need the whole thing in RAM)
    # If the dataset is huge, use streaming=True. If it fits in RAM, remove streaming=True.
    ds = load_dataset(HF_DATASET_REPO, split="train", streaming=False)

    if MAX_SAMPLES:
        print(f"⚠️  Limiting to first {MAX_SAMPLES} samples for testing.")
        ds = ds.select(range(MAX_SAMPLES))

    total_rows = len(ds)
    print(f"✅ Loaded {total_rows} rows. Starting processing with {NUM_THREADS} threads...")

    # Prepare arguments for parallel processing
    # We convert dataset to a list of (index, row) tuples for the executor
    # Note: If dataset is massive, iterate directly instead of list comprehension to save RAM
    work_items = ((i, row) for i, row in enumerate(ds))

    # Use ThreadPoolExecutor for parallel IO operations
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(tqdm(
            executor.map(process_single_row, work_items),
            total=total_rows,
            unit="img"
        ))

    print("\n✅ Done! Dataset stored in:", OUTPUT_DIR.resolve())
    print(f" - {len(list(POS_DIR.glob('*.jpeg')))} positive cases")
    print(f" - {len(list(NEG_DIR.glob('*.jpeg')))} negative cases")
    print(f" - {len(list(UNSURE_DIR.glob('*.jpeg')))} unsure cases")

if __name__ == "__main__":
    main()