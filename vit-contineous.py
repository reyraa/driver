import sys
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)
from torchvision.transforms import (
    Compose, 
    RandomHorizontalFlip, 
    RandomRotation, 
    ColorJitter, 
    ToTensor, 
    Normalize, 
    Resize
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import multiprocessing


num_workers = min(4, multiprocessing.cpu_count()) # Cap at 4 to prevent RAM OOM on 8GB machine

PREVIOUS_MODEL_PATH = "./vit-roadwork-output" 
NEW_OUTPUT_DIR = "./vit-roadwork-output-refined"

dataset = load_dataset(
    "imagefolder",
    data_dir="clean_dataset",
)

processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)
normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
size = (
    processor.size["shortest_edge"]
    if "shortest_edge" in processor.size
    else (processor.size["height"], processor.size["width"])
)
_transforms = Compose([
    Resize(size),
    RandomHorizontalFlip(p=0.5),        # 50% chance to flip
    RandomRotation(degrees=15),         # Rotate slightly
    ColorJitter(brightness=0.1, contrast=0.1), # Slight lighting changes
    ToTensor(),
    normalize,
])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def transform(batch):
    batch["pixel_values"] = [_transforms(x.convert("RGB")) for x in batch["image"]]
    del batch["image"]
    return batch

dataset = dataset.with_transform(transform)
print(f"Loading best weights from {PREVIOUS_MODEL_PATH}...")
model = ViTForImageClassification.from_pretrained(
    PREVIOUS_MODEL_PATH,
    num_labels=2,
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1},
    ignore_mismatched_sizes=True,
)

training_args = TrainingArguments(
    output_dir=NEW_OUTPUT_DIR,

    # --- HARDWARE OPTIMIZATIONS ---
    fp16=True,
    per_device_train_batch_size=16, 
    gradient_accumulation_steps=2, 
    # Data Loading Workers
    dataloader_num_workers=num_workers,
    dataloader_pin_memory=True, # Faster CPU->GPU transfer
    # PyTorch 2.0 Compilation
    torch_compile=False, 

    # --- STANDARD ARGS ---
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    num_train_epochs=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=True,
    report_to='tensorboard',
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        print("Starting refinement training...")
        trainer.train()
        trainer.save_model(NEW_OUTPUT_DIR)
        print(f"Refinement complete. Model saved to {NEW_OUTPUT_DIR}")
    else:
        print("Starting evaluation process...")
        trainer.evaluate()
