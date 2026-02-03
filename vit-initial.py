import sys
import torch # Needed to check compilation support
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Optimizing for Data Loading Speed
# Use standard library multiprocessing to detect core count for workers
import multiprocessing
num_workers = min(4, multiprocessing.cpu_count()) # Cap at 4 to prevent RAM OOM on 8GB machine

dataset = load_dataset(
    "imagefolder",
    data_dir="clean_dataset",
)

processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)

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
        "f1": f1,
    }

def transform(batch):
    pixel_values = processor(
        batch["image"],
        return_tensors="pt"
    )["pixel_values"]

    batch["pixel_values"] = pixel_values
    del batch["image"]
    return batch

dataset = dataset.with_transform(transform)

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1},
    ignore_mismatched_sizes=True,
)

training_args = TrainingArguments(
    output_dir="./vit-roadwork-output",
    
    # --- HARDWARE OPTIMIZATIONS ---
    # 1. FP16 (Mixed Precision): Critical for GTX 1660 Super
    # Reduces VRAM usage by ~50%, allowing larger batches and faster memory access.
    fp16=True, 

    # 2. Batch Size & Accumulation
    # 32 might fit in 6GB VRAM with FP16. If OOM, revert to 16.
    per_device_train_batch_size=16, 
    # Simulate a larger batch (e.g., 64) without extra VRAM
    gradient_accumulation_steps=2, 
    
    # 3. Data Loading Workers
    # Default is 0 (main process), which kills speed. Set to 4 to pre-load data.
    dataloader_num_workers=num_workers,
    dataloader_pin_memory=True, # Faster CPU->GPU transfer

    # 4. PyTorch 2.0 Compilation
    # Free speedup for ViT models (graph optimization)
    torch_compile=False, 
    
    # --- STANDARD ARGS ---
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    num_train_epochs=5,
    learning_rate=3e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=True,
    report_to='tensorboard',
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    # tokenizer=image_processor,   # or feature_extractor
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.evaluate()