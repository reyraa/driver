import sys
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)

dataset = load_dataset(
    "imagefolder",
    data_dir="clean_dataset",
)

processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)

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
    per_device_train_batch_size=16,
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
)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        trainer.train()
    else:
        trainer.evaluate()