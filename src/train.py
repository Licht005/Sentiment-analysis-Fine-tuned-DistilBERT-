import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate

#steps
# Load dataset
# Load tokenizer and model
# Tokenization 
# metrics
# training

# 1. Load dataset (IMDB reviews: pos/neg)
dataset = load_dataset("imdb")

# 2. Load tokenizer & model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. Tokenization function
def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(4000))  # small subset,since im using CPU
test_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))

# 4. Define metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./outputs",
    #evaluation_strategy="epoch",
    # fallback in case of old API
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,   # 1–2 epochs for CPU, increase on GPU
    weight_decay=0.01,
    report_to="none",     # disable wandb/logging
)

# 6. Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 7. Train & evaluate
trainer.train()
metrics = trainer.evaluate()
print(metrics)

# 8. Save model
trainer.save_model("./outputs/fine_tuned_distilbert")
tokenizer.save_pretrained("./outputs/fine_tuned_distilbert")
