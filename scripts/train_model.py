import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset

# Load auto-labeled training and validation sets
df_train = pd.read_csv("data/louis_ck_train_auto_labeled.csv")
df_val = pd.read_csv("data/louis_ck_val_auto_labeled.csv")

# Map string labels to integers
label_map = {"non_irony": 0, "irony": 1}
df_train["label"] = df_train["irony_sarcasm_label"].str.lower().str.strip().map(label_map).astype(int)
df_val["label"] = df_val["irony_sarcasm_label"].str.lower().str.strip().map(label_map).astype(int)

# Convert to Hugging Face Datasets
dataset_train = Dataset.from_pandas(df_train[["context_window", "label"]])
dataset_val = Dataset.from_pandas(df_val[["context_window", "label"]])

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(
        examples["context_window"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset_train = dataset_train.map(tokenize_function, batched=True)
dataset_val = dataset_val.map(tokenize_function, batched=True)

# Set format for PyTorch
dataset_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
dataset_val.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Define compute_metrics
def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(pred.label_ids, preds),
        "f1": f1_score(pred.label_ids, preds)
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/roberta_louisck",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the model and tokenizer
trainer.save_model("./models/roberta_louisck")
tokenizer.save_pretrained("./models/roberta_louisck")

print("Model training complete and saved to ./models/roberta_louisck")
