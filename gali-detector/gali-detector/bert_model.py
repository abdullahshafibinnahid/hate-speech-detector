"""
bert_model.py - Bangla BERT Model
Best model for Bangla + English mixed hate speech
Model: sagorsarker/bangla-bert-base
Train on Kaggle with GPU (T4 x2)
"""

# In Kaggle, first run:
# !pip install transformers datasets torch -q

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

MODEL_NAME = "sagorsarker/bangla-bert-base"
CSV_PATH = "full_mixed_dataset.csv"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

print(f"Loading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=['text'])
print(f"Total: {len(df)} | Distribution: {df['label'].value_counts().to_dict()}")

train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding=True, max_length=128)

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)
train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)
train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

args = TrainingArguments(
    output_dir="./bangla-bert-gali",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("./bangla-bert-gali-final")
tokenizer.save_pretrained("./bangla-bert-gali-final")
print("Model saved to ./bangla-bert-gali-final")
