import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# ====================== CONFIG ======================
MODEL_NAME = "google/mt5-small"       # Seq2Seq model for multilingual text
DATA_FILE = "marathi_finetune.jsonl"  # {"prompt": "...", "completion": "..."}
OUTPUT_DIR = "./fine_tuned_mt5"
MAX_LENGTH = 512
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 5e-5

# ====================== LOAD DATASET ======================
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

# ====================== TOKENIZER ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ====================== TOKENIZATION ======================
def tokenize_function(examples):
    inputs = examples["prompt"]
    targets = examples["completion"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        targets,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# ====================== DATA COLLATOR ======================
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None, padding=True)

# ====================== LOAD MODEL ======================
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ====================== TRAINING ======================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    evaluation_strategy="no",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,           # Use FP16 if GPU supports
    warmup_steps=10,
    gradient_accumulation_steps=4  # Helps with small GPU memory
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ====================== TRAIN ======================
trainer.train()

# ====================== SAVE ======================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning completed! Model saved to:", OUTPUT_DIR)

fine_tune_viz.py
