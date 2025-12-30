import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# ====================== CONFIG ======================
MODEL_NAME = "google/muril-base-cased"   # MuRIL multilingual model
DATA_FILE = "dataset.jsonl"              # Should contain {"text": "..."} in Marathi
OUTPUT_DIR = "./fine_tuned_muril"
BLOCK_SIZE = 256                         # Shorter works better for Q&A
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 5e-5

# ====================== LOAD DATASET ======================
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

# ====================== TOKENIZER ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ====================== TOKENIZATION ======================
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=BLOCK_SIZE,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# ====================== DATA COLLATOR ======================
# MuRIL is a masked language model (BERT-like), so MLM=True
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# ====================== LOAD MODEL ======================
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

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
    logging_steps=50,
    fp16=False,
    warmup_steps=50
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
