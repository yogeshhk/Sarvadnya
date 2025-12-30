import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# ====================== CONFIG ======================
MODEL_NAME = "bigscience/bloomz-560m"   # BLOOMZ smaller variant (you can switch to larger if GPU allows)
DATA_FILE = "marathi_finetune.jsonl"              # JSONL with {"prompt": "...", "completion": "..."} for Q&A
OUTPUT_DIR = "./fine_tuned_bloom"
MAX_LENGTH = 512
BATCH_SIZE = 2
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
    # If batched=True, 'examples' is a dict of lists, not a list of dicts
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
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=None,
    padding=True
)

# ====================== LOAD MODEL ======================
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

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
    fp16=True,              # Use FP16 if your GPU supports
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

