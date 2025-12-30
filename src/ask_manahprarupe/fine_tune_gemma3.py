import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv
from huggingface_hub import login

# ==============================
# 0. Hugging Face authentication
# ==============================
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token is None:
    raise ValueError("HUGGINGFACE_TOKEN not found in .env")
login(token=hf_token)

# ==============================
# 1. Load base model + tokenizer
# ==============================
MODEL_NAME = "google/gemma-3-270m"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)

# Gemma models often don’t have a pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=hf_token,
    torch_dtype=torch.float32,   # CPU-friendly
    device_map=None,            # Force CPU
    attn_implementation="eager"
)

# ====================================
# 2. Load your Marathi fine-tune data
# ====================================
dataset = load_dataset("json", data_files={"train": "alpaca_marathi.jsonl"})

def preprocess(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output_text = example["output"]

    if input_text.strip() != "":
        # when input is provided
        text = f"प्रश्न: {instruction}\nमाहिती: {input_text}\nउत्तर: {output_text}"
    else:
        # no extra input, only instruction
        text = f"प्रश्न: {instruction}\nउत्तर: {output_text}"

    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()  # 👈 Needed for causal LM loss
    return tokenized

tokenized_dataset = dataset.map(preprocess, batched=False)

# ===============================
# 3. Add LoRA adapters (efficient)
# ===============================
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# ===============================
# 4. Training arguments
# ===============================
training_args = TrainingArguments(
    output_dir="./gemma3-marathi-lora",
    per_device_train_batch_size=1,   # small for CPU
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=False,
    bf16=False,
    report_to="none"
)

# ===============================
# 5. Trainer
# ===============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer
)

trainer.train()

# ===============================
# 6. Save fine-tuned model
# ===============================
trainer.save_model("./gemma3-marathi-lora")
tokenizer.save_pretrained("./gemma3-marathi-lora")
