# Fine Tune Phi-2 Model on Your Dataset https://www.youtube.com/watch?v=eLy74j0KCrY

import gc
import os
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer

hf_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
login(token=hf_api_token)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config here

dataset_path = "D:/Yogesh/GitHub/Sarvadnya/src/fine-tuning/data/AmodMentalHealthCounselingConversations_train.csv"
formatted_path = "D:/Yogesh/GitHub/Sarvadnya/src/fine-tuning/data/AmodMentalHealthCounselingConversations_formatted.csv"

base_model = "microsoft/phi-2"
fine_tuned_model = "phi2-mental-health"
# ----------------------------------------
# dataset = load_dataset("csv", data_files=dataset_path, split="train")
# df = pd.DataFrame(dataset)
#
#
# # Each LLM has different instruction tuning prompt format
# def convert_to_llama_instruct_format(row):
#     question = row['Context']
#     answer = row['Response']
#     formatted_string = f"[INST] {question} [/INST] {answer} "
#     return formatted_string
#
#
# df['text'] = df.apply(convert_to_llama_instruct_format, axis=1)
# new_df = df[['text']] # skip other columns, 'text' is default name
# new_df.to_csv(formatted_path, index=False)
#---------------------------------------------------

training_dataset = load_dataset("csv", data_files=formatted_path, split="train")
# Split into training and evaluation datasets
split_datasets  = training_dataset.train_test_split(test_size=0.1, seed=42)  # 90% train, 10% eval

# Separate datasets
train_dataset = split_datasets ["train"]
eval_dataset = split_datasets ["test"]
# Inspect datasets
print(train_dataset.column_names)
print(eval_dataset.column_names)

# ---
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnd_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnd_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True
    # device_map={"layer1": "cuda:0", "layer2": "cuda:1"}  # Correct example
)

model.config.use_cache = False
model.config.pretraining_tp = 1

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
training_arguments = TrainingArguments(
    output_dir="./models",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=32,
    eval_strategy="steps",
    eval_steps=1000,
    logging_steps=100,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    save_steps=1000,
    warmup_ratio=0.05,
    weight_decay=0.01,
    max_steps=-1
)

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=["Wqkv", "fc1", "fc2"]
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Ensure this is provided
    peft_config=peft_config,
    # dataset_text_field="Text",
    # max_sequence=690,
    tokenizer=tokenizer,
    args=training_arguments
)

trainer.train()

# Run text generation pipeline with our next model
prompt = "I am not able to sleep in night. Do you have any suggestions?"
pipe = transformers.pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=250)
result = pipe(f"[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

del model
del pipe
del trainer
gc.collect()
