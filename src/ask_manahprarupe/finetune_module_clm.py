# CLM (Causal Language Model) is for text generation and PEFT like methods can be used to do domain adapatation with just documents
# and not QnA pairs.
#
# Currently getting following error:
#   trainer = Trainer(
# WARNING:accelerate.big_modeling:You shouldn't move a model that is dispatched using accelerate hooks.
# ❌ Error: You can't move a model that has some modules offloaded to cpu or disk.

import os
import logging
from pathlib import Path
from typing import List, Dict

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FineTuner:
    def __init__(self,
                 data_directory: str = "data",
                 model_name: str = "google/gemma-7b",
                 max_seq_length: int = 512,
                 output_dir: str = "./models"):
        self.data_directory = data_directory
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.output_dir = output_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        logger.info(f"Initialized EssayFineTuner with model: {model_name}")

    def load_texts(self) -> Dataset:
        texts = []
        for file_path in Path(self.data_directory).rglob("*"):
            if file_path.suffix.lower() in [".txt", ".md", ".tex"]:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            texts.append({"text": content})
                except Exception as e:
                    logger.warning(f"Error reading file {file_path}: {e}")
        logger.info(f"Loaded {len(texts)} text files")
        return Dataset.from_list(texts)

    def tokenize_function(self, examples: Dict[str, str]) -> Dict[str, List[int]]:
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length
        )

    def fine_tune(self,
                  num_train_epochs: int = 3,
                  per_device_train_batch_size: int = 2,
                  learning_rate: float = 5e-5):
        logger.info("Starting fine-tuning process...")

        dataset = self.load_texts()
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # for causal language modeling
        )

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to=None  # Turn off W&B/logging
        )

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )

        trainer.train()

        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        logger.info(f"Fine-tuning complete. Model saved to {self.output_dir}")

    def generate_response(self, prompt: str, max_new_tokens: int = 256) -> str:
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


if __name__ == "__main__":
    print("🧪 Testing essay-based fine-tuning module")

    try:
        fine_tuner = FineTuner(data_directory="data")
        fine_tuner.fine_tune()

        prompt = "मनोविज्ञानात confirmation bias म्हणजे"
        response = fine_tuner.generate_response(prompt)
        print(f"🤖 Response: {response}")

    except Exception as e:
        print(f"❌ Error: {e}")
