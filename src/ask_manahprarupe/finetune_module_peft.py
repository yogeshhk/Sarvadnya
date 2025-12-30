# This PEFT needs training data in QnA pairs. I just had text. Wanted to do domain adaptation in CLM (Causal Language Model)
# Thats not possible with this tutorial code. Need to work later
import os
import json
import logging
from typing import List, Dict
from pathlib import Path

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments
)
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch
from trl import SFTTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FineTuner:
    def __init__(self, data_directory: str,
                 model_name: str = "google/gemma-7b-it",
                 max_seq_length: int = 2048,
                 load_in_4bit: bool = False):
        self.data_directory = data_directory
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self.training_data = []

        logger.info("FineTuner initialized")

    def load_model(self):
        """Load base model and tokenizer, and apply LoRA"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )

            config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, config)
            logger.info("Model and tokenizer loaded with LoRA applied")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def prepare_training_data(self):
        logger.info("Preparing training data...")
        data_path = Path(self.data_directory)
        raw_texts = []

        for file_path in data_path.rglob('*'):
            if file_path.suffix.lower() in ['.txt', '.tex', '.md']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            raw_texts.append(content)
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")

        self.training_data = self._create_qa_pairs(raw_texts)
        logger.info(f"Prepared {len(self.training_data)} training samples")

    def _create_qa_pairs(self, texts: List[str]) -> List[Dict]:
        qa_pairs = []
        for text in texts:
            lines = text.split('\n')
            current_model, current_content = None, []

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if any(k in line.lower() for k in ['fallacy', 'bias', 'model', 'effect']):
                    if current_model and current_content:
                        qa_pairs.extend(self._generate_qa_for_model(current_model, '\n'.join(current_content)))
                    current_model = line
                    current_content = []
                else:
                    current_content.append(line)

            if current_model and current_content:
                qa_pairs.extend(self._generate_qa_for_model(current_model, '\n'.join(current_content)))
        return qa_pairs

    def _generate_qa_for_model(self, model_name: str, content: str) -> List[Dict]:
        questions = [
            f"{model_name} या mental model बद्दल सांगा",
            f"{model_name} म्हणजे काय?",
            f"{model_name} चे उदाहरण द्या",
            f"{model_name} कसे टाळावे?",
            f"{model_name} ला मराठीत काय म्हणतात?",
        ]
        return [{
            "conversations": [
                {"from": "human", "value": q},
                {"from": "gpt", "value": content}
            ]
        } for q in questions]

    def format_training_data(self) -> Dataset:
        def format_prompt(example):
            messages = example["conversations"]
            text = ""
            for m in messages:
                prefix = "Human: " if m["from"] == "human" else "Assistant: "
                text += prefix + m["value"].strip() + "\n"
            return {"text": text.strip()}

        dataset = Dataset.from_list(self.training_data)
        dataset = dataset.map(format_prompt)
        logger.info("Training data formatted")
        return dataset

    def fine_tune_model(self, output_dir: str = "./models/fine_tuned_model",
                        num_train_epochs: int = 3,
                        learning_rate: float = 2e-4,
                        per_device_train_batch_size: int = 2):
        if not self.model or not self.tokenizer:
            self.load_model()
        if not self.training_data:
            self.prepare_training_data()

        train_dataset = self.format_training_data()

        training_args = TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_steps=1,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=output_dir,
            save_steps=100,
            save_total_limit=2,
            report_to=None,
            fp16=torch.cuda.is_available(),
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            args=training_args,
        )

        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Fine-tuned model saved to {output_dir}")

    def generate_response(self, prompt: str, max_new_tokens: int = 256) -> str:
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded")

        self.model.eval()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()


if __name__ == "__main__":
    print("🧪 Testing inference with fine-tuned model...")

    try:
        fine_tuner = FineTuner(data_directory="data")
        print("📦 Loading fine-tuned model...")
        fine_tuner.tokenizer = AutoTokenizer.from_pretrained("./models/fine_tuned_model")
        fine_tuner.model = AutoModelForCausalLM.from_pretrained(
            "./fine_tuned_model",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        fine_tuner.model.eval()
        print("✅ Model loaded!")

        questions = [
            "What is the Marathi explanation of Sunk Cost Fallacy?",
            "Explain Confirmation Bias in Marathi with example.",
            "What does Anchoring Bias mean? Answer in Marathi."
        ]

        for question in questions:
            print(f"\n📝 Question: {question}")
            response = fine_tuner.generate_response(question)
            print(f"🤖 Response: {response}")

    except Exception as e:
        print(f"❌ Error during inference: {e}")
