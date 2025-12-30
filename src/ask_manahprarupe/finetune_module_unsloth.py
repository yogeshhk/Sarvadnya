#
# Unsloth could not be installed on Windows 11. I have Nvidia GPU MX570 A with Pytorch 2.7.1 on cu118 (but installed CUDA 12.1)
# Tried to get pytorch+cu version compatible with Unsloth. Thought this worked, 'triton gave problem. There is Windows triton hack
# After installation it gave that triton.ops not avilable. May be via bitsandbytes call. Its instlllation did not work. 
# So ABANDONING for now.
#
import os
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path
import tempfile

# Unsloth imports for efficient fine-tuning
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    logging.warning("Unsloth not available. Please install unsloth for fine-tuning functionality.")

# Transformers and related imports
from transformers import TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from datasets import Dataset
import torch
from peft import LoraConfig, TaskType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FineTuner:
    """
    Fine-tuning class for Gemma model using Unsloth and LoRA
    Specialized for Mental Models data in Marathi
    """
    
    def __init__(self, data_directory: str, 
                 model_name: str = "google/gemma-7b-it",
                 max_seq_length: int = 2048,
                 load_in_4bit: bool = True):
        """
        Initialize the fine-tuner
        
        Args:
            data_directory: Path to directory containing training data
            model_name: Base model name for fine-tuning
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to load model in 4-bit precision
        """
        self.data_directory = data_directory
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.training_data = []
        
        if not UNSLOTH_AVAILABLE:
            raise ImportError("Unsloth is required for fine-tuning. Please install it using: pip install unsloth")
        
        logger.info("FineTuner initialized")
    
    def load_model(self):
        """Load the base model and tokenizer using Unsloth"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load model with Unsloth for efficient training
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=None,  # Auto-detect
                load_in_4bit=self.load_in_4bit,
            )
            
            # Setup LoRA configuration
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,  # LoRA rank
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_training_data(self):
        """Prepare training data from the data directory"""
        try:
            logger.info("Preparing training data...")
            
            data_path = Path(self.data_directory)
            raw_texts = []
            
            # Load all text files
            for file_path in data_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.tex', '.md']:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content:
                                raw_texts.append(content)
                    except Exception as e:
                        logger.warning(f"Error reading file {file_path}: {e}")
            
            # Convert to question-answer format for instruction tuning
            self.training_data = self._create_qa_pairs(raw_texts)
            
            logger.info(f"Prepared {len(self.training_data)} training samples")
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
    
    def _create_qa_pairs(self, texts: List[str]) -> List[Dict]:
        """
        Create question-answer pairs from the mental models data
        
        Args:
            texts: List of text content from files
            
        Returns:
            List of formatted training samples
        """
        qa_pairs = []
        
        for text in texts:
            # Extract mental model information
            lines = text.split('\n')
            current_model = None
            current_content = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this is a mental model title
                if any(keyword in line.lower() for keyword in ['fallacy', 'bias', 'model', 'effect']):
                    if current_model and current_content:
                        # Create QA pairs for the previous model
                        qa_pairs.extend(self._generate_qa_for_model(current_model, '\n'.join(current_content)))
                    
                    current_model = line
                    current_content = []
                else:
                    current_content.append(line)
            
            # Handle the last model
            if current_model and current_content:
                qa_pairs.extend(self._generate_qa_for_model(current_model, '\n'.join(current_content)))
        
        return qa_pairs
    
    def _generate_qa_for_model(self, model_name: str, content: str) -> List[Dict]:
        """Generate multiple QA pairs for a single mental model"""
        qa_pairs = []
        
        # Question templates in Marathi
        question_templates = [
            f"{model_name} या mental model बद्दल सांगा",
            f"{model_name} म्हणजे काय?",
            f"{model_name} चे उदाहरण द्या",
            f"{model_name} कसे टाळावे?",
            f"{model_name} ला मराठीत काय म्हणतात?"
        ]
        
        # Create QA pairs using chat template format
        for question in question_templates:
            qa_pair = {
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": content}
                ]
            }
            qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def format_training_data(self) -> Dataset:
        """Format training data for Unsloth"""
        try:
            # Get chat template
            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template="gemma",
            )
            
            # Format conversations
            def formatting_prompts_func(examples):
                convos = examples["conversations"]
                texts = []
                for convo in convos:
                    text = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
                    texts.append(text)
                return {"text": texts}
            
            # Create dataset
            dataset = Dataset.from_list(self.training_data)
            dataset = dataset.map(formatting_prompts_func, batched=True)
            
            logger.info("Training data formatted successfully")
            return dataset
            
        except Exception as e:
            logger.error(f"Error formatting training data: {e}")
            raise
    
    def fine_tune_model(self, output_dir: str = "./fine_tuned_model", 
                       num_train_epochs: int = 3,
                       learning_rate: float = 2e-4,
                       per_device_train_batch_size: int = 2):
        """
        Fine-tune the model using the prepared data
        
        Args:
            output_dir: Directory to save the fine-tuned model
            num_train_epochs: Number of training epochs
            learning_rate: Learning rate for training
            per_device_train_batch_size: Batch size per device
        """
        try:
            if not self.model or not self.tokenizer:
                self.load_model()
            
            if not self.training_data:
                self.prepare_training_data()
            
            logger.info("Starting fine-tuning...")
            
            # Format training data
            train_dataset = self.format_training_data()
            
            # Training arguments
            training_args = TrainingArguments(
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                num_train_epochs=num_train_epochs,
                learning_rate=learning_rate,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=42,
                output_dir=output_dir,
                save_steps=100,
                save_total_limit=2,
                dataloader_num_workers=0,
                report_to=None,  # Disable wandb logging
            )
            
            # Create trainer using Unsloth
            from trl import SFTTrainer
            
            trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=train_dataset,
                dataset_text_field="text",
                max_seq_length=self.max_seq_length,
                dataset_num_proc=2,
                args=training_args,
            )
            
            # Start training
            trainer.train()
            
            # Save the model
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Fine-tuning completed. Model saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            raise
    
    def generate_response(self, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Generate response using the fine-tuned model
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        try:
            if not self.model or not self.tokenizer:
                raise ValueError("Model not loaded. Please load or fine-tune the model first.")
            
            # Enable fast inference
            FastLanguageModel.for_inference(self.model)
            
            # Format prompt as conversation
            messages = [{"from": "human", "value": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            response = generated_text[len(formatted_prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def save_training_data(self, filepath: str):
        """Save training data to JSON file for inspection"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Training data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving training data: {e}")

if __name__ == "__main__":
    """
    Load fine-tuned model and test inference with a few Marathi questions.
    Assumes the fine-tuned model is already available in './fine_tuned_model'.
    """
    print("🧪 Testing inference with fine-tuned model...")

    try:
        fine_tuner = FineTuner(data_directory="data")  # data_directory is not used here

        # Load fine-tuned model from disk
        print("📦 Loading fine-tuned model...")
        fine_tuner.model, fine_tuner.tokenizer = FastLanguageModel.from_pretrained(
            model_name="./fine_tuned_model",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True
        )
        print("✅ Model loaded!")

        # Test inference
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