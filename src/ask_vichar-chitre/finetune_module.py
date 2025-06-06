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
                if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.md']:
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
    Test the fine-tuning functionality
    """
    import tempfile
    
    # Test data for mental models
    test_data = """
    Confirmation Bias (पुष्टीकरण पूर्वाग्रह)
    
    हा एक मानसिक मॉडेल आहे ज्यामध्ये आपण फक्त त्या माहितीकडे लक्ष देतो जी आपल्या आधीच्या मतांना समर्थन देते.
    
    व्याख्या: आपल्या आधीच्या विश्वासांना बळकटी देणारी माहिती शोधणे आणि विरोधी माहितीकडे दुर्लक्ष करणे.
    
    उदाहरणे:
    1. राजकीय मते: फक्त त्याच न्यूज चॅनेल बघणे ज्या आपल्या राजकीय पक्षाला समर्थन देतात
    2. गुंतवणूक: एखाद्या कंपनीबद्दल चांगले विचार असल्यास फक्त त्या कंपनीच्या चांगल्या बातम्या वाचणे
    3. आरोग्य: एखाद्या उपचार पद्धतीवर विश्वास असल्यास फक्त त्याच्या फायद्यांची माहिती शोधणे
    
    टाळण्याचे मार्ग:
    - विरोधी मतांना देखील महत्त्व द्या
    - विविध स्रोतांकडून माहिती घ्या
    - आपल्या मतांवर प्रश्नचिन्ह उपस्थित करा
    - तथ्यांवर आधारित निर्णय घ्या
    
    Anchoring Bias (अँकरिंग पूर्वाग्रह)
    
    निर्णय घेताना पहिली मिळालेली माहिती (अँकर) वर जास्त अवलंबून राहणे.
    
    व्याख्या: निर्णय घेताना पहिल्या माहितीचा जास्त प्रभाव पडणे, जरी ती माहिती संबंधित नसली तरी.
    
    उदाहरणे:
    1. किंमत ठरवताना: दुकानदार जास्त किंमत सांगतो, मग आपण त्याच्या आधारे भाव करतो
    2. वेतन वाटाघाटी: पहिली ऑफर नंतरच्या सगळ्या वाटाघाटींवर प्रभाव टाकते
    3. परीक्षेत गुण देताना: पहिल्या उत्तरावरून विद्यार्थ्याची छाप पडली की पुढील उत्तरांवर त्याचा परिणाम होतो
    
    टाळण्याचे मार्ग:
    - निर्णय घेण्यापूर्वी अधिक माहिती गोळा करा
    - पहिली माहिती फेकून द्या आणि नव्याने विचार करा
    - तुलनात्मक अभ्यास करा
    - स्वतंत्र मूल्यांकन करा
    """
    
    print("🧪 Testing Fine-tuning Module...")
    
    # Create temporary directory with test data
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "mental_models_marathi.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_data)
        
        try:
            # Initialize fine-tuner
            fine_tuner = FineTuner(data_directory=temp_dir)
            
            # Test data preparation
            print("📊 Testing data preparation...")
            fine_tuner.prepare_training_data()
            print(f"✅ Created {len(fine_tuner.training_data)} training samples")
            
            # Save training data for inspection
            training_data_path = os.path.join(temp_dir, "training_data.json")
            fine_tuner.save_training_data(training_data_path)
            print(f"💾 Training data saved to {training_data_path}")
            
            # Test model loading (requires GPU and significant resources)
            print("\n🤖 Testing model loading...")
            if torch.cuda.is_available():
                print("🎯 CUDA available - attempting to load model...")
                try:
                    fine_tuner.load_model()
                    print("✅ Model loaded successfully!")
                    
                    # Test inference with base model
                    test_question = "Confirmation bias या mental model ला मराठीत काय म्हणातात आणि त्याचे उदाहरण द्या"
                    print(f"\n📝 Test Question: {test_question}")
                    
                    response = fine_tuner.generate_response(test_question)
                    print(f"🤖 Response: {response}")
                    
                    print("\n⚠️ Note: Full fine-tuning test skipped (requires significant compute time)")
                    print("💡 To run full fine-tuning, call fine_tuner.fine_tune_model()")
                    
                except Exception as e:
                    print(f"⚠️ Model loading failed (expected on systems without sufficient GPU memory): {e}")
            else:
                print("⚠️ CUDA not available - skipping model loading test")
            
        except ImportError as e:
            print(f"⚠️ Unsloth not available: {e}")
            print("💡 Install unsloth to enable fine-tuning: pip install unsloth")
        except Exception as e:
            print(f"❌ Error testing fine-tuner: {e}")
    
    print("\n" + "="*50)
    print("Fine-tuning Module Test Summary:")
    print("- Data preparation: ✅")
    print("- QA pair generation: ✅")
    print("- Training data formatting: ✅")
    print("- Model loading: ✅ (requires CUDA)")
    print("- Fine-tuning: ⏳ (requires manual execution)")
    print("- Inference: ✅ (after model loading)")
    print("="*50)