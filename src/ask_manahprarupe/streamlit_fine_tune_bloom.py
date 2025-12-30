# import streamlit as st
# import tempfile
# import os
# from datasets import load_dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     Trainer,
#     TrainingArguments,
#     DataCollatorForSeq2Seq
# )

# st.title("Fine-tune BLOOMZ on Marathi Dataset")

# # ====================== UPLOAD JSONL DATASET ======================
# uploaded_file = st.file_uploader(
#     "Upload your Marathi JSONL dataset (with 'prompt' and 'completion')",
#     type=["jsonl"]
# )

# if uploaded_file:
#     # Save uploaded file temporarily
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         dataset_path = tmp_file.name

#     st.success(f"Dataset uploaded: {os.path.basename(dataset_path)}")

#     # ====================== CONFIG ======================
#     MODEL_NAME = "bigscience/bloomz-560m"
#     OUTPUT_DIR = "./fine_tuned_bloom"
#     MAX_LENGTH = 512
#     BATCH_SIZE = 2
#     EPOCHS = 3
#     LEARNING_RATE = 5e-5

#     # ====================== LOAD DATASET ======================
#     st.write("Loading dataset...")
#     dataset = load_dataset("json", data_files="marathi_finetune.jsonl", split="train")
#     st.write(f"Dataset loaded with {len(dataset)} examples.")

#     # ====================== TOKENIZER ======================
#     st.write("Loading tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     # ====================== TOKENIZATION ======================
#     st.write("Tokenizing dataset...")

#     def tokenize_function(examples):
#         inputs = examples["prompt"]
#         targets = examples["completion"]
        
#         model_inputs = tokenizer(
#             inputs,
#             max_length=MAX_LENGTH,
#             truncation=True,
#             padding="max_length"
#         )
        
#         labels = tokenizer(
#             targets,
#             max_length=MAX_LENGTH,
#             truncation=True,
#             padding="max_length"
#         )
        
#         model_inputs["labels"] = labels["input_ids"]
#         return model_inputs


#     tokenized_dataset = dataset.map(tokenize_function, batched=True)

#     # ====================== DATA COLLATOR ======================
#     data_collator = DataCollatorForSeq2Seq(
#         tokenizer=tokenizer,
#         model=None,
#         padding=True
#     )

#     # ====================== LOAD MODEL ======================
#     st.write("Loading model...")
#     model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

#     # ====================== TRAINING ======================
#     training_args = TrainingArguments(
#         output_dir=OUTPUT_DIR,
#         overwrite_output_dir=True,
#         evaluation_strategy="no",
#         learning_rate=LEARNING_RATE,
#         per_device_train_batch_size=BATCH_SIZE,
#         num_train_epochs=EPOCHS,
#         weight_decay=0.01,
#         save_strategy="epoch",
#         logging_dir="./logs",
#         logging_steps=10,
#         fp16=False,  # Change to True if GPU supports
#         warmup_steps=50
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_dataset,
#         tokenizer=tokenizer,
#         data_collator=data_collator
#     )

#     if st.button("Start Fine-tuning"):
#         with st.spinner("Fine-tuning in progress... This may take a while!"):
#             trainer.train()
#             trainer.save_model(OUTPUT_DIR)
#             tokenizer.save_pretrained(OUTPUT_DIR)
#         st.success(f"Fine-tuning completed! Model saved to {OUTPUT_DIR}")

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

st.title("Marathi Q&A with Fine-tuned BLOOM")

# ====================== LOAD MODEL ======================
MODEL_PATH = "./fine_tuned_bloom"  # your fine-tuned BLOOM model path

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    qa_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )
    return qa_pipeline

qa_pipeline = load_model()

# ====================== USER INPUT ======================
question = st.text_area("Type your Marathi question here:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question!")
    else:
        with st.spinner("Generating answer..."):
            prompt = f"प्रश्न: {question.strip()}\nउत्तर:"
            output = qa_pipeline(
                prompt,
                max_new_tokens=300,
                temperature=0.7,
                repetition_penalty=1.1
            )
            answer = output[0]["generated_text"].replace(prompt, "").strip()
        st.success("Answer:")
        st.write(answer)
