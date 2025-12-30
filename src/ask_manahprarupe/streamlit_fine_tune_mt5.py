import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

st.title("Marathi Q&A with Fine-tuned BLOOM")

# ====================== LOAD MODEL ======================
MODEL_PATH = "./fine_tuned_bloom"  # path to your fine-tuned BLOOMZ model

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    
    qa_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        max_length=600,          # allow longer answers
        temperature=0.7,
        do_sample=True,
        top_p=0.95,              # nucleus sampling for coherent text
        top_k=50
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
            # Structured prompt to guide the model
            prompt = f"Question: {question.strip()}\nAnswer:"
            output = qa_pipeline(prompt, max_new_tokens=300)
            # Remove the prompt from the generated text
            answer = output[0]["generated_text"].replace(prompt, "").strip()
        st.success("Answer:")
        st.write(answer)
