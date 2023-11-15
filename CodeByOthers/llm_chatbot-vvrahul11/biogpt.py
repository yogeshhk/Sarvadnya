"""Module to interact with biogpt, accept user input and generate outputs"""

from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM
import streamlit as st


set_seed(42)
XFORMERS_MORE_DETAILS=1

model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
generator = pipeline("text-generation",model=model,tokenizer=tokenizer)


def generate_response(prompt):
    message = generator(
        prompt,
        max_length=200,
        num_return_sequences=1,
        do_sample=False
    )
    return message[0]['generated_text']


def get_text():
    input_text = st.text_input("You: ","", key="input")
    return input_text 