"""Module to create falcon pipeline, accept user input, and generate outputs"""

import streamlit as st
from transformers import AutoTokenizer
import transformers
import torch


model =  "tiiuae/falcon-40b-instruct" #"tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model)


def create_pipeline():
    """Function to create a falcon pipeline"""
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    return pipeline


def generate_response(prompt):
    pipeline = create_pipeline()
    sequences = pipeline(
        prompt,
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    message = []
    for seq in sequences:
        message.append(seq['generated_text'])
    return message


def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text 

