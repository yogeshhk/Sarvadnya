# """Module to interact with biogpt, accept user input and generate outputs"""

# from transformers import pipeline, set_seed
# from transformers import BioGptTokenizer, BioGptForCausalLM

# set_seed(42)
# XFORMERS_MORE_DETAILS=1

# model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
# tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
# generator = pipeline("text-generation",model=model,tokenizer=tokenizer)


# def generate_response(prompt):
    # message = generator(
        # prompt,
        # max_length=200,
        # num_return_sequences=1,
        # do_sample=False
    # )
    # return message[0]['generated_text']

import streamlit as st
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)

def generate_response(prompt):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful biomedical assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

def get_text():
    return st.text_input("You:", "", key="input")
