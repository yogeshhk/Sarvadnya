import streamlit as st
import requests
import os

# Make sure this is set: export GROQ_API_KEY=gsk_your_key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "gsk_your_key_here")
if not GROQ_API_KEY.startswith("gsk_"):
    raise ValueError("Invalid Groq API key. It must start with 'gsk_'.")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-8b-8192"  # Stable and supported on Groq

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def generate_response(prompt):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful general assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }

    response = requests.post(GROQ_API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

def get_text():
    return st.text_input("You:", "", key="input")
