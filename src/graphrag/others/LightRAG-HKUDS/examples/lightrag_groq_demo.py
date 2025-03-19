# pip install -q -U google-genai to use gemini as a client

import os
import numpy as np
# from google import genai
from groq import Groq
# from google.genai import types
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG, QueryParam
from sentence_transformers import SentenceTransformer
from lightrag.kg.shared_storage import initialize_pipeline_status

import asyncio
# import nest_asyncio

# Apply nest_asyncio to solve event loop issues
# nest_asyncio.apply()

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

WORKING_DIR = "./dickens"

if os.path.exists(WORKING_DIR):
    import shutil

    shutil.rmtree(WORKING_DIR)

os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    # 1. Initialize the GenAI Client with your Gemini API Key
    client = Groq(api_key=groq_api_key)

    # 2. Combine prompts: system prompt, history, and user prompt
    if history_messages is None:
        history_messages = []

    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"

    for msg in history_messages:
        # Each msg is expected to be a dict: {"role": "...", "content": "..."}
        combined_prompt += f"{msg['role']}: {msg['content']}\n"

    # Finally, add the new user prompt
    combined_prompt += f"user: {prompt}"

    # 3. Call the Groq model
    # response = client.models.generate_content(
    #     model="llama3-8b-8192",
    #     contents=[combined_prompt],
    #     # config=types.GenerateContentConfig(max_output_tokens=500, temperature=0.1),
    # )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": combined_prompt,
            }
        ],
        model="llama3-8b-8192",
    )
    response = chat_completion.choices[0].message.content
    # print(response)

    # 4. Return the response text
    return response


async def embedding_func(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())
    file_path = "./data/book.txt" # "story.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    rag.insert(text)

    response = rag.query(
        query="What is the main theme of the book?",
        param=QueryParam(mode="hybrid", top_k=5, response_type="single line"),
    )

    print(response)


if __name__ == "__main__":
    main()
