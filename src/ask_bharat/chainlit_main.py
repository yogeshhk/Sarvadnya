import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import chainlit as cl
import ollama

DB_FAISS_PATH = "vectorstore/db_faiss"

custom_prompt_template = """Use the following context to answer the question. Be helpful and concise.
If unsure, just say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def ollama_call(prompt: str) -> str:
    response = ollama.chat(
        model="llama2",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

def init_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

@cl.on_chat_start
async def start():
    retriever = init_vectorstore().as_retriever(search_kwargs={"k": 3})
    cl.user_session.set("retriever", retriever)
    await cl.Message(content=" Hi! Ask me anything from your uploaded data.").send()

@cl.on_message
async def main(message: cl.Message):
    retriever = cl.user_session.get("retriever")

    docs = retriever.get_relevant_documents(message.content)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt_template = set_custom_prompt()
    final_prompt = prompt_template.format(context=context, question=message.content)

    answer = ollama_call(final_prompt)

    if docs:
        sources = "\n".join(f" {doc.metadata.get('source', 'Unknown')} (pg {doc.metadata.get('page', '?')})" for doc in docs)
        answer += f"\n\nSources:\n{sources}"

    await cl.Message(content=answer).send()
