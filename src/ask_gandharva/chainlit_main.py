import os
import chainlit as cl
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI  # for Groq

# Load the .env file
load_dotenv()

# Fetch the API key from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = "https://api.groq.com/openai/v1"

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say you don't know.

Context: {context}
Question: {question}

Helpful answer:"""

def set_custom_prompt():
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    llm = ChatOpenAI(
        model_name="llama3-8b-8192",  # or mixtral-8x7b-32768
        temperature=0.3,
        streaming=True,
        openai_api_key=GROQ_API_KEY,
        openai_api_base=GROQ_API_BASE
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt()}
    )

@cl.on_chat_start
async def start():
    chain = qa_bot()
    cl.user_session.set("chain", chain)
    await cl.Message(content="Hi! Welcome to the Ask Bot. What would you like to know?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    if not chain:
        await cl.Message(content="Chain not initialized.").send()
        return

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res.get("source_documents", [])

    if sources:
        answer += "\n\nSources:\n"
        for doc in sources:
            src = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', '?')
            content = doc.page_content.strip().split('\n')[0]
            answer += f"- {src} (page {page}): {content}\n"
    else:
        answer += "\n\nNo sources found."

    await cl.Message(content=answer).send()
