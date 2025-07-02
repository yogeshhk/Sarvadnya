import chainlit as cl
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import Settings
import faiss
import pickle

from config import DOCS_INDEX, FAISS_STORE_PKL
from embedding import MiniLMEmbedding  # custom local embedding

@cl.on_chat_start
async def on_chat_start():
    Settings.embed_model = MiniLMEmbedding()
    Settings.llm = None  # No external LLM

    raw_faiss_index = faiss.read_index(DOCS_INDEX)
    vector_store = FaissVectorStore(raw_faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    with open(FAISS_STORE_PKL, "rb") as f:
        index = pickle.load(f)

    cl.user_session.set("index", index)
    await cl.Message(content=" Hii! Ask me anything about your documents.").send()

@cl.on_message
async def on_message(message: cl.Message):
    index = cl.user_session.get("index")
    if index is None:
        await cl.Message(content=" Index not loaded. Please run the ingestion first.").send()
        return

    query_engine = index.as_query_engine(llm=None)  # Force local only
    response = query_engine.query(message.content)

    # Format response with markdown
    await cl.Message(
        content=f"""### Answer

{str(response)}

---

_Ask me another question or type **exit** to end._"""
    ).send()
