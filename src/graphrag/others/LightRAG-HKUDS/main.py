import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed

from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger
import numpy as np
from lightrag.llm.hf import hf_model_complete, hf_embed
from transformers import AutoModel, AutoTokenizer

WORKING_DIR = "./"
setup_logger("lightrag", level="INFO")

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "llama3-70b-8192",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        **kwargs
    )

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
            embed_model=AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
        )
    # return await openai_embed(
    #     texts,
    #     model="llama3-70b-8192",
    #     api_key=os.getenv("GROQ_API_KEY"),
    #     base_url="https://api.groq.com/openai/v1"
    #)

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=4096,
            max_token_size=8192,
            func=embedding_func
        )
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag
    
# async def initialize_rag():
    # rag = LightRAG(
        # working_dir="your/path",
        # embedding_func=openai_embed,
        # llm_model_func=gpt_4o_mini_complete
    # )

    # await rag.initialize_storages()
    # await initialize_pipeline_status()

    # return rag

def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())
    # Insert text
    data_file = "./data/book.txt"
    with open(data_file, "r", encoding="utf-8") as f:
            file_content = f.read()
            print(file_content)
            rag.insert(file_content)    
    # rag.insert("Your text")

    # Perform naive search
    mode="naive"
    # Perform local search
    mode="local"
    # Perform global search
    mode="global"
    # Perform hybrid search
    mode="hybrid"
    # Mix mode Integrates knowledge graph and vector retrieval.
    mode="mix"

    rag.query(
        "What are the top themes in this story?",
        param=QueryParam(mode=mode)
    )

if __name__ == "__main__":
    main()