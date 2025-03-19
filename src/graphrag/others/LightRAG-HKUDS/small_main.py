import os
import asyncio
from lightrag import LightRAG, QueryParam
# from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger
from lightrag.utils import EmbeddingFunc
from lightrag.llm.hf import hf_model_complete, hf_embed
from transformers import AutoModel, AutoTokenizer

setup_logger("lightrag", level="INFO")
WORKING_DIR = "./"
async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=hf_model_complete,
        llm_model_name='meta-llama/Llama-3.2-1B-Instruct',
        embedding_func=EmbeddingFunc(
            embedding_dim=4096,
            max_token_size=5000,
            func=lambda texts:hf_embed(
                texts,
                tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
                embed_model=AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                )
            )
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())
    # Insert text
    data_file = "./data/book.txt"
    with open(data_file, "r", encoding="utf-8") as f:
            file_content = f.read()
            print(file_content)
            rag.insert(file_content)   

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