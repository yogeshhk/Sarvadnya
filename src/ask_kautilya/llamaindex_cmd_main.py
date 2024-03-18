# Ref https://github.com/amrrs/QABot-LangChain/blob/main/Q%26A_Bot_with_Llama_Index_and_LangChain.ipynb

#from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
# from langchain import OpenAI
from langchain.llms import VertexAI, LlamaCpp
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex,GPTVectorStoreIndex, PromptHelper
from llama_index import LLMPredictor, ServiceContext, load_index_from_storage, StorageContext
import sys
import os
from langchain.embeddings import HuggingFaceEmbeddings, LlamaCppEmbeddings


def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20

    # Chunk overlap ratio must be between 0 and 1
    chunk_overlap_ratio = 0.1

    # set chunk size limit
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, chunk_overlap_ratio, chunk_size_limit=chunk_size_limit)

    # define LLM
    model_path = "D:/Yogesh/GitHub/Sarvadnya/src/models/llama-7b.ggmlv3.q4_0.gguf.bin"
    llm = LlamaCpp(model_path=model_path) # VertexAI()  # need GCP account, project, config set in ENV variable
    # llm = VertexAI()  # need GCP account, project, config set in ENV variable
    llm_predictor = LLMPredictor(llm=llm)

    # embeddings = HuggingFaceEmbeddings()
    embeddings = LlamaCppEmbeddings(model_path=model_path)
    embed_model = LangchainEmbedding(embeddings)

    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)
    index_obj = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    index_obj.storage_context.persist('./model')

    return index_obj


def ask_bot(input_index='model/index.json'):
    storage_context = StorageContext.from_defaults(persist_dir="./model")
    index_obj = load_index_from_storage(storage_context)
    query_engine = index_obj.as_query_engine()
    # index_obj = GPTVectorStoreIndex(input_index)
    while True:
        query = input('What do you want to ask the bot?   \n')
        if query == "nothing":
            return
        response = query_engine.query(query)
        print("\nBot says: \n\n" + response + "\n\n\n")


index = construct_index("data/")

ask_bot('model/index.json')
