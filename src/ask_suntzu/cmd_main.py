# Ref https://github.com/amrrs/QABot-LangChain/blob/main/Q%26A_Bot_with_Llama_Index_and_LangChain.ipynb

#from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex,GPTSimpleVectorIndex, PromptHelper
from llama_index import LLMPredictor, ServiceContext
import sys
import os


def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    index_obj = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    index_obj.save_to_disk('model/index.json')

    return index_obj


def ask_bot(input_index='model/index.json'):
    index_obj = GPTSimpleVectorIndex.load_from_disk(input_index)
    while True:
        query = input('What do you want to ask the bot?   \n')
        if query == "nothing":
            return
        response = index_obj.query(query, response_mode="compact")
        print("\nBot says: \n\n" + response.response + "\n\n\n")


index = construct_index("data/")

ask_bot('model/index.json')
