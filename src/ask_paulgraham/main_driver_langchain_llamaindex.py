## LlamaIndex is not working well
# https://github.com/PradipNichite/Youtube-Tutorials/blob/main/LlamaIndex_Tutorial.ipynb
# https://github.com/jerryjliu/llama_index/issues/544
# https://colab.research.google.com/drive/16QMQePkONNlDpgiltOi7oRQgmB8dU5fl?usp=sharing#scrollTo=3323ec57

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.embeddings import VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.llms import GPT4All
from langchain.llms.base import LLM
import torch
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index import (
    GPTVectorStoreIndex,
    LangchainEmbedding,
    LLMPredictor,
    ServiceContext,
    StorageContext,
    download_loader,
    PromptHelper
)
from transformers import pipeline

PyMuPDFReader = download_loader("PyMuPDFReader")
documents = PyMuPDFReader().load(file_path='../data/On-Paul-Graham-2.pdf', metadata=True)

# ensure document texts are not bytes objects
for doc in documents:
    doc.text = doc.text.decode()

# GPT4ALL locall does not give any results
local_llm_path = "../models/ggml-gpt4all-j-v1.3-groovy.bin"
llm = GPT4All(model=local_llm_path, streaming=True)  # , backend='gptj', streaming=True, n_ctx=512)
llm_predictor = LLMPredictor(llm=llm)


class FlanLLM(LLM):
    model_name = "google/flan-t5-xl"
    pipeline = pipeline("text2text-generation", model=model_name, device="cpu",
                        model_kwargs={"torch_dtype": torch.bfloat16})

    def _call(self, prompt, stop=None):
        return self.pipeline(prompt, max_length=9999)[0]["generated_text"]

    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    def _llm_type(self):
        return "custom"


huggingFace_llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b", model_kwargs={"temperature": 1e-10})  # "google/flan-t5-xl"

llm_predictor = LLMPredictor(llm=huggingFace_llm)  # FlanLLM()

prompt_helper = PromptHelper(max_input_size=512, num_output=256, max_chunk_overlap=-1000)
huggingFace_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embed_model = LangchainEmbedding(huggingFace_embed)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embed_model,
    prompt_helper=prompt_helper,
    node_parser=SimpleNodeParser(text_splitter=TokenTextSplitter(chunk_size=300, chunk_overlap=20))
)

index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()

response = query_engine.query("What is this text about?")
print(response)

response = query_engine.query("who is this text about?")
print(response)

index.storage_context.persist(persist_dir="./storage")

from llama_index import load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
new_index = load_index_from_storage(storage_context, service_context=service_context)

query_engine = new_index.as_query_engine(similarity_top_k=1, service_context=service_context)
# response_stream = query_engine.query("who is this text about?")
# response_stream.print_response_stream()

response = query_engine.query("list 5 important points from this book")
print(response)

response = query_engine.query("what naval says about wealth creation")
print(response)
