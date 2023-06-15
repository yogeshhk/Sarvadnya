# https://github.com/jerryjliu/llama_index/blob/main/examples/langchain_demo/LangchainDemo.ipynb

# Using LlamaIndex as a Callable Tool

from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain import HuggingFaceHub

from llama_index import VectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext

documents = SimpleDirectoryReader('data/experiment').load_data()
repo_id = "tiiuae/falcon-7b"

llm_predictor = LLMPredictor(llm=HuggingFaceHub(repo_id=repo_id,
                                                model_kwargs={"temperature": 0.1, 'truncation': 'only_first',
                                                              "max_length": 512}))
service_context = ServiceContext.from_defaults(chunk_size=64, llm_predictor=llm_predictor)

index = VectorStoreIndex.from_documents(documents=documents, service_context=service_context)
print(index)