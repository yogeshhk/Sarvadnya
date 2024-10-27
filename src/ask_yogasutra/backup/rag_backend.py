import json

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.llms.langchain import LangChainLLM
from llama_index.readers.json.base import JSONReader


def load_llm(llm_name):
    # This is a placeholder function. In a real application, you would load the appropriate model here.
    # For demonstration purposes, we'll return a mock LLM.
    class MockLLM:
        def __call__(self, prompt):
            return f"This is a response from {llm_name} model."

    return MockLLM()


def process_query(query, uploaded_file, llm_name):
    # Load the selected LLM
    llm = load_llm(llm_name)

    # Load and parse the JSON file
    json_data = json.load(uploaded_file)
    documents = JSONReader().load_data(json_data)

    llm_langchain = LangChainLLM(llm=llm)

    # Create embeddings
    embed_model = HuggingFaceEmbeddings()

    # Create service context
    service_context = ServiceContext.from_defaults(
        llm=llm_langchain,
        embed_model=embed_model,
    )

    # Create index
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context
    )

    # Query the index
    query_engine = index.as_query_engine()
    response = query_engine.query(query)

    return response.response

# You can add more helper functions here as needed
