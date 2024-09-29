# from llama_index import ServiceContext, LLMPredictor, PromptHelper
import json
from typing import Dict, Any
from llama_index.core import ServiceContext

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import KnowledgeGraphIndex
from llama_index.llms.langchain import LangChainLLM
from llama_index.core.query_engine import ComposableGraphQueryEngine
from llama_index.core.storage.storage_context import StorageContext


def load_llm(llm_name: str):
    # This is a placeholder function. In a real application, you would load the appropriate model here.
    class MockLLM:
        def __call__(self, prompt: str) -> str:
            return f"This is a response from {llm_name} model."

    return MockLLM()


def process_json_to_graph(json_data: Dict[str, Any]) -> SimpleGraphStore:
    graph_store = SimpleGraphStore()

    # Process nodes
    for node in json_data.get('nodes', []):
        node_id = node.get('id')
        node_data = node.get('data', {})
        graph_store.upsert_node(node_id, node_data)

    # Process edges
    for edge in json_data.get('edges', []):
        source = edge.get('source')
        target = edge.get('target')
        relation = edge.get('relation', 'related_to')
        graph_store.upsert_edge(source, target, relation)

    return graph_store


def process_query(query: str, uploaded_file, llm_name: str) -> str:
    # Load the selected LLM
    llm = load_llm(llm_name)

    # Load and parse the JSON file
    json_data = json.load(uploaded_file)

    llm_langchain = LangChainLLM(llm=llm)

    # Create embeddings
    embed_model = HuggingFaceEmbeddings()

    # Create service context
    service_context = ServiceContext.from_defaults(
        llm=llm_langchain,
        embed_model=embed_model,
    )

    # Process JSON to graph
    graph_store = process_json_to_graph(json_data)
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    # Create knowledge graph index
    kg_index = KnowledgeGraphIndex.from_documents(
        [],  # Empty list since we've already processed the JSON
        storage_context=storage_context,
        service_context=service_context,
        include_embeddings=True,
    )

    # Create a composable graph query engine
    query_engine = ComposableGraphQueryEngine(
        kg_index,
        llm=llm,
        service_context=service_context,
    )

    # Query the graph
    response = query_engine.query(query)

    return response.response

# You can add more helper functions here as needed
