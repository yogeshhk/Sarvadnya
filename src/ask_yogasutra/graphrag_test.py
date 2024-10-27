import streamlit as st
import json
import warnings
import os
from typing import Dict, Any, List, Optional
from llama_index import (
    ServiceContext,
    StorageContext,
    KnowledgeGraphIndex,
    VectorStoreIndex,
)
from llama_index.llms import LlamaCPP
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import SimpleGraphStore
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.schema import TextNode, Document
from llama_index.response.schema import Response
import networkx as nx
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLAMA_MODEL_PATH = "llama-2-7b-chat.Q4_K_M.gguf"

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'graph_store' not in st.session_state:
        st.session_state.graph_store = None
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'knowledge_graph' not in st.session_state:
        st.session_state.knowledge_graph = None

def check_model_path():
    """Check if the model file exists and return the full path."""
    model_path = os.path.abspath(LLAMA_MODEL_PATH)
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.error("Please make sure the model file is in the correct location.")
        st.stop()
    return model_path

def extract_text_from_node(node_data: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """
    Extract only the specified fields from node data.
    Returns a tuple of (text, metadata).
    """
    relevant_fields = [
        'Sanskrit_Text',
        'Word_for_Word_Analysis',
        'Vyasa_commentary'
    ]
    
    text_parts = []
    essential_metadata = {
        'id': node_data.get('id'),
        'sutra_number': node_data.get('sutra_number', ''),
        'chapter': node_data.get('chapter', '')
    }
    
    # Add only the specified fields to the text content
    for field in relevant_fields:
        if field in node_data and node_data[field]:
            text_parts.append(f"{field}: {node_data[field]}")
    
    return " ".join(text_parts), essential_metadata

class CitationQueryEngine:
    def __init__(self, base_query_engine):
        self.base_query_engine = base_query_engine

    def query(self, query_str: str) -> Response:
        # Get the base response
        response = self.base_query_engine.query(query_str)
        
        # Extract source nodes
        source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
        
        # Get unique IDs from source nodes
        referenced_ids = set()
        for node in source_nodes:
            if hasattr(node, 'metadata') and 'id' in node.metadata:
                referenced_ids.add(node.metadata['id'])
        
        # Format the response with citations
        formatted_response = f"{response.response}\n\nReferences: {', '.join(sorted(referenced_ids))}"
        
        # Create a new response object with the formatted text
        new_response = Response(response=formatted_response, source_nodes=response.source_nodes)
        return new_response

def setup_knowledge_base(json_data: Dict[str, Any]):
    """Set up the knowledge base with optimized node processing."""
    try:
        # Initialize the graph store
        graph_store = SimpleGraphStore()
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        
        model_path = check_model_path()
        llm = LlamaCPP(
            model_path=model_path,
            model_kwargs={
                "n_ctx": 2048,
                "n_batch": 256,
                "n_threads": 4,
                "n_gpu_layers": 1
            },
            temperature=0.1,
            max_new_tokens=512,
            context_window=2048,
            generate_kwargs={},
            verbose=True
        )
        
        # Initialize embedding model
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
        
        # Create service context with optimized settings
        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
            chunk_size=512,  # Increased chunk size
            chunk_overlap=20
        )
        
        # Process nodes and edges from JSON
        elements = json_data.get('elements', {})
        nodes = elements.get('nodes', [])
        edges = elements.get('edges', [])
        
        # Create documents list
        documents = []
        node_lookup = {}
        
        # Process nodes in larger batches for better performance
        batch_size = 20  # Increased batch size
        total_nodes = len(nodes)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for batch_start in range(0, total_nodes, batch_size):
            batch_end = min(batch_start + batch_size, total_nodes)
            status_text.text(f"Processing nodes {batch_start + 1} to {batch_end} of {total_nodes}")
            
            batch_nodes = nodes[batch_start:batch_end]
            for node in batch_nodes:
                node_data = node.get('data', {})
                node_id = node_data.get('id')
                
                if node_id:
                    try:
                        node_text, essential_metadata = extract_text_from_node(node_data)
                        
                        if node_text.strip():  # Only process nodes with actual content
                            doc = Document(
                                text=node_text,
                                metadata=essential_metadata
                            )
                            documents.append(doc)
                            node_lookup[node_id] = doc
                            
                    except Exception as e:
                        st.warning(f"Error processing node {node_id}: {str(e)}")
                        continue
            
            progress = (batch_end / total_nodes)
            progress_bar.progress(progress)
        
        # Process edges
        status_text.text("Processing edges...")
        
        for edge in edges:
            edge_data = edge.get('data', {})
            source = edge_data.get('source')
            target = edge_data.get('target')
            
            if source and target and source in node_lookup and target in node_lookup:
                relation = edge_data.get('relation', 'connected_to')
                source_doc = node_lookup[source]
                source_doc.metadata['relationships'] = source_doc.metadata.get('relationships', [])
                source_doc.metadata['relationships'].append({
                    'target': target,
                    'relation': relation
                })
        
        status_text.text("Creating knowledge graph index...")
        
        try:
            index = KnowledgeGraphIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                service_context=service_context,
                max_triplets_per_chunk=5,
                include_embeddings=True,
                show_progress=True
            )
            
            status_text.text("Knowledge graph created successfully!")
            progress_bar.empty()
            
            st.session_state.knowledge_graph = graph_store
            
            # Create the base query engine
            base_query_engine = index.as_query_engine(
                response_mode="compact",
                verbose=True
            )
            
            # Wrap it with our citation query engine
            return CitationQueryEngine(base_query_engine)
            
        except Exception as e:
            st.error(f"Error creating knowledge graph index: {str(e)}")
            raise
        
    except Exception as e:
        st.error(f"Error setting up knowledge base: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def get_memory_usage():
    """Get current memory usage of the process."""
    import psutil
    import os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def visualize_graph(graph_store: SimpleGraphStore):
    """Create a visualization of the knowledge graph."""
    try:
        G = nx.DiGraph()
        
        for node_id, node_data in graph_store._nodes.items():
            G.add_node(node_id, **node_data.get('metadata', {}))
            relationships = node_data.get('metadata', {}).get('relationships', [])
            for rel in relationships:
                target = rel.get('target')
                relation = rel.get('relation')
                if target:
                    G.add_edge(node_id, target, relation=relation)
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=2000, font_size=8, font_weight='bold')
        
        plt.savefig("temp_graph.png")
        plt.close()
        
        st.image("temp_graph.png")
        os.remove("temp_graph.png")
        
        st.write(f"Number of nodes: {G.number_of_nodes()}")
        st.write(f"Number of edges: {G.number_of_edges()}")
        
        return G
    except Exception as e:
        st.error(f"Error visualizing graph: {str(e)}")
        return None

def process_query(query: str) -> str:
    """Process a query using the query engine."""
    if st.session_state.query_engine is None:
        return "Please upload a JSON file first."
    
    try:
        response = st.session_state.query_engine.query(query)
        return response.response
    except Exception as e:
        return f"Error processing query: {str(e)}"

def display_chat_messages():
    """Display chat messages in the Streamlit interface."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    st.title("Yoga Sutras Graph RAG Chatbot")
    
    initialize_session_state()

    with st.sidebar:
        st.header("Instructions")
        st.write("""
        1. Upload your JSON file containing the graph data
        2. Wait for the graph to be processed
        3. Ask questions about the Yoga Sutras
        
        Example questions:
        - What is the definition of yoga?
        - What does the Vyasa commentary say about sutra 1.1?
        - Explain the Sanskrit text and word analysis of sutra 1.2
        """)
        
        st.header("Model Information")
        if st.session_state.query_engine is not None:
            st.success("✓ Models loaded and ready")
            if st.session_state.knowledge_graph is not None:
                st.success("✓ Knowledge graph active")
        else:
            st.info("⋯ Waiting for file upload")

    uploaded_file = st.file_uploader("Choose a JSON file", type="json")

    if uploaded_file is not None:
        try:
            json_data = json.load(uploaded_file)
            
            if st.session_state.query_engine is None:
                with st.spinner("Processing graph data..."):
                    st.session_state.query_engine = setup_knowledge_base(json_data)
                if st.session_state.query_engine is not None:
                    st.success("Graph data processed successfully!")
                    
                    with st.expander("View Knowledge Graph"):
                        st.write("Knowledge Graph Visualization")
                        if st.session_state.knowledge_graph is not None:
                            visualize_graph(st.session_state.knowledge_graph)

        except Exception as e:
            st.error(f"Error loading JSON file: {str(e)}")

    display_chat_messages()

    if prompt := st.chat_input("What would you like to know about the Yoga Sutras?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = process_query(prompt)
            st.markdown(response)
            
        st.session_state.messages.append({"role": "assistant", "content": response})
        
    if st.checkbox("Show Memory Usage"):
        memory_placeholder = st.empty()
        import time
        while True:
            memory_mb = get_memory_usage()
            memory_placeholder.text(f"Current Memory Usage: {memory_mb:.2f} MB")
            time.sleep(1)

if __name__ == "__main__":
    main()