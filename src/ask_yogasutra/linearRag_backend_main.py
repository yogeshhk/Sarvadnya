import warnings
import os
from typing import Dict, Any, List, Optional
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    Document,
    StorageContext
)
from llama_index.llms import LlamaCPP
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.schema import TextNode
from llama_index.response.schema import Response

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLAMA_MODEL_PATH = "llama-2-7b-chat.Q4_K_M.gguf"

class CitationQueryEngine:
    def __init__(self, base_query_engine):
        self.base_query_engine = base_query_engine

    def query(self, query_str: str) -> Response:
        response = self.base_query_engine.query(query_str)
        source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
        
        referenced_ids = set()
        for node in source_nodes:
            if hasattr(node, 'metadata') and 'id' in node.metadata:
                referenced_ids.add(node.metadata['id'])
        
        formatted_response = f"{response.response}\n\nReferences: {', '.join(sorted(referenced_ids))}"
        new_response = Response(response=formatted_response, source_nodes=response.source_nodes)
        return new_response

def extract_text_from_node(node_data: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """Extract text and metadata from node data."""
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
    
    for field in relevant_fields:
        if field in node_data and node_data[field]:
            text_parts.append(f"{field}: {node_data[field]}")
    
    return " ".join(text_parts), essential_metadata

class LinearRAGBackend:
    def __init__(self):
        self.query_engine = None
        self.storage_context = None

    def check_model_path(self):
        """Check if the model file exists and return the full path."""
        model_path = os.path.abspath(LLAMA_MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        return model_path

    def setup_knowledge_base(self, json_data: Dict[str, Any], progress_callback=None):
        """Set up the knowledge base with the provided JSON data."""
        try:
            model_path = self.check_model_path()
            
            # Initialize LLM
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
            
            # Create service context
            service_context = ServiceContext.from_defaults(
                llm=llm,
                embed_model=embed_model,
                chunk_size=512,
                chunk_overlap=20
            )
            
            # Process nodes
            elements = json_data.get('elements', {})
            nodes = elements.get('nodes', [])
            
            documents = []
            batch_size = 20
            total_nodes = len(nodes)
            
            for batch_start in range(0, total_nodes, batch_size):
                batch_end = min(batch_start + batch_size, total_nodes)
                if progress_callback:
                    progress_callback(batch_start, batch_end, total_nodes)
                
                batch_nodes = nodes[batch_start:batch_end]
                for node in batch_nodes:
                    node_data = node.get('data', {})
                    node_text, essential_metadata = extract_text_from_node(node_data)
                    
                    if node_text.strip():
                        doc = Document(
                            text=node_text,
                            metadata=essential_metadata
                        )
                        documents.append(doc)
            
            # Create vector store index
            storage_context = StorageContext.from_defaults()
            index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                service_context=service_context,
                show_progress=True
            )
            
            self.storage_context = storage_context
            base_query_engine = index.as_query_engine(
                response_mode="compact",
                verbose=True
            )
            self.query_engine = CitationQueryEngine(base_query_engine)
            
            return True
            
        except Exception as e:
            raise Exception(f"Error setting up knowledge base: {str(e)}")

    def process_query(self, query: str) -> str:
        """Process a query using the query engine."""
        if self.query_engine is None:
            raise Exception("Knowledge base not initialized. Please setup the knowledge base first.")
        
        try:
            response = self.query_engine.query(query)
            return response.response
        except Exception as e:
            raise Exception(f"Error processing query: {str(e)}")