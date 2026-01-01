import warnings
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

from llama_index.core import (
    ServiceContext,
    StorageContext,
    KnowledgeGraphIndex,
    Document,
    Settings,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
# from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode
from llama_index.core import Response
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.groq import Groq
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.base.llms.types import ChatMessage
import os
import unittest
import tempfile

# from llama_index import (
#     ServiceContext,
#     StorageContext,
#     KnowledgeGraphIndex,
#     VectorStoreIndex,
# )
# from llama_index.llms import LlamaCPP
# from llama_index.storage.storage_context import StorageContext
# from llama_index.graph_stores import SimpleGraphStore
# from llama_index.embeddings import HuggingFaceEmbedding
# from llama_index.schema import TextNode, Document
# from llama_index.response.schema import Response

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# LLAMA_MODEL_PATH = "llama-2-7b-chat.Q4_K_M.gguf"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME = "llama-3.1-8b-instant" #"llama-3.1-8b-instant" or "mistral-saba-24b", "llama-3.3-70b-versatile"
CONVERSATION_MODE = True

# Persistence configuration
PERSIST_BASE_DIR = "models"

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

class CitationConversationEngine:
    def __init__(self, base_conversation_engine):
        self.base_conversation_engine = base_conversation_engine

    def query(self, query: str, msgs: List[ChatMessage]) -> Response:
        response = self.base_conversation_engine.chat(query, msgs)
        source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
        
        referenced_ids = set()
        for node in source_nodes:
            if hasattr(node, 'metadata') and 'id' in node.metadata:
                referenced_ids.add(node.metadata['id'])
        
        formatted_response = f"{response.response}"
        new_response = Response(response=formatted_response, source_nodes=response.source_nodes)
        return new_response


def extract_text_from_node(node_data: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """Extract only the specified fields from node data."""
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

class GraphRAGBackend:
    def __init__(self, persist_base_dir: str = PERSIST_BASE_DIR):
        self.query_engine = None
        self.graph_store = None
        self.conversation_engine = None
        self.persist_base_dir = persist_base_dir
        self.data_source = None
        self.persist_dir = None
        self.index = None

    # def check_model_path(self):
    #     """Check if the model file exists and return the full path."""
    #     model_path = os.path.abspath(LLAMA_MODEL_PATH)
    #     if not os.path.exists(model_path):
    #         raise FileNotFoundError(f"Model file not found at: {model_path}")
    #     return model_path

    def _compute_data_hash(self, json_data: Dict[str, Any]) -> str:
        """Compute a hash of the JSON data to detect changes."""
        # Create a stable string representation of the data
        data_str = json.dumps(json_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _save_metadata(self, data_hash: str):
        """Save metadata about the persisted index."""
        metadata = {
            "data_hash": data_hash,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "llm_model": GROQ_MODEL_NAME
        }
        metadata_path = Path(self.persist_dir) / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load metadata about the persisted index."""
        metadata_path = Path(self.persist_dir) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None

    def _should_rebuild_index(self, json_data: Dict[str, Any]) -> bool:
        """Check if the index needs to be rebuilt."""
        if not Path(self.persist_dir).exists():
            return True
        
        metadata = self._load_metadata()
        if metadata is None:
            return True
        
        current_hash = self._compute_data_hash(json_data)
        if metadata.get("data_hash") != current_hash:
            return True
        
        if metadata.get("embedding_model") != EMBEDDING_MODEL_NAME:
            return True
        
        return False

    def _load_persisted_index(self):
        """Load the index from disk."""
        try:
            # Initialize LLM and embedding model
            llm = Groq(
                model=GROQ_MODEL_NAME,
                api_key=GROQ_API_KEY,
                temperature=0.1,
                max_tokens=512
            )
            embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
            
            # Configure Settings
            Settings.llm = llm
            Settings.embed_model = embed_model
            Settings.chunk_size = 512
            Settings.chunk_overlap = 20
            
            # Load storage context from disk (graph store is included)
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            
            # Load index
            index = load_index_from_storage(storage_context)
            
            # Get graph store from storage context
            self.graph_store = storage_context.graph_store
            self.index = index
            base_query_engine = index.as_query_engine(
                response_mode="compact",
                verbose=True
            )
            base_conversation_engine = index.as_chat_engine(
                chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT
            )
            self.query_engine = CitationQueryEngine(base_query_engine)
            self.conversation_engine = CitationConversationEngine(base_conversation_engine)

            print(f"Index loaded from {self.persist_dir} (data source: {self.data_source})")
            return True
        except Exception as e:
            raise Exception(f"Error loading persisted index: {str(e)}")

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for use in directory names."""
        import re
        # Replace non-alphanumeric characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Ensure it's not empty
        return sanitized if sanitized else "default"

    def setup_knowledge_base(self, json_data: Dict[str, Any], progress_callback=None, force_rebuild: bool = False, data_source: str = None):
        """Set up the knowledge base with the provided JSON data.

        Args:
            json_data: The JSON data containing the knowledge base
            progress_callback: Optional callback for progress updates
            force_rebuild: If True, rebuild the index even if persisted version exists
            data_source: Optional identifier for the data source (used for directory naming)
        """
        try:
            # Set data source and determine persist directory
            self.data_source = data_source or "default"
            # Sanitize data source name for directory name
            safe_data_source = self._sanitize_filename(self.data_source)
            self.persist_dir = f"{self.persist_base_dir}/graphrag_{safe_data_source}"

            # Check if we can load from disk
            if not force_rebuild and not self._should_rebuild_index(json_data):
                print(f"Loading persisted index from {self.persist_dir}...")
                return self._load_persisted_index()

            print(f"Building new index...")
            graph_store = SimpleGraphStore()
            storage_context = StorageContext.from_defaults(graph_store=graph_store)
            llm = Groq(
                model=GROQ_MODEL_NAME,
                api_key=GROQ_API_KEY,
                temperature=0.1,
                max_tokens=512
            )
            
            # model_path = self.check_model_path()
            # llm = LlamaCPP(
            #     model_path=model_path,
            #     model_kwargs={
            #         "n_ctx": 2048,
            #         "n_batch": 256,
            #         "n_threads": 4,
            #         "n_gpu_layers": 1
            #     },
            #     temperature=0.1,
            #     max_new_tokens=512,
            #     context_window=2048,
            #     generate_kwargs={},
            #     verbose=True
            # )
            
            embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
            
            # service_context = ServiceContext.from_defaults(
            #     llm=llm,
            #     embed_model=embed_model,
            #     chunk_size=512,
            #     chunk_overlap=20
            # )
            
            # Use Settings instead of ServiceContext (new in v0.10+)
            Settings.llm = llm
            Settings.embed_model = embed_model
            Settings.chunk_size = 512
            Settings.chunk_overlap = 20
            
            # Use SentenceSplitter for text splitting
            text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)            
                    
            elements = json_data.get('elements', {})
            nodes = elements.get('nodes', [])
            edges = elements.get('edges', [])
            
            documents = []
            node_lookup = {}
            
            batch_size = 20
            total_nodes = len(nodes)
            
            for batch_start in range(0, total_nodes, batch_size):
                batch_end = min(batch_start + batch_size, total_nodes)
                if progress_callback:
                    progress_callback(batch_start, batch_end, total_nodes)
                
                batch_nodes = nodes[batch_start:batch_end]
                for node in batch_nodes:
                    node_data = node.get('data', {})
                    node_id = node_data.get('id')
                    
                    if node_id:
                        node_text, essential_metadata = extract_text_from_node(node_data)
                        if node_text.strip():
                            doc = Document(
                                text=node_text,
                                metadata=essential_metadata
                            )
                            documents.append(doc)
                            node_lookup[node_id] = doc
            
            # Process edges
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
            
            index = KnowledgeGraphIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                # service_context=service_context,
                max_triplets_per_chunk=5,
                include_embeddings=True,
                show_progress=True
            )
            
            # Persist the index and graph store
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            # Persist storage context - use positional argument
            # This will also persist the graph store as it's part of storage_context
            index.storage_context.persist(self.persist_dir)
            
            # Save metadata
            data_hash = self._compute_data_hash(json_data)
            self._save_metadata(data_hash)
            
            print(f"Index and graph store persisted to {self.persist_dir} (data source: {self.data_source})")
            
            self.graph_store = graph_store
            self.index = index
            base_query_engine = index.as_query_engine(
                response_mode="compact",
                verbose=True
            )
            base_conversation_engine = index.as_chat_engine(
                chat_mode= ChatMode.CONDENSE_PLUS_CONTEXT
            )
            self.query_engine = CitationQueryEngine(base_query_engine)
            self.conversation_engine = CitationConversationEngine(base_conversation_engine)
            
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
        
    def process_conversation(self, query: str, msgs: List[ChatMessage]) -> str:
        """Process a query using the query engine."""
        if self.conversation_engine is None:
            raise Exception("Knowledge base not initialized. Please setup the knowledge base first.")
        
        try:
            response = self.conversation_engine.query(query, msgs)
            return response.response
        except Exception as e:
            raise Exception(f"Error processing query: {str(e)}")
        
class TestGraphRAGBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load actual graph data from file."""
        cls.json_file = 'graph_small.json'  # or 'data/graph.json'
        with open(cls.json_file, 'r', encoding='utf-8') as f:
            cls.test_data = json.load(f)
        
        # Only initialize if GROQ_API_KEY is available
        if not GROQ_API_KEY:
            raise unittest.SkipTest("GROQ_API_KEY not set")
    
    def test_extract_text_from_node(self):
        """Test node text extraction with real data."""
        node_data = self.test_data["elements"]["nodes"][0]["data"]
        text, metadata = extract_text_from_node(node_data)
        
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        self.assertIn("id", metadata)
    
    def test_setup_knowledge_base(self):
        """Test knowledge base setup with real data."""
        backend = GraphRAGBackend()
        success = backend.setup_knowledge_base(self.test_data)
        self.assertTrue(success)
        self.assertIsNotNone(backend.query_engine)
        self.assertIsNotNone(backend.graph_store)
    
    def test_process_simple_query(self):
        """Test processing a simple query."""
        backend = GraphRAGBackend()
        backend.setup_knowledge_base(self.test_data)
        
        query = "What is the definition of yoga?"
        response = backend.process_query(query)
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        self.assertIn("References:", response)
    
    def test_process_commentary_query(self):
        """Test query about Vyasa commentary."""
        backend = GraphRAGBackend()
        backend.setup_knowledge_base(self.test_data)
        
        query = "What does Vyasa commentary say?"
        response = backend.process_query(query)
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

def run_tests():
    """Run all GraphRAG backend tests."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGraphRAGBackend)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

if __name__ == "__main__":
    print("Running GraphRAG Backend tests...")
    success = run_tests()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")        