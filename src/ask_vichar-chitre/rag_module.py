import os
import logging
from typing import List, Optional
from pathlib import Path

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

# Chroma imports
import chromadb
from chromadb.config import Settings as ChromaSettings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed model configurations
LLM_MODEL_AT_GROQ = "gemma2-9b-it"
EMBEDDING_MODEL_AT_HUGGINGFACE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class RAGChatbot:
    """
    Retrieval Augmented Generation Chatbot for Mental Models in Marathi
    Uses LlamaIndex with ChromaDB for vector storage and Groq API for Gemma model
    """
    
    def __init__(self, data_directory: str, groq_api_key: str, 
                 model_name: str = LLM_MODEL_AT_GROQ, 
                 embedding_model: str = EMBEDDING_MODEL_AT_HUGGINGFACE):
        """
        Initialize the RAG chatbot
        
        Args:
            data_directory: Path to directory containing mental models data files
            groq_api_key: Groq API key for accessing Gemma models
            model_name: Groq model name to use
            embedding_model: HuggingFace embedding model for multilingual support
        """
        self.data_directory = data_directory
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.db_path = "./chroma_db/"
        self.collection_name = "mental_models_marathi"
        
        # Initialize components step by step with diagnostics
        print("🔧 Step 1: Setting up LLM...")
        self._setup_llm()
        self._test_llm()
        
        print("🔧 Step 2: Setting up embeddings...")
        self._setup_embeddings()
        
        print("🔧 Step 3: Setting up vector store...")
        self._setup_vector_store()
        
        print("🔧 Step 4: Loading documents...")
        self._load_documents()
        
        print("🔧 Step 5: Creating index...")
        self._create_index()
        
        print("🔧 Step 6: Setting up query engine...")
        self._setup_query_engine()
        
        logger.info("RAG Chatbot initialized successfully")
    
    def _setup_llm(self):
        """Setup Groq LLM"""
        try:
            self.llm = Groq(
                model=self.model_name,
                api_key=self.groq_api_key,
                temperature=0.3,
                max_tokens=1024
            )
            Settings.llm = self.llm
            logger.info(f"LLM setup completed with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error setting up LLM: {e}")
            raise
    
    def _test_llm(self):
        """Test LLM directly"""
        try:
            print("🧪 Testing LLM directly...")
            test_prompt = "What is a mental model? Answer in one sentence."
            response = self.llm.complete(test_prompt)
            print(f"✅ LLM Direct Test Response: {str(response)}")
            
            # Test with Marathi
            marathi_prompt = "Mental model म्हणजे काय? एका वाक्यात उत्तर द्या."
            marathi_response = self.llm.complete(marathi_prompt)
            print(f"✅ LLM Marathi Test Response: {str(marathi_response)}")
            
        except Exception as e:
            print(f"❌ LLM Test Failed: {e}")
            raise
    
    def _setup_embeddings(self):
        """Setup multilingual embeddings"""
        try:
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.embedding_model,
                cache_folder="./embeddings_cache"
            )
            Settings.embed_model = self.embed_model
            logger.info(f"Embeddings setup completed with model: {self.embedding_model}")
        except Exception as e:
            logger.error(f"Error setting up embeddings: {e}")
            raise
    
    def _setup_vector_store(self):
        """Setup ChromaDB vector store"""
        try:
            os.makedirs(self.db_path, exist_ok=True)
            
            chroma_client = chromadb.PersistentClient(
                path=self.db_path,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            try:
                chroma_collection = chroma_client.get_collection(self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except:
                chroma_collection = chroma_client.create_collection(self.collection_name)
                logger.info(f"Created new collection: {self.collection_name}")
            
            self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
        except Exception as e:
            logger.error(f"Error setting up vector store: {e}")
            raise
    
    def _load_documents(self):
        """Load documents from data directory"""
        try:
            self.documents = []
            data_path = Path(self.data_directory)
            
            if not data_path.exists():
                raise ValueError(f"Data directory {data_path} does not exist")
            
            supported_extensions = ['.txt', '.tex', '.md', '.json']
            
            for file_path in data_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if content.strip():
                                doc = Document(
                                    text=content,
                                    metadata={
                                        "filename": file_path.name,
                                        "filepath": str(file_path),
                                        "file_type": file_path.suffix
                                    }
                                )
                                print(f"📄 Loaded: {file_path.name} ({len(content)} chars)")
                                self.documents.append(doc)
                    except Exception as e:
                        logger.warning(f"Error reading file {file_path}: {e}")
            
            logger.info(f"Total documents loaded: {len(self.documents)}")
            
            if not self.documents:
                raise ValueError("No valid documents found in the data directory")
                
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
    
    def _create_index(self):
        """Create vector index"""
        try:
            # Check if collection already has documents
            try:
                collection_count = self.vector_store.client.count()
                if collection_count > 0:
                    logger.info(f"Loading existing index with {collection_count} documents")
                    self.index = VectorStoreIndex.from_vector_store(self.vector_store)
                    return
            except:
                pass
            
            # Create new index
            logger.info("Creating new vector index...")
            
            # Setup text splitter
            text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
            Settings.text_splitter = text_splitter

            # Create index from documents
            self.index = VectorStoreIndex.from_documents(
                self.documents,
                storage_context=self.storage_context,
                show_progress=True
            )
            
            logger.info("Vector index created successfully")
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    def _setup_query_engine(self):
        """Setup query engine"""
        try:
            # Create retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=3
            )
            
            # Create query engine - simplest approach
            self.query_engine = RetrieverQueryEngine(
                retriever=retriever,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=0.5)
                ]
            )
            
            logger.info("Query engine setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up query engine: {e}")
            raise
    
    def diagnose_query_engine(self, question: str):
        """Diagnose query engine step by step"""
        try:
            print(f"\n🔍 Diagnosing query: {question}")
            
            # Step 1: Test retrieval
            print("Step 1: Testing retrieval...")
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=3
            )
            
            nodes = retriever.retrieve(question)
            print(f"✅ Retrieved {len(nodes)} nodes")
            
            if nodes:
                for i, node in enumerate(nodes):
                    print(f"   Node {i+1}: Score={node.score:.3f}, Text='{node.text[:100]}...'")
            
            # Step 2: Test query engine response
            print("Step 2: Testing query engine...")
            response = self.query_engine.query(question)
            print(f"Query engine response type: {type(response)}")
            print(f"Query engine response: '{str(response)}'")
            print(f"Response length: {len(str(response))}")
            
            # Step 3: Test with direct LLM call using retrieved context
            print("Step 3: Testing direct LLM with context...")
            if nodes:
                context = "\n\n".join([node.text for node in nodes[:2]])
                
                direct_prompt = f"""Context:
{context}

Question: {question}

Based on the context above, please answer the question. If the question is in Marathi, answer in Marathi."""
                
                direct_response = self.llm.complete(direct_prompt)
                print(f"✅ Direct LLM Response: {str(direct_response)}")
                
                return str(direct_response)
            
            return str(response)
            
        except Exception as e:
            print(f"❌ Diagnosis failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def get_response(self, question: str) -> str:
        """Get response with full diagnosis"""
        return self.diagnose_query_engine(question)

if __name__ == "__main__":
    """
    Test the RAG chatbot
    """
    import os

    print("🔍 Testing RAG Chatbot...")
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        print("❌ GROQ_API_KEY not found. Please set it in your environment.")
    else:
        try:
            chatbot = RAGChatbot(
                data_directory="data",
                groq_api_key=groq_api_key
            )
            
            # Test one question with full diagnosis
            test_question = "Mental model म्हणजे काय?"
            print(f"\n" + "="*80)
            print(f"FULL DIAGNOSTIC TEST")
            print(f"="*80)
            
            response = chatbot.get_response(test_question)
            
            print(f"\n🎯 Final Response: {response}")

        except Exception as e:
            print(f"❌ Error during testing: {e}")
            import traceback
            traceback.print_exc()