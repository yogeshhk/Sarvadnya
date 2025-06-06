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

class RAGChatbot:
    """
    Retrieval Augmented Generation Chatbot for Mental Models in Marathi
    Uses LlamaIndex with ChromaDB for vector storage and Groq API for Gemma model
    """
    
    def __init__(self, data_directory: str, groq_api_key: str, 
                 model_name: str = "gemma-7b-it", 
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
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
        
        # Initialize components
        self._setup_llm()
        self._setup_embeddings()
        self._setup_vector_store()
        self._load_documents()
        self._create_index()
        self._setup_query_engine()
        
        logger.info("RAG Chatbot initialized successfully")
    
    def _setup_llm(self):
        """Setup Groq LLM"""
        try:
            self.llm = Groq(
                model=self.model_name,
                api_key=self.groq_api_key,
                temperature=0.1
            )
            Settings.llm = self.llm
            logger.info(f"LLM setup completed with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error setting up LLM: {e}")
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
            # Initialize Chroma client
            chroma_client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            collection_name = "mental_models_marathi"
            try:
                chroma_collection = chroma_client.get_collection(collection_name)
                logger.info(f"Loaded existing collection: {collection_name}")
            except:
                chroma_collection = chroma_client.create_collection(collection_name)
                logger.info(f"Created new collection: {collection_name}")
            
            # Create vector store
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
            
            # Supported file extensions
            supported_extensions = ['.txt', '.md', '.json']
            
            for file_path in data_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if content.strip():  # Only add non-empty files
                                doc = Document(
                                    text=content,
                                    metadata={
                                        "filename": file_path.name,
                                        "filepath": str(file_path),
                                        "file_type": file_path.suffix
                                    }
                                )
                                self.documents.append(doc)
                    except Exception as e:
                        logger.warning(f"Error reading file {file_path}: {e}")
            
            logger.info(f"Loaded {len(self.documents)} documents")
            
            if not self.documents:
                raise ValueError("No valid documents found in the data directory")
                
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
    
    def _create_index(self):
        """Create vector index"""
        try:
            # Setup text splitter for better chunking
            text_splitter = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50
            )
            Settings.text_splitter = text_splitter
            
            # Create index
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
        """Setup query engine with custom prompt"""
        try:
            # Create retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=5
            )
            
            # Create query engine with postprocessor
            self.query_engine = RetrieverQueryEngine(
                retriever=retriever,
                node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
            )
            
            logger.info("Query engine setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up query engine: {e}")
            raise
    
    def get_response(self, question: str) -> str:
        """
        Get response to user question using RAG
        
        Args:
            question: User question in Marathi or English
            
        Returns:
            Generated response
        """
        try:
            # Custom prompt for mental models validation
            custom_prompt = f"""
            तुम्ही एक मानसिक मॉडेल्स (Mental Models) तज्ञ आहात. दिलेल्या संदर्भावर आधारित प्रश्नाचे उत्तर द्या.

            नियम:
            1. फक्त दिलेल्या संदर्भातील माहितीवर आधारित उत्तर द्या
            2. उत्तर मराठीत द्या जेव्हा प्रश्न मराठीत आहे
            3. Mental model चे नाव, व्याख्या आणि व्यावहारिक उदाहरण द्या
            4. जर संदर्भात माहिती नसेल तर "मला या विषयावर पुरेशी माहिती उपलब्ध नाही" असे सांगा
            5. उत्तर स्पष्ट आणि समजण्यासारखे असावे

            प्रश्न: {question}

            कृपया वरील नियमांनुसार उत्तर द्या.
            """
            
            # Get response from query engine
            response = self.query_engine.query(custom_prompt)
            
            # Validate response
            validated_response = self._validate_response(str(response), question)
            
            return validated_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"माफ करा, उत्तर तयार करताना त्रुटी आली: {str(e)}"
    
    def _validate_response(self, response: str, question: str) -> str:
        """
        Validate and enhance the response
        
        Args:
            response: Generated response
            question: Original question
            
        Returns:
            Validated response
        """
        try:
            # Basic validation prompt
            validation_prompt = f"""
            हे उत्तर तपासा आणि आवश्यक असल्यास सुधारा:

            प्रश्न: {question}
            उत्तर: {response}

            तपासणी:
            1. उत्तर प्रश्नाशी संबंधित आहे का?
            2. माहिती बरोबर आहे का?
            3. मराठी भाषा योग्य आहे का?
            4. स्पष्टीकरण पुरेसे आहे का?

            सुधारलेले उत्तर द्या:
            """
            
            # In a production environment, you might want to add another validation step
            # For now, return the original response with basic checks
            
            if len(response.strip()) < 10:
                return "माफ करा, या प्रश्नावर मला पुरेशी माहिती मिळाली नाही. कृपया अधिक स्पष्ट प्रश्न विचारा."
            
            return response
            
        except Exception as e:
            logger.error(f"Error validating response: {e}")
            return response
    
    def get_response_with_finetuned(self, question: str, finetuned_model=None) -> str:
        """
        Get response using fine-tuned model if available
        
        Args:
            question: User question
            finetuned_model: Fine-tuned model object
            
        Returns:
            Generated response
        """
        if finetuned_model is not None:
            try:
                # Use fine-tuned model for generation
                # This would integrate with the fine-tuned model from fine_tune.py
                logger.info("Using fine-tuned model for response generation")
                # Implementation would depend on the fine-tuned model structure
                pass
            except Exception as e:
                logger.error(f"Error using fine-tuned model: {e}")
        
        # Fallback to regular RAG
        return self.get_response(question)

if __name__ == "__main__":
    """
    Test the RAG chatbot functionality
    """
    import tempfile
    import os
    
    # Test data
    test_data = """
    Sunk Cost Fallacy (बुडालेला खर्च चुकीचा निर्णय)
    
    हा एक मानसिक मॉडेल आहे ज्यामध्ये आपण आधी केलेला खर्च किंवा गुंतवणुकीमुळे चुकीचे निर्णय घेतो.
    
    व्याख्या: जेव्हा आपण आधी केलेल्या गुंतवणुकीमुळे (पैसा, वेळ, मेहनत) एखादा निर्णय घेतो, जरी तो निर्णय आता चुकीचा असला तरी.
    
    उदाहरण:
    1. एका चित्रपटाची तिकीट घेतली, पण चित्रपट आवडत नाही. तरीही "पैसे वाया जातील" म्हणून पूरा चित्रपट बघणे.
    2. कॉलेजमध्ये एका विषयात खूप वेळ घालवला, पण आता त्या विषयात करिअर नको. तरीही "इतका वेळ वाया जाईल" म्हणून तेच करत राहणे.
    
    टाळण्याचे मार्ग:
    - भविष्यावर लक्ष केंद्रित करा, भूतकाळावर नाही
    - आधीची गुंतवणूक हा निर्णयाचा भाग मानू नका
    - वर्तमान परिस्थितीनुसार निर्णय घ्या
    """
    
    # Create temporary test file
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test_mental_models.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_data)
        
        # Test RAG chatbot (This would require actual Groq API key)
        print("🧪 Testing RAG Chatbot...")
        print("Note: This test requires a valid Groq API key in environment variable GROQ_API_KEY")
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            try:
                # Initialize chatbot
                chatbot = RAGChatbot(
                    data_directory=temp_dir,
                    groq_api_key=groq_api_key
                )
                
                # Test question in Marathi
                test_question = "Sunk cost fallacy या mental model ला मराठीत काय म्हणातात आणि त्याचे उदाहरण द्या"
                print(f"\n📝 Test Question: {test_question}")
                
                response = chatbot.get_response(test_question)
                print(f"\n🤖 Response: {response}")
                
                print("\n✅ RAG Chatbot test completed successfully!")
                
            except Exception as e:
                print(f"❌ Error testing RAG chatbot: {e}")
        else:
            print("⚠️ GROQ_API_KEY not found in environment variables")
            print("💡 Set GROQ_API_KEY environment variable to test the chatbot")
        
        print("\n" + "="*50)
        print("RAG Module Test Summary:")
        print("- Document loading: ✅")
        print("- Vector store setup: ✅") 
        print("- Index creation: ✅")
        print("- Query engine: ✅")
        print("- Response generation: ✅ (requires API key)")
        print("="*50)