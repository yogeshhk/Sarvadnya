# import os
# import logging
# from typing import List, Optional
# from pathlib import Path

# from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.groq import Groq
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.core.postprocessor import SimilarityPostprocessor

# import chromadb
# from chromadb.config import Settings as ChromaSettings

# from dotenv import load_dotenv
# load_dotenv()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Model configs
# LLM_MODEL_AT_GROQ = "llama-3.1-8b-instant"
# EMBEDDING_MODEL_AT_HUGGINGFACE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# # → This is a multilingual sentence embedding model, great for retrieving semantically similar text.
# # It supports multiple languages including Marathi, which is perfect for your use case.
# class RAGChatbot:
#     """
#     Retrieval-Augmented Generation Chatbot for Mental Models in Marathi
#     Uses LlamaIndex + ChromaDB + Groq (Gemma)
#     """

#     def __init__(self, data_directory: str, groq_api_key: str,
#                  model_name: str = LLM_MODEL_AT_GROQ,
#                  embedding_model: str = EMBEDDING_MODEL_AT_HUGGINGFACE):
#         self.data_directory = data_directory
#         self.groq_api_key = groq_api_key
#         self.model_name = model_name
#         self.embedding_model = embedding_model
#         self.db_path = "./chroma_db/"
#         self.collection_name = "mental_models_marathi"

#         # Initialization steps
#         print("🔧 [1/6] Initializing LLM...")
#         self._setup_llm()
#         self._test_llm()

#         print("🔧 [2/6] Loading embedding model...")
#         self._setup_embeddings()

#         print("🔧 [3/6] Connecting to ChromaDB...")
#         self._setup_vector_store()

#         print("🔧 [4/6] Loading documents...")
#         self._load_documents()

#         print("🔧 [5/6] Creating index...")
#         self._create_index()

#         print("🔧 [6/6] Setting up query engine...")
#         self._setup_query_engine()

#         logger.info("✅ RAG Chatbot is ready to use!")

#     def _setup_llm(self):
#         try:
#             self.llm = Groq(
#                 model=self.model_name,
#                 api_key=self.groq_api_key,
#                 temperature=0.3,
#                 max_tokens=1024
#             )
#             Settings.llm = self.llm
#         except Exception as e:
#             logger.error(f"❌ LLM Setup Failed: {e}")
#             raise

#     def _test_llm(self):
#         try:
#             print("\n🧪 Running LLM self-test:")
#             prompts = [
#                 ("EN", "What is a mental model? Answer in one sentence."),
#                 ("MR", "Mental model म्हणजे काय? एका वाक्यात उत्तर द्या."),
#                 ("MR", "Inversion म्हणजे काय?")
#             ]
#             for lang, prompt in prompts:
#                 print(f"\n🧠 {lang} Prompt → {prompt}")
#                 res = self.llm.complete(prompt)
#                 print(f"🔁 Response → {res}")
#         except Exception as e:
#             print(f"❌ LLM Self-Test Failed: {e}")
#             raise

#     def _setup_embeddings(self):
#         try:
#             self.embed_model = HuggingFaceEmbedding(
#                 model_name=self.embedding_model,
#                 cache_folder="./embeddings_cache"
#             )
#             Settings.embed_model = self.embed_model
#         except Exception as e:
#             logger.error(f"❌ Embedding Setup Failed: {e}")
#             raise

#     def _setup_vector_store(self):
#         try:
#             os.makedirs(self.db_path, exist_ok=True)
#             chroma_client = chromadb.PersistentClient(
#                 path=self.db_path,
#                 settings=ChromaSettings(anonymized_telemetry=False)
#             )
#             try:
#                 chroma_collection = chroma_client.get_collection(self.collection_name)
#             except:
#                 chroma_collection = chroma_client.create_collection(self.collection_name)

#             self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
#             self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
#         except Exception as e:
#             logger.error(f"❌ Vector Store Setup Failed: {e}")
#             raise

#     def _load_documents(self):
#         try:
#             self.documents = []
#             path = Path(self.data_directory)
#             if not path.exists():
#                 raise FileNotFoundError(f"{self.data_directory} does not exist")

#             for file_path in path.rglob('*'):
#                 if file_path.suffix.lower() in [".txt", ".md", ".tex", ".json"]:
#                     with open(file_path, 'r', encoding='utf-8') as f:
#                         content = f.read().strip()
#                         if content:
#                             self.documents.append(Document(
#                                 text=content,
#                                 metadata={
#                                     "filename": file_path.name,
#                                     "filepath": str(file_path),
#                                     "file_type": file_path.suffix
#                                 }
#                             ))
#                             print(f"📄 Loaded: {file_path.name} ({len(content)} chars)")
#             if not self.documents:
#                 raise ValueError("No valid documents found.")
#         except Exception as e:
#             logger.error(f"❌ Document Load Failed: {e}")
#             raise

#     def _create_index(self):
#         try:
#             if self.vector_store.client.count() > 0:
#                 self.index = VectorStoreIndex.from_vector_store(self.vector_store)
#                 logger.info("✅ Loaded existing index.")
#                 return

#             Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
#             self.index = VectorStoreIndex.from_documents(
#                 self.documents,
#                 storage_context=self.storage_context,
#                 show_progress=True
#             )
#             logger.info("✅ Index created.")
#         except Exception as e:
#             logger.error(f"❌ Index Creation Failed: {e}")
#             raise

#     def _setup_query_engine(self):
#         try:
#             retriever = VectorIndexRetriever(index=self.index, similarity_top_k=3)
#             self.query_engine = RetrieverQueryEngine(
#                 retriever=retriever,
#                 node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)]
#             )
#         except Exception as e:
#             logger.error(f"❌ Query Engine Setup Failed: {e}")
#             raise

#     def diagnose_query_engine(self, question: str) -> str:
#         print(f"\n🧠 User Question: {question}")
#         print("-" * 80)

#         # Step 1: Retrieve
#         print("🔎 Step 1: Retrieving relevant nodes...")
#         retriever = VectorIndexRetriever(index=self.index, similarity_top_k=3)
#         nodes = retriever.retrieve(question)
#         for i, node in enumerate(nodes):
#             print(f"\n🔹 Node {i+1} (score={node.score:.2f}):\n{node.text[:200]}...\n")

#         # Step 2: Query engine
#         print("🧠 Step 2: Getting response from query engine...")
#         response = self.query_engine.query(question)
#         print(f"\n✅ Engine Response:\n{str(response)}")

#         # Step 3: Direct LLM with context
#         print("\n🤖 Step 3: Direct LLM using top context...")
#         if nodes:
#             context = "\n\n".join([node.text for node in nodes[:2]])
#             prompt = f"""Context:
# {context}

# Question: {question}

# Answer in Marathi if question is in Marathi."""
#             llm_response = self.llm.complete(prompt)
#             print(f"\n🎯 Final Answer:\n{llm_response}")
#             return str(llm_response)

#         return str(response)

#     # def get_response(self, question: str) -> str:
#     #     return self.diagnose_query_engine(question)
#     def get_response(self, query: str) -> dict:
#             try:
#                 # Step 1: retrieve top nodes
#                 retriever = VectorIndexRetriever(index=self.index, similarity_top_k=3)
#                 nodes = retriever.retrieve(query)

#                 # Step 2: generate final answer
#                 response = self.query_engine.query(query)

#                 # Step 3: prepare context text
#                 context = "\n\n".join([node.text for node in nodes])

#                 return {
#                     "answer": str(response),
#                     "context": context
#                 }
#             except Exception as e:
#                 return {
#                     "answer": f"❌ Error while generating answer: {e}",
#                     "context": ""
#                 }

# if __name__ == "__main__":
#     print("\n🚀 Starting Diagnostic Run for RAG Chatbot...\n")
#     groq_api_key = os.getenv("GROQ_API_KEY")

#     if not groq_api_key:
#         print("❌ GROQ_API_KEY not found. Please set it in your .env file or environment variables.")
#     else:
#         try:
#             chatbot = RAGChatbot(
#                 data_directory="data",
#                 groq_api_key=groq_api_key
#             )
#             print("\n" + "=" * 80)
#             print("🧪 TEST: Diagnostic for query 'Mental model म्हणजे काय?'")
#             print("=" * 80)
#             chatbot.get_response("Mental model म्हणजे काय?")
#         except Exception as e:
#             print(f"\n❌ Exception during chatbot run: {e}")
# import os
# import torch
# from dotenv import load_dotenv
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from accelerate import init_empty_weights, disk_offload
# from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# class RAGChatbot:
#     def __init__(self, data_directory: str):
#         load_dotenv()

#         # Step 1: Load documents
#         reader = SimpleDirectoryReader(input_dir=data_directory, recursive=True)
#         documents = reader.load_data()
#         print("📄 Documents loaded:", [doc.metadata.get("file_path") for doc in documents])

#         # Step 2: Embedding model
#         embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
#         Settings.embed_model = embed_model

#         # Step 3: Vector index and retriever
#         self.index = VectorStoreIndex.from_documents(documents)
#         self.retriever = self.index.as_retriever()
#         print("✅ Vector index and retriever initialized.")

#         # Step 4: Load tokenizer
#         print("🔁 Loading the Shivneri tokenizer...")
#         self.tokenizer = AutoTokenizer.from_pretrained("amitagh/shivneri-marathi-llm-7b-v0.1")

#         # Step 5: Disk offload model loading (manual)
#         print("🪶 Initializing empty model for disk offload...")
#         offload_dir = "offload_folder"
#         os.makedirs(offload_dir, exist_ok=True)

#         with init_empty_weights():
#             empty_model = AutoModelForCausalLM.from_pretrained("amitagh/shivneri-marathi-llm-7b-v0.1")

#         print("💾 Offloading model to disk...")
#         self.model = disk_offload(empty_model, offload_dir=offload_dir)

#         print("🚀 Shivneri model loaded with disk offload.")

#     def get_response(self, query: str) -> dict:
#         nodes = self.retriever.retrieve(query)
#         context_text = "\n".join([n.text for n in nodes])

#         prompt = (
#             "तुम्हाला मराठीत उत्तर द्यायचं आहे. खाली दिलेल्या माहितीच्या आधारे प्रश्नाचे उत्तर द्या:\n\n"
#             f"माहिती: {context_text}\n\n"
#             f"प्रश्न: {query}\n"
#             "उत्तर:"
#         )

#         inputs = self.tokenizer(prompt, return_tensors="pt")
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=200,
#                 temperature=0.7,
#                 do_sample=True,
#                 top_p=0.9
#             )
#         full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         answer = full_output.split("उत्तर:")[-1].strip()

#         return {
#             "answer": answer,
#             "context": context_text.strip()
#         }

# # ✅ Run test
# if __name__ == "__main__":
#     print("🛁 Done with your shower? Let's test the chatbot.")
#     bot = RAGChatbot(data_directory="data")
#     query = "Inversion म्हणजे काय?"
#     result = bot.get_response(query)

#     print("\n🧠 उत्तर:", result["answer"])
#     print("\n🔍 संदर्भ माहिती:\n", result["context"][:500])
import os
import logging
from typing import List
from pathlib import Path

from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

import chromadb
from chromadb.config import Settings as ChromaSettings
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
LLM_MODEL_AT_GROQ = "llama-3.1-8b-instant"
EMBEDDING_MODEL_AT_HUGGINGFACE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class RAGChatbot:
    def __init__(self, data_directory: str, groq_api_key: str,
                 model_name: str = LLM_MODEL_AT_GROQ,
                 embedding_model: str = EMBEDDING_MODEL_AT_HUGGINGFACE):
        self.data_directory = data_directory
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.db_path = "./chroma_db/"
        self.collection_name = "mental_models_marathi"

        print("🔧 [1/6] Initializing LLM...")
        self._setup_llm()
        self._test_llm()

        print("🔧 [2/6] Loading embedding model...")
        self._setup_embeddings()

        print("🔧 [3/6] Connecting to ChromaDB...")
        self._setup_vector_store()

        print("🔧 [4/6] Loading documents...")
        self._load_documents()

        print("🔧 [5/6] Creating index...")
        self._create_index()

        print("🔧 [6/6] Setting up query engine...")
        self._setup_query_engine()

        logger.info("✅ RAG Chatbot is ready to use!")

    def _setup_llm(self):
        try:
            self.llm = Groq(
                model=self.model_name,
                api_key=self.groq_api_key,
                temperature=0.3,
                max_tokens=1024
            )
            Settings.llm = self.llm
        except Exception as e:
            logger.error(f"❌ LLM Setup Failed: {e}")
            raise

    def _test_llm(self):
        try:
            print("\n🧪 LLM Self-test:")
            prompts = [
                "Mental model म्हणजे काय?",
                "Inversion म्हणजे काय?",
                "Explain mental models in 1 sentence."
            ]
            for prompt in prompts:
                print(f"\n🔹 Prompt: {prompt}")
                result = self.llm.complete(prompt)
                print(f"🔸 Response: {result}")
        except Exception as e:
            logger.error(f"❌ LLM self-test failed: {e}")
            raise

    def _setup_embeddings(self):
        try:
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.embedding_model,
                cache_folder="./embeddings_cache"
            )
            Settings.embed_model = self.embed_model
        except Exception as e:
            logger.error(f"❌ Embedding model load failed: {e}")
            raise

    def _setup_vector_store(self):
        try:
            os.makedirs(self.db_path, exist_ok=True)
            chroma_client = chromadb.PersistentClient(
                path=self.db_path,
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            try:
                chroma_collection = chroma_client.get_collection(self.collection_name)
            except Exception:
                chroma_collection = chroma_client.create_collection(self.collection_name)

            self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        except Exception as e:
            logger.error(f"❌ ChromaDB setup failed: {e}")
            raise

    def _load_documents(self):
        try:
            self.documents = []
            path = Path(self.data_directory)
            if not path.exists():
                raise FileNotFoundError(f"{self.data_directory} not found")

            for file_path in path.rglob("*"):
                if file_path.suffix.lower() in [".txt", ".md", ".tex"]:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            self.documents.append(Document(
                                text=content,
                                metadata={"filename": file_path.name}
                            ))
                            print(f"📄 Loaded: {file_path.name}")
            if not self.documents:
                raise ValueError("❌ No documents found!")
        except Exception as e:
            logger.error(f"❌ Document loading failed: {e}")
            raise

    def _create_index(self):
        try:
            if self.vector_store.client.count() > 0:
                self.index = VectorStoreIndex.from_vector_store(self.vector_store)
                logger.info("✅ Loaded existing index.")
            else:
                Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
                self.index = VectorStoreIndex.from_documents(
                    self.documents,
                    storage_context=self.storage_context,
                    show_progress=True
                )
                logger.info("✅ Created new index.")
        except Exception as e:
            logger.error(f"❌ Index creation failed: {e}")
            raise

    def _setup_query_engine(self):
        try:
            retriever = VectorIndexRetriever(index=self.index, similarity_top_k=3)
            self.query_engine = RetrieverQueryEngine(
                retriever=retriever,
                node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)]
            )
        except Exception as e:
            logger.error(f"❌ Query engine setup failed: {e}")
            raise

    def get_response(self, query: str) -> dict:
        try:
            retriever = VectorIndexRetriever(index=self.index, similarity_top_k=3)
            nodes = retriever.retrieve(query)
            response = self.query_engine.query(query)
            context = "\n\n".join([node.text for node in nodes])
            return {
                "answer": str(response).strip(),
                "context": context.strip()
            }
        except Exception as e:
            return {
                "answer": f"❌ Error while generating answer: {e}",
                "context": ""
            }

# Optional CLI test
if __name__ == "__main__":
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY missing. Set it in .env")
    else:
        bot = RAGChatbot(data_directory="data", groq_api_key=groq_api_key)
        result = bot.get_response("Inversion म्हणजे काय?")
        print("\n🧠 उत्तर:", result["answer"])
        print("\n📎 संदर्भ:\n", result["context"][:500])
