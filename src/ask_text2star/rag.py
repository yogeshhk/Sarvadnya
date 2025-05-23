# rag.py
import os
import pandas as pd
# from dotenv import load_dotenv

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter # Though Settings.chunk_size handles this generally
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore # Corrected import
import chromadb

# # Load environment variables at the module level if needed by LlamaIndex settings early
# load_dotenv()

class RAGSystem:
    def __init__(self,
                 embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
                 llm_model_name="llama3-8b-8192",
                 chunk_size=512,
                 top_k=3,
                 db_path="./db/chroma_db_excel_poc_class",
                 collection_name="my_excel_rag_collection_class"):
        """
        Initializes the RAG system with specified configurations.
        """
        self.embed_model_name = embed_model_name
        self.llm_model_name = llm_model_name
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.db_path = db_path
        self.collection_name = collection_name
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        self._configure_llamaindex()
        self.index = None # Will be loaded or created

    def _configure_llamaindex(self):
        """Configures LlamaIndex global settings."""
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name)
        Settings.llm = Groq(model=self.llm_model_name, api_key=self.groq_api_key)
        Settings.chunk_size = self.chunk_size
        # Settings.node_parser = SentenceSplitter(chunk_size=self.chunk_size) # Alternative

    def _load_data_from_excel(self, file_path):
        """Loads data from a single Excel file."""
        try:
            df = pd.read_excel(file_path)
            if "English Query" not in df.columns or "Response" not in df.columns:
                print(f"Warning: Excel file {file_path} must contain 'English Query' and 'Response' columns. Skipping this file.")
                return []
            
            documents = []
            for _, row in df.iterrows():
                query = str(row["English Query"])
                response = str(row["Response"])
                doc = Document(
                    text=query,
                    metadata={
                        "response": response,
                        "filename": os.path.basename(file_path)
                    }
                )
                documents.append(doc)
            return documents
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return []
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            return []

    def load_and_index_data(self, file_paths: list):
        """
        Loads data from a list of Excel file paths and builds/loads the vector index.
        Returns a message about the outcome.
        """
        all_documents = []
        for file_path in file_paths:
            print(f"Loading data from: {file_path}")
            documents = self._load_data_from_excel(file_path)
            all_documents.extend(documents)

        if not all_documents:
            message = "No valid documents found in the provided files. Index not built/updated."
            print(message)
            # Try to load existing index if no new documents are provided
            if not self.index:
                 self._load_existing_index()
            return message, 0, 0


        # Ensure the directory for ChromaDB exists
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

        db = chromadb.PersistentClient(path=self.db_path)
        try:
            chroma_collection = db.get_collection(name=self.collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            # If documents are provided, we might want to add them to the existing index.
            # For simplicity in this version, if collection exists and new docs are provided,
            # we are creating a new index from these docs.
            # A more advanced version would handle updates.
            if all_documents:
                print(f"Re-building index for collection: {self.collection_name} with new documents.")
                # This will overwrite or create a new one if not handled carefully.
                # For persistent Chroma, `add` is better if index object already exists.
                # Simpler approach: always rebuild from all_documents if new files are processed.
                # Or, if the goal is to add to an existing index without re-reading all old data:
                # self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
                # self.index.insert_nodes([Document(...) for ... in all_documents]) # This requires nodes
                # For now, let's just rebuild from the current set of documents if files are processed.
                self.index = VectorStoreIndex.from_documents(
                    all_documents,
                    vector_store=vector_store, # This should be passed here to use the existing collection
                                               # or create if it does not exist (though we get it first)
                                               # This usage might be tricky; let's use from_vector_store if loading,
                                               # and from_documents if truly building fresh.
                )
                # A safer way for "create or update":
                # 1. Try to get collection.
                # 2. If exists, load index from it. Then add new documents.
                # 3. If not exists, create collection and index from new documents.

                # Simplified: If we process files, we rebuild the index with the content of these files.
                # To persist old data, user should re-upload all relevant files.
                # For true "load or create" and "add incrementally" the logic for index
                # and collection creation/update needs to be more granular.

                # Let's try a slightly more robust "load or create"
                new_index_created = False
                try:
                    # This line implicitly tries to use the existing collection if it exists
                    # and is compatible or creates a new one.
                    # However, `from_documents` is primarily for building from scratch.
                    # If we have `all_documents` it usually means we want to index these.
                    # For Chroma:
                    chroma_collection = db.get_or_create_collection(name=self.collection_name) # Ensures collection exists
                    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                    self.index = VectorStoreIndex.from_documents(
                        all_documents,
                        vector_store=vector_store, # This ensures it uses the specified collection
                    )
                    print(f"Built index with {len(all_documents)} documents into collection: {self.collection_name}")
                    new_index_created = True # Or updated
                except Exception as e: # More specific error handling might be needed
                    print(f"Error during index creation/update: {e}. Trying to load existing one if possible.")
                    self._load_existing_index()


            elif not self.index: # No new documents, try to load
                 self._load_existing_index()


        except Exception as e: # Broad exception for initial db.get_collection or create_collection
            print(f"Error accessing or creating Chroma collection: {e}. Attempting to create.")
            try:
                chroma_collection = db.create_collection(name=self.collection_name)
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                if all_documents:
                    self.index = VectorStoreIndex.from_documents(
                        all_documents,
                        vector_store=vector_store,
                    )
                    print(f"Created new index and collection: {self.collection_name} with {len(all_documents)} documents.")
                else:
                    # If no documents and collection had to be created, index remains None
                    self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store) # Empty index
                    print(f"Created new empty collection: {self.collection_name}. Index is empty.")
            except Exception as e_create:
                message = f"Critical error: Could not create ChromaDB collection: {e_create}"
                print(message)
                self.index = None
                return message, 0, 0
        
        if self.index:
            message = f"Index built/loaded successfully with {len(all_documents)} new query-response pairs processed."
            if all_documents: # Only count new docs if they were processed
                 return message, len(all_documents), len(file_paths)
            else: # Index was loaded, no new docs
                 # We need a way to count total items in loaded index if desired
                 # For now, report 0 new docs.
                 return "Existing index loaded. No new files processed.", 0, 0
        else:
            message = "Failed to build or load the index."
            print(message)
            return message, 0, 0

    def _load_existing_index(self):
        """Attempts to load an existing index from the configured ChromaDB."""
        try:
            db = chromadb.PersistentClient(path=self.db_path)
            chroma_collection = db.get_collection(name=self.collection_name) # Raises if not found
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            print(f"Successfully loaded existing index from collection: {self.collection_name}")
            # To get the count of items in a loaded Chroma index via LlamaIndex is not straightforward
            # without querying or accessing underlying store details.
            # print(f"Index contains approximately {len(self.index.docstore.docs)} documents.") # This might work
            return True
        except Exception as e: # Catches chromadb.errors.CollectionNotFoundError etc.
            print(f"Could not load existing index for collection '{self.collection_name}': {e}. A new one may be created if data is provided.")
            self.index = None
            return False


    def query_index(self, user_query: str):
        """
        Queries the RAG index.
        """
        if self.index is None:
            return "Index not available. Please load data first.", []

        retriever = self.index.as_retriever(similarity_top_k=self.top_k)
        retrieved_nodes = retriever.retrieve(user_query)

        if not retrieved_nodes:
            return "I couldn't find any relevant information for your query.", []

        context_parts = []
        retrieved_info_for_display = []

        for node_with_score in retrieved_nodes:
            original_query = node_with_score.node.get_text()
            response_from_excel = node_with_score.node.metadata.get("response", "No response found.")
            score = node_with_score.score
            filename = node_with_score.node.metadata.get("filename", "N/A")

            context_parts.append(f"Retrieved Question: {original_query}\nRetrieved Answer: {response_from_excel}\n")
            retrieved_info_for_display.append({
                "matched_query": original_query,
                "response": response_from_excel,
                "score": f"{score:.4f}",
                "source": filename
            })
        
        context_str = "\n---\n".join(context_parts)
        
        prompt_template = f"""
        You are a helpful assistant. Based on the following retrieved information, which consists of question-answer pairs from our knowledge base,
        please answer the user's query.

        If the user's query is very similar to one of the 'Retrieved Questions', prioritize using its corresponding 'Retrieved Answer'.
        If the user's query is broader or requires combining information, synthesize a concise answer from the relevant 'Retrieved Answers'.
        If none of the retrieved information seems directly relevant to the user's query, state that you couldn't find a specific answer in the provided context.

        Retrieved Information:
        {context_str}

        User's Query: {user_query}

        Your Answer:
        """

        try:
            response_llm = Settings.llm.complete(prompt_template)
            final_answer = str(response_llm)
        except Exception as e:
            print(f"Error during LLM completion: {e}")
            final_answer = "Sorry, I encountered an error while generating the response."
        
        return final_answer, retrieved_info_for_display


# --- Main block for testing RAGSystem ---
if __name__ == "__main__":
    print("Testing RAGSystem - Inference Mode...")

    # --- Configuration ---
    DATA_DIR = "./data"
    TEST_DB_PATH = "./db/test_rag_chroma_db_inference" # Use a specific DB for this test
    TEST_COLLECTION_NAME = "test_rag_collection_inference"

    # --- Check for API Key ---
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not set. Please set it in your .env file or environment.")
        exit()

    # --- Check for Data Directory and Files ---
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        print("Please create it and place your Excel files (faq_data.xlsx, etc.) inside.")
        exit()

    excel_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.xlsx')]

    if not excel_files:
        print(f"Error: No .xlsx files found in the '{DATA_DIR}' directory.")
        exit()

    print(f"Found {len(excel_files)} Excel files in '{DATA_DIR}':")
    for f in excel_files:
        print(f"  - {os.path.basename(f)}")

    # 1. Initialize RAGSystem
    try:
        rag_system = RAGSystem(
            db_path=TEST_DB_PATH,
            collection_name=TEST_COLLECTION_NAME
        )
        print("\nRAGSystem initialized.")
    except ValueError as e:
        print(f"Failed to initialize RAGSystem: {e}")
        exit()

    # 2. Load and Index Data (This will build/load the index)
    print(f"\nAttempting to load and index data from: {DATA_DIR}")
    message, num_docs, num_files = rag_system.load_and_index_data(excel_files)
    print(f"Load and Index Result: {message}")

    if rag_system.index is None:
        print("Index was not created or loaded. Cannot proceed with inference. Exiting test.")
        exit()
    else:
        print("Index is ready for querying.")


    # 3. Test Querying (Inference)
    print("\n--- Testing Inference Queries ---")
    queries_to_test = [
        # FAQ Queries
        "What are your business hours?",
        "How can I reset my password?",
        # Text-to-SQL Queries
        "Show me all users from New York",
        "How many orders were placed last month?",
        # QA Testing Queries
        "How to test user login functionality?",
        "Steps to verify search product feature?",
        # Coding Queries
        "How to write a Python function to read a CSV?",
        "Basic Flask app structure in Python",
        # General/Out-of-Scope Query
        "What's the weather like today?"
    ]

    for q_idx, query in enumerate(queries_to_test):
        print(f"\n--- Query {q_idx+1}: '{query}' ---")
        answer, retrieved = rag_system.query_index(query)
        print(f"🤖 Answer: {answer}")
        if retrieved:
            print("🔍 Retrieved Context:")
            for i, doc_info in enumerate(retrieved):
                print(f"  - Source: {doc_info['source']}, Score: {doc_info['score']}")
                print(f"    Matched Q: {doc_info['matched_query'][:80]}...") # Show snippet
                # print(f"    Retrieved A: {doc_info['response'][:80]}...") # Optionally show answer snippet
        else:
            print("  (No specific documents retrieved)")

    print("\n--- RAGSystem Inference Testing Complete ---")
    print(f"Note: The test database is stored at '{TEST_DB_PATH}'. You might want to remove it manually if not needed.")