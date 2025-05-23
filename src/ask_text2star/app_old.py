import streamlit as st
import pandas as pd
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os
from dotenv import load_dotenv

# --- 1. Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="Excel RAG Chatbot")

# Load environment variables (for GROQ_API_KEY)
load_dotenv()

# --- Configuration ---
# Use a local model for embeddings (faster and free)
# Using a smaller, faster model like 'all-MiniLM-L6-v2'
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# LLM from Groq - make sure GROQ_API_KEY is set in your environment
LLM_MODEL_NAME = "llama3-8b-8192"  # Or "mixtral-8x7b-32768"
CHUNK_SIZE = 512 # For splitting documents
TOP_K = 3 # Number of similar documents to retrieve

# --- LlamaIndex Settings ---
@st.cache_resource
def configure_llamaindex():
    """Configures LlamaIndex settings for embedding and LLM."""
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.llm = Groq(model=LLM_MODEL_NAME, api_key=os.getenv("GROQ_API_KEY"))
    Settings.chunk_size = CHUNK_SIZE
    # Settings.transformations = [SentenceSplitter(chunk_size=CHUNK_SIZE)] # Redundant if using Settings.chunk_size
    return Settings

Settings = configure_llamaindex()

# --- Data Loading and Indexing ---
def load_data_from_excel(file_path):
    """Loads data from an Excel file."""
    try:
        df = pd.read_excel(file_path)
        if "English Query" not in df.columns or "Response" not in df.columns:
            st.error(f"Excel file {file_path} must contain 'English Query' and 'Response' columns.")
            return []
        documents = []
        for _, row in df.iterrows():
            query = str(row["English Query"])
            response = str(row["Response"])
            # We store the response in metadata to retrieve it directly
            # The content of the document is the query itself, as that's what we search against
            doc = Document(
                text=query, # Content for similarity search
                metadata={
                    "response": response, # The actual answer
                    "filename": os.path.basename(file_path)
                }
            )
            documents.append(doc)
        return documents
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        st.error(f"Error loading data from {file_path}: {e}")
        return []

@st.cache_resource
def create_vector_index(documents, db_path="./models/chroma_db", collection_name="rag_excel_data"):
    """Creates or loads a ChromaDB vector index."""
    if not documents:
        st.warning("No documents provided to create index.")
        return None

    db = chromadb.PersistentClient(path=db_path)
    try:
        chroma_collection = db.get_collection(name=collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
        )
        st.info(f"Loaded existing index from collection: {collection_name}")
    except: # Heuristic: if get_collection fails, collection likely doesn't exist
        st.info(f"Creating new index for collection: {collection_name}")
        chroma_collection = db.create_collection(name=collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store, # This was missing, should be passed during construction
        )
        # No need to call index.storage_context.persist() for ChromaDB when client is persistent
    return index

# --- RAG Querying ---
def query_rag_index(index, user_query, top_k=TOP_K):
    """
    Queries the RAG index.
    Retrieves top_k similar documents and uses their 'response' metadata.
    Constructs a prompt for the LLM to generate a final answer.
    """
    if index is None:
        return "Index not available. Please upload data and build the index first.", []

    retriever = index.as_retriever(similarity_top_k=top_k)
    retrieved_nodes = retriever.retrieve(user_query)

    if not retrieved_nodes:
        return "I couldn't find any relevant information for your query.", []

    # For this specific RAG approach, we want to use the stored "Response" directly
    # if the query matches well, or let the LLM synthesize if needed.

    # Let's prepare context from the *responses* of the retrieved documents (queries)
    context_parts = []
    retrieved_info_for_display = []

    for node_with_score in retrieved_nodes:
        original_query = node_with_score.node.get_text()
        response_from_excel = node_with_score.node.metadata.get("response", "No response found in metadata.")
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

    # Build a more focused prompt for the LLM
    # The goal is to get the *best* answer from the retrieved ones or synthesize if necessary.
    # If one of the retrieved questions is very similar to the user query, its answer is likely the best.
    # The LLM can help pick or slightly rephrase.

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

    # Using the LLM to process the retrieved context and user query
    try:
        response_llm = Settings.llm.complete(prompt_template)
        final_answer = str(response_llm)
    except Exception as e:
        st.error(f"Error during LLM completion: {e}")
        final_answer = "Sorry, I encountered an error while generating the response."

    return final_answer, retrieved_info_for_display


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Excel RAG Chatbot")
st.title("📄 Chat with your Excel Data (RAG PoC)")

# --- Sidebar for Data Upload and Indexing ---
with st.sidebar:
    st.header("⚙️ Setup")

    uploaded_files = st.file_uploader(
        "Upload Excel Files (with 'English Query' and 'Response' columns)",
        type=["xlsx"],
        accept_multiple_files=True,
        help="Upload one or more Excel files. Each should have 'English Query' and 'Response' columns."
    )

    # Global variable to hold the index
    if 'vector_index' not in st.session_state:
        st.session_state.vector_index = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False


    if uploaded_files:
        if st.button("🔄 Process Files & Build/Load Index", key="process_files"):
            with st.spinner("Processing files and building index... This may take a moment."):
                all_documents = []
                for uploaded_file in uploaded_files:
                    # Save uploaded file temporarily to read with pandas
                    temp_file_path = os.path.join(".", uploaded_file.name) # Save in current dir
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    st.info(f"Loading data from: {uploaded_file.name}")
                    documents = load_data_from_excel(temp_file_path)
                    all_documents.extend(documents)
                    os.remove(temp_file_path) # Clean up temporary file

                if all_documents:
                    # Use a consistent collection name for persistence
                    collection_name = "my_excel_rag_collection_v2" # Changed for clarity
                    db_path = "./chroma_db_excel_poc" # Define a persistent path

                    # Ensure the directory for ChromaDB exists
                    if not os.path.exists(db_path):
                        os.makedirs(db_path)

                    st.session_state.vector_index = create_vector_index(
                        all_documents,
                        db_path=db_path,
                        collection_name=collection_name
                    )
                    if st.session_state.vector_index:
                        st.success(f"✅ Index built/loaded successfully with {len(all_documents)} query-response pairs from {len(uploaded_files)} file(s)!")
                        st.session_state.data_loaded = True
                    else:
                        st.error("Failed to build or load the index.")
                else:
                    st.warning("No valid documents found in the uploaded files. Index not built.")
                    st.session_state.data_loaded = False
    else:
        st.info("Upload Excel files to begin.")
        # Attempt to load existing index if no files are uploaded but an index might exist
        if st.session_state.vector_index is None and not st.session_state.data_loaded:
             # Try to load if a known persistent path and collection name exist
            collection_name = "my_excel_rag_collection_v2"
            db_path = "./chroma_db_excel_poc"
            if os.path.exists(db_path): # Only try if the db path exists
                try:
                    db_client = chromadb.PersistentClient(path=db_path)
                    # Check if collection exists before trying to get it
                    existing_collections = [col.name for col in db_client.list_collections()]
                    if collection_name in existing_collections:
                        st.session_state.vector_index = create_vector_index(
                            [], # Pass empty list if just loading
                            db_path=db_path,
                            collection_name=collection_name
                        )
                        if st.session_state.vector_index:
                            st.success(f"✅ Loaded existing index from '{collection_name}'. Ready to chat.")
                            st.session_state.data_loaded = True # Assume data was loaded if index is present
                        else:
                            st.info("Could not load a pre-existing index. Please upload files.")
                    else:
                        st.info("No pre-existing index collection found. Please upload files to create one.")
                except Exception as e:
                    st.warning(f"Could not automatically load existing index: {e}. Please upload files.")
            else:
                st.info("No local database found. Please upload files to create one.")


    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown(f"""
    This RAG PoC uses:
    - **Embeddings:** `{EMBED_MODEL_NAME}`
    - **LLM:** Groq `{LLM_MODEL_NAME}`
    - **Vector Store:** ChromaDB (local)
    - **Top-K Retrieval:** `{TOP_K}`
    """)
    if os.getenv("GROQ_API_KEY"):
        st.success("GROQ_API_KEY found!")
    else:
        st.error("GROQ_API_KEY not found. Please set it as an environment variable.")


# --- Main Chat Interface ---
st.header("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload your Excel files and I'll help you query them."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_query := st.chat_input("Ask something about your data...", key="chat_input"):
    if not st.session_state.data_loaded or st.session_state.vector_index is None:
        st.error("⚠️ Please upload Excel file(s) and click 'Process Files & Build/Load Index' first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.spinner("Thinking... 🤔"):
        answer, retrieved_docs = query_rag_index(st.session_state.vector_index, user_query)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

        if retrieved_docs:
            with st.expander("📚 See Retrieved Information"):
                for i, doc_info in enumerate(retrieved_docs):
                    st.markdown(f"**📄 Document {i+1} (Score: {doc_info['score']})** - Source: `{doc_info['source']}`")
                    st.text_area(f"Matched Query {i+1}", value=doc_info['matched_query'], height=50, disabled=True)
                    st.text_area(f"Retrieved Response {i+1}", value=doc_info['response'], height=100, disabled=True)
                    st.markdown("---")