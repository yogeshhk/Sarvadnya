# app.py
import streamlit as st
import os
from rag import RAGSystem # Import the class from rag.py
from dotenv import load_dotenv

# --- 1. Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="Excel RAG Chatbot")

# --- 2. Load environment variables (for GROQ_API_KEY) ---
load_dotenv() # Ensures GROQ_API_KEY is available for RAGSystem

# --- 3. Initialize RAGSystem ---
# Cache the RAGSystem instance to avoid re-initializing on every script rerun
# and to preserve the loaded index.
@st.cache_resource
def get_rag_system():
    """Initializes and returns the RAGSystem."""
    try:
        # You can customize db_path and collection_name here if needed
        rag_system_instance = RAGSystem(
            db_path="./db/app_chroma_db", # Separate DB for the app
            collection_name="app_rag_collection"
        )
        # Attempt to load an existing index on startup if no files are processed yet.
        # This happens if rag_system_instance.index is still None after init.
        if rag_system_instance.index is None:
            rag_system_instance._load_existing_index() # Try to load
        return rag_system_instance
    except ValueError as e: # Catches GROQ_API_KEY error from RAGSystem
        st.error(f"Failed to initialize RAG System: {e}")
        st.error("Please ensure your GROQ_API_KEY is correctly set in your environment or .env file.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during RAG system initialization: {e}")
        return None

rag_system = get_rag_system()

# --- UI ---
st.title("📄 Chat with your Excel Data (RAG PoC)")

# --- Sidebar for Data Upload and Indexing ---
with st.sidebar:
    st.header("⚙️ Setup")

    if rag_system is None:
        st.sidebar.error("RAG System could not be initialized. Please check configuration (e.g., GROQ_API_KEY).")
    else:
        uploaded_files = st.file_uploader(
            "Upload Excel Files (with 'English Query' and 'Response' columns)",
            type=["xlsx"],
            accept_multiple_files=True,
            help="Upload one or more Excel files. Each should have 'English Query' and 'Response' columns."
        )

        if uploaded_files:
            if st.button("🔄 Process Files & Build/Load Index", key="process_files"):
                with st.spinner("Processing files and building index... This may take a moment."):
                    temp_file_paths = []
                    for uploaded_file in uploaded_files:
                        temp_dir = "temp_uploaded_files"
                        if not os.path.exists(temp_dir):
                            os.makedirs(temp_dir)
                        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        temp_file_paths.append(temp_file_path)
                    
                    if temp_file_paths:
                        message, num_docs, num_files = rag_system.load_and_index_data(temp_file_paths)
                        if rag_system.index:
                            st.success(f"✅ {message} Processed {num_docs} new Q&A pairs from {num_files} file(s).")
                            st.session_state.data_loaded_and_indexed = True
                        else:
                            st.error(f"⚠️ Failed to process files: {message}")
                            st.session_state.data_loaded_and_indexed = False
                        
                        # Clean up temporary files
                        for p in temp_file_paths:
                            try:
                                os.remove(p)
                            except Exception as e:
                                st.warning(f"Could not remove temporary file {p}: {e}")
                        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                            os.rmdir(temp_dir)

                    else:
                        st.warning("No files were successfully prepared for processing.")
        else:
            st.info("Upload Excel files to build or update the knowledge base.")

        # Display index status
        if rag_system and rag_system.index is not None:
            st.sidebar.success("Index is loaded and ready.")
            # Could add more details here if RAGSystem exposed them (e.g., number of items in index)
            st.session_state.data_loaded_and_indexed = True
        elif rag_system: # rag_system exists but index is None
            st.sidebar.warning("Index is not yet loaded. Please upload files or ensure an existing database can be found.")
            st.session_state.data_loaded_and_indexed = False


        st.markdown("---")
        st.subheader("ℹ️ About")
        if rag_system:
            st.markdown(f"""
            - **Embeddings:** `{rag_system.embed_model_name}`
            - **LLM:** Groq `{rag_system.llm_model_name}`
            - **Vector Store:** ChromaDB (local)
            - **Top-K Retrieval:** `{rag_system.top_k}`
            - **DB Path:** `{rag_system.db_path}`
            - **Collection:** `{rag_system.collection_name}`
            """)
        if os.getenv("GROQ_API_KEY"):
            st.success("GROQ_API_KEY found!")
        else:
            st.error("GROQ_API_KEY not found. The application may not function correctly.")


# --- Main Chat Interface ---
st.header("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload your Excel files via the sidebar, process them, and then I can help you query the data."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if user_query := st.chat_input("Ask something about your data...", key="chat_input"):
    if rag_system is None:
        st.error("⚠️ RAG System is not available. Please check the setup in the sidebar.")
    elif not st.session_state.get("data_loaded_and_indexed", False): # Check if index is ready
        st.error("⚠️ Please upload Excel file(s) and click 'Process Files & Build/Load Index' first, or ensure an existing index was loaded.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.spinner("Thinking... 🤔"):
            answer, retrieved_docs = rag_system.query_index(user_query)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)

            if retrieved_docs:
                with st.expander("📚 See Retrieved Information"):
                    for i, doc_info in enumerate(retrieved_docs):
                        st.markdown(f"**📄 Document {i+1} (Score: {doc_info['score']})** - Source: `{doc_info['source']}`")
                        st.text_area(f"Matched Query {i+1}", value=doc_info['matched_query'], height=70, disabled=True)
                        st.text_area(f"Retrieved Response {i+1}", value=doc_info['response'], height=100, disabled=True)
                        st.markdown("---")