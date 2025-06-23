
# Usage: streamlit run streamlit_main.py --server.fileWatcherType none

import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Suppress the torch.classes warning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define vectorstore directory directly (avoid config import issues)
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")

st.set_page_config(page_title="Ask Almanack of Naval Ravikant")

st.title("Ask Almanack of Naval Ravikant")
st.markdown("Ask anything from Almanack of Naval Ravikant documents. Powered by Groq & LLaMA3.")

# Load vectorstore (FAISS index) using the new format
@st.cache_resource
def load_vectorstore():
    try:
        # Check if vectorstore directory exists
        if not os.path.exists(VECTORSTORE_DIR):
            st.error(f"Vectorstore directory not found: {VECTORSTORE_DIR}")
            st.error("Please run the ingestion script first!")
            return None
        
        # Check if the required files exist
        index_file = os.path.join(VECTORSTORE_DIR, "index.faiss")
        pkl_file = os.path.join(VECTORSTORE_DIR, "index.pkl")
        
        if not os.path.exists(index_file) or not os.path.exists(pkl_file):
            st.error(f"Required vectorstore files not found:")
            st.error(f"Looking for: {index_file} and {pkl_file}")
            st.error("Please run the ingestion script first!")
            return None
        
        # Load embeddings (must match the one used during ingestion)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load FAISS vectorstore
        store = FAISS.load_local(
            VECTORSTORE_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        st.success(f"Vectorstore loaded successfully from {VECTORSTORE_DIR}")
        st.info(f"Found files: index.faiss, index.pkl")
        return store
        
    except Exception as e:
        st.error(f"Error loading vectorstore: {e}")
        st.error("Make sure you have run the ingestion script first!")
        return None

# Load vectorstore
store = load_vectorstore()

if store is not None:
    # Load Groq LLM
    try:
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY", "your_groq_api_key_here"),  # Use environment variable
            model_name="llama3-70b-8192"
        )
        
        # Setup Retrieval QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=store.as_retriever(search_kwargs={"k": 3})  # Return top 3 similar documents
        )
        
        # User Input
        query = st.text_input("Ask your question:")
        
        if query:
            with st.spinner("Thinking..."):
                try:
                    result = qa.invoke({"query": query})
                    st.markdown("### Answer")
                    st.write(result['result'])
                    
                    # Optional: Show source documents
                    with st.expander("View Source Documents"):
                        docs = store.similarity_search(query, k=3)
                        for i, doc in enumerate(docs, 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                            st.json(doc.metadata)
                            st.markdown("---")
                            
                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    
    except Exception as e:
        st.error(f"Error initializing Groq LLM: {e}")
        st.error("Please check your GROQ_API_KEY environment variable!")
        
else:
    st.warning("Vectorstore not loaded. Please run the ingestion script first!")
    st.code("python ingest.py")