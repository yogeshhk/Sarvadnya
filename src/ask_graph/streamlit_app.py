import streamlit as st
import os
import pandas as pd
from graphrag import GraphRAG
import tempfile

# Page configuration
st.set_page_config(
    page_title="GraphRAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'graph_rag' not in st.session_state:
    st.session_state.graph_rag = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def initialize_graphrag(api_key):
    """Initialize GraphRAG instance"""
    try:
        graph_rag = GraphRAG(api_key)
        return graph_rag, None
    except Exception as e:
        return None, str(e)

def load_data(graph_rag, uploaded_file):
    """Load and process uploaded data"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load and segment data
        segments = graph_rag.load_and_segment_data(tmp_path)
        
        if not segments:
            return False, "No data segments found in the uploaded file"
        
        # Create graph
        graph_rag.create_dummy_graph(segments)
        
        # Store in ChromaDB
        graph_rag.store_in_chroma()
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return True, f"Successfully loaded {len(segments)} data segments and created graph with {graph_rag.graph.number_of_nodes()} nodes"
        
    except Exception as e:
        return False, f"Error loading data: {str(e)}"

def main():
    st.title("🤖 GraphRAG Chatbot")
    st.markdown("Ask questions about your data using Graph-based Retrieval Augmented Generation")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key")
        
        if api_key:
            if st.session_state.graph_rag is None:
                with st.spinner("Initializing GraphRAG..."):
                    graph_rag, error = initialize_graphrag(api_key)
                    if graph_rag:
                        st.session_state.graph_rag = graph_rag
                        st.success("GraphRAG initialized successfully!")
                    else:
                        st.error(f"Failed to initialize GraphRAG: {error}")
        
        st.divider()
        
        # File upload
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])
        
        if uploaded_file and st.session_state.graph_rag and not st.session_state.data_loaded:
            if st.button("Load Data"):
                with st.spinner("Loading and processing data..."):
                    success, message = load_data(st.session_state.graph_rag, uploaded_file)
                    if success:
                        st.session_state.data_loaded = True
                        st.success(message)
                    else:
                        st.error(message)
        
        # Display data status
        if st.session_state.data_loaded:
            st.success("✅ Data loaded and graph created")
            if st.session_state.graph_rag:
                st.info(f"Nodes: {st.session_state.graph_rag.graph.number_of_nodes()}")
                st.info(f"Edges: {st.session_state.graph_rag.graph.number_of_edges()}")
        
        st.divider()
        
        # Clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface
    if not api_key:
        st.info("👈 Please enter your Groq API key in the sidebar to get started")
        return
    
    if not st.session_state.graph_rag:
        st.info("Please wait while GraphRAG is being initialized...")
        return
        
    if not st.session_state.data_loaded:
        st.info("👈 Please upload an Excel file in the sidebar to load your data")
        return
    
    # Chat interface
    st.header("Chat with your data")
    
    # Display chat history
    for i, (query, response) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**You:** {query}")
            st.markdown(f"**Bot:** {response['answer']}")
            
            # Show additional details in expander
            with st.expander(f"Details for query {i+1}"):
                st.markdown("**Context used:**")
                st.text(response.get('context', 'No context available'))
                st.markdown(f"**Nodes retrieved:** {response.get('num_nodes_retrieved', 0)}")
                
                if response.get('retrieved_nodes'):
                    st.markdown("**Retrieved nodes:**")
                    for node in response['retrieved_nodes']:
                        st.text(f"- {node['id']}: {node['text'][:100]}...")
            
            st.divider()
    
    # Query input
    user_query = st.text_input("Ask a question about your data:", key="user_input")
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        ask_button = st.button("Ask", type="primary")
    
    if ask_button and user_query:
        if st.session_state.graph_rag and st.session_state.data_loaded:
            with st.spinner("Generating response..."):
                try:
                    response = st.session_state.graph_rag.answer_query(user_query)
                    st.session_state.chat_history.append((user_query, response))
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        else:
            st.error("Please ensure GraphRAG is initialized and data is loaded")
    
    # Sample queries
    if st.session_state.data_loaded:
        st.markdown("### 💡 Sample Questions")
        sample_queries = [
            "What products are available?",
            "Show me items under $100",
            "What electronics do you have?",
            "Tell me about the most expensive item"
        ]
        
        cols = st.columns(2)
        for i, query in enumerate(sample_queries):
            with cols[i % 2]:
                if st.button(query, key=f"sample_{i}"):
                    if st.session_state.graph_rag:
                        with st.spinner("Generating response..."):
                            try:
                                response = st.session_state.graph_rag.answer_query(query)
                                st.session_state.chat_history.append((query, response))
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
