import streamlit as st
import json
import os
import psutil
from graphrag_backend import CONVERSATION_MODE, GraphRAGBackend
from llama_index.core.base.llms.types import ChatMessage, MessageRole

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'backend' not in st.session_state:
        st.session_state.backend = GraphRAGBackend()

def get_memory_usage():
    """Get current memory usage of the process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def display_chat_messages():
    """Display chat messages handling Dicts, Objects, and Enums."""
    
    if "messages" not in st.session_state:
        return

    for message in st.session_state.messages:
        role = ""
        content = ""

        # --- CASE 1: Message is a Dictionary ---
        if isinstance(message, dict):
            role = message["role"]
            content = message["content"]

        # --- CASE 2: Message is a LlamaIndex Object ---
        else:
            # 1. Get content
            content = message.content
            
            # 2. Handle the Enum Role
            # Check if role is an Enum (has a .value attribute)
            if hasattr(message.role, 'value'):
                role = message.role.value  # Converts MessageRole.USER -> "user"
            else:
                # Fallback if it happens to be a string already
                role = str(message.role)

        # --- VALIDATION: Ensure Streamlit compatibility ---
        # LlamaIndex might use "chatbot" or "system", map them if necessary.
        # Streamlit only has icons for "user" and "assistant".
        if role == "chatbot":
            role = "assistant"
            
        # Render
        with st.chat_message(role):
            st.markdown(content)

def main():
    st.title("Yoga Sutras Graph RAG Chatbot")
    
    initialize_session_state()

    with st.sidebar:
        st.header("Instructions")
        st.write("""
        1. Upload your JSON file containing the graph data
        2. Wait for the graph to be processed
        3. Ask questions about the Yoga Sutras
        
        Example questions:
        - What is the definition of yoga?
        - What does the Vyasa commentary say about sutra 1.1?
        - Explain the Sanskrit text and word analysis of sutra 1.2
        """)
        
        st.header("Model Information")
        if st.session_state.backend.query_engine is not None:
            st.success("✓ Models loaded and ready")
            if st.session_state.backend.graph_store is not None:
                st.success("✓ Knowledge graph active")
        else:
            st.info("⋯ Waiting for file upload")

    uploaded_file = st.file_uploader("Choose a JSON file", type="json")

    if uploaded_file is not None:
        try:
            json_data = json.load(uploaded_file)
            
            if st.session_state.backend.query_engine is None:
                with st.spinner("Processing graph data..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(batch_start, batch_end, total_nodes):
                        progress = batch_end / total_nodes
                        progress_bar.progress(progress)
                        status_text.text(f"Processing nodes {batch_start + 1} to {batch_end} of {total_nodes}")
                    
                    success = st.session_state.backend.setup_knowledge_base(
                        json_data,
                        progress_callback=update_progress
                    )
                    
                    if success:
                        st.success("Graph data processed successfully!")
                        progress_bar.empty()
                        status_text.empty()

        except Exception as e:
            st.error(f"Error loading JSON file: {str(e)}")

    display_chat_messages()

    if prompt := st.chat_input("What would you like to know about the Yoga Sutras?"):
        st.session_state.messages.append(ChatMessage(role= MessageRole.USER, content= prompt)) if CONVERSATION_MODE else st.session_state.messages.append({"role": "user", "content": prompt}) 
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.backend.process_conversation(prompt, st.session_state.messages) if CONVERSATION_MODE else st.session_state.backend.process_query(prompt)
                    st.markdown(response)
                    st.session_state.messages.append(ChatMessage(role= MessageRole.ASSISTANT, content= response)) if CONVERSATION_MODE else st.session_state.messages.append({"role": "assistant", "content": response})
                    print("prompt: ",st.session_state.messages)
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        
    if st.checkbox("Show Memory Usage"):
        memory_placeholder = st.empty()
        import time
        while True:
            memory_mb = get_memory_usage()
            memory_placeholder.text(f"Current Memory Usage: {memory_mb:.2f} MB")
            time.sleep(1)

if __name__ == "__main__":
    main()