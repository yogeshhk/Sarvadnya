import streamlit as st
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.node_parser import SimpleNodeParser, TokenTextSplitter
# from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.readers.json.base import JSONReader
from graphrag_backend import process_query

# Set page configuration
st.set_page_config(page_title="Ask Yogasutra", layout="wide")

# Sidebar
st.sidebar.title("Configuration")

# File uploader widget
uploaded_file = st.sidebar.file_uploader("Upload JSON file", type="json")

# LLM selection dropdown
llm_options = ["gemma", "llama", "mistral"]
selected_llm = st.sidebar.selectbox("Select LLM", llm_options)

# Main window
st.title("Ask Yogasutra")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know about Yogasutra?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if uploaded_file is not None:
        response = process_query(prompt, uploaded_file, selected_llm)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please upload a JSON file to proceed.")

# Add a footer
st.markdown("---")
st.markdown("Built with ❤️ using LangChain, LlamaIndex, and Streamlit")
