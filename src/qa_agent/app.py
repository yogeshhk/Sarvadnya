import streamlit as st
from agent import initialize_app
import io
import pypdf 
import os
from langchain_groq.chat_models import ChatGroq


# App title
st.title("Testcase Generation Agent")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Left sidebar
with st.sidebar:
    st.header("Configuration")
    
    uploaded_file = st.file_uploader("Upload Requirements Document", type=["txt", "pdf", "docx"])
    if "uploaded_file" not in st.session_state or uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    model_options = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "llama-3.1-8b-instant"
    ]
    
    # Initialize session state for the model if it doesn't exist
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "llama-3.1-8b-instant"
            
    selected_model = st.selectbox("Select Model", model_options, key="selected_model", index=model_options.index(st.session_state.selected_model))
    
    # Update the model in session state when changed
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        # Create a new LLM instance when model changes
        st.session_state.llm = ChatGroq(model=selected_model, temperature=0.0)
    
    # Initialize LLM if it doesn't exist
    if "llm" not in st.session_state:
        st.session_state.llm = ChatGroq(model=selected_model, temperature=0.0)
            
    reset_button = st.button("🔄 Reset Conversation", key="reset_button")
    if reset_button:
        st.session_state.messages = []   
        st.rerun()

# Initialize the LangGraph application with the selected model
app = initialize_app(model_name=st.session_state.selected_model)

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# Get requirements document content
requirements_docs_content = ""
if "uploaded_file" in st.session_state and st.session_state.uploaded_file is not None:
    if st.session_state.uploaded_file.type == "text/plain":
        requirements_docs_content = st.session_state.uploaded_file.getvalue().decode("utf-8")
    elif st.session_state.uploaded_file.type == "application/pdf":
        pdf_reader = pypdf.PdfReader(io.BytesIO(st.session_state.uploaded_file.getvalue()))
        for page in pdf_reader.pages:
            requirements_docs_content += page.extract_text()
elif os.path.exists("./content.txt"):  # Check if default file exists
    try:
        with open("./content.txt", "r", encoding='utf-8') as f:
            requirements_docs_content = f.read()
    except Exception as e:
        st.error(f"Error reading default file: {e}")
                         
# Main window
user_request = st.chat_input("Enter your request:")

if user_request:
    if len(user_request) > 150:
        st.error("Your question exceeds 150 characters. Please shorten it.")
    else:
        # Add user's message to session state and display it
        st.session_state.messages.append({"role": "user", "content": user_request})
        with st.chat_message("user"):
            # st.markdown(f"**You:** {user_request}")
            st.markdown(user_request)


        # Process with AI and get response
        with st.chat_message("assistant"):
            with st.spinner("Generating test cases..."):
                inputs = {"user_request": user_request, "requirements_docs_content": requirements_docs_content}
                
                # Create a placeholder for streaming output
                response_placeholder = st.empty()
                
                # Stream the output
                total_answer = ""
                for output in app.stream(inputs):
                    for node_name, state in output.items():
                        if 'answer' in state:
                            total_answer += state['answer']
                response_placeholder.markdown(total_answer)
                
                # Add the assistant's response to the chat history
                st.session_state.messages.append({"role": "assistant", "content": total_answer})
