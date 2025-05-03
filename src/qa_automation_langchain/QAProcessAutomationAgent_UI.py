import streamlit as st
import io
import pypdf
import os
from QAProcessAutomationAgent import simple_AI_Function_Agent, convert_requirements_to_testcases
from groq import Groq

# App title
st.title("Requirements to Testcase Converter")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Setup Groq client
if "groq_client" not in st.session_state:
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        st.session_state.groq_client = Groq(api_key=api_key)
    else:
        st.error("GROQ_API_KEY not found in environment variables")

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
        "gemma2-9b-it"
    ]

    # Initialize session state for the model if it doesn't exist
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "llama-3.3-70b-versatile"

    selected_model = st.selectbox("Select Model", model_options, key="selected_model",
                                  index=model_options.index(st.session_state.selected_model))

    # Update the model in session state when changed
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model

    # Add workflow selection
    workflow_options = [
        "Complete Workflow (Summary → Gherkin → Selenium)",
        "Generate Summary Only",
        "Generate Gherkin Testcases Only",
        "Generate Selenium Testcases Only"
    ]

    if "selected_workflow" not in st.session_state:
        st.session_state.selected_workflow = workflow_options[0]

    selected_workflow = st.selectbox("Select Workflow", workflow_options,
                                     key="selected_workflow",
                                     index=workflow_options.index(st.session_state.selected_workflow))

    # Update workflow in session state
    if selected_workflow != st.session_state.selected_workflow:
        st.session_state.selected_workflow = selected_workflow

    reset_button = st.button("🔄 Reset Conversation", key="reset_button")
    if reset_button:
        st.session_state.messages = []
        st.rerun()

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

# Text area for manual input
if not requirements_docs_content:
    requirements_docs_content = st.text_area(
        "Or enter requirements document text here:",
        height=200,
        placeholder="Enter requirements document text here...",
        help="Enter your requirements document content if you don't have a file to upload."
    )

# Main window
generate_button = st.button("Generate Testcases")

if generate_button and requirements_docs_content:
    # Add user's message to session state and display it
    user_message = f"**Requirements Document:**\n\n{requirements_docs_content[:500]}..." if len(
        requirements_docs_content) > 500 else requirements_docs_content
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    # Process with AI and get response
    with st.chat_message("assistant"):
        with st.spinner("Processing requirements..."):
            workflow = st.session_state.selected_workflow
            model = st.session_state.selected_model

            # Call the function from reqDocToTestcaseConvertor.py
            result = convert_requirements_to_testcases(
                requirements_docs_content,
                workflow_type=workflow,
                model=model
            )

            st.markdown(result)

            # Add the assistant's response to the chat history
            st.session_state.messages.append({"role": "assistant", "content": result})
elif generate_button and not requirements_docs_content:
    st.error("Please upload or enter requirements document content.")