import streamlit as st
import os
from rag import RAGChatbot
from fine_tune import FineTuner
import tempfile
import shutil

# Page configuration
st.set_page_config(
    page_title="Ask Vichar-Chitre Chatbot",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if 'rag_chatbot' not in st.session_state:
    st.session_state.rag_chatbot = None
if 'fine_tuner' not in st.session_state:
    st.session_state.fine_tuner = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False

# Title and description
st.title("🧠 Ask Vichar-Chitre Chatbot")
st.markdown("### मानसिक मॉडेल्सवर आधारित मराठी चॅटबॉट")
st.markdown("Mental Models chatbot in Marathi - Ask questions about cognitive biases, decision-making frameworks, and thinking patterns.")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    groq_api_key = st.text_input("Groq API Key", type="password", 
                                help="Enter your Groq API key to use Gemma models")
    
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    
    # Model selection
    model_type = st.selectbox(
        "Select Model Type",
        ["RAG with Raw Gemma", "RAG with Fine-tuned Gemma"],
        help="Choose between raw Gemma model or fine-tuned version"
    )
    
    st.divider()
    
    # Data upload section
    st.header("📁 Data Upload")
    uploaded_files = st.file_uploader(
        "Upload Mental Models Data Files",
        type=['txt', 'md', 'json'],
        accept_multiple_files=True,
        help="Upload text files containing mental models descriptions in Marathi"
    )
    
    if uploaded_files and groq_api_key:
        if st.button("🔄 Process Data & Initialize Chatbot"):
            with st.spinner("Processing uploaded files..."):
                try:
                    # Create temporary directory for uploaded files
                    temp_dir = tempfile.mkdtemp()
                    
                    # Save uploaded files
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    # Initialize RAG chatbot
                    st.session_state.rag_chatbot = RAGChatbot(
                        data_directory=temp_dir,
                        groq_api_key=groq_api_key
                    )
                    
                    # Initialize fine-tuner if needed
                    if model_type == "RAG with Fine-tuned Gemma":
                        st.session_state.fine_tuner = FineTuner(
                            data_directory=temp_dir
                        )
                    
                    st.session_state.data_uploaded = True
                    st.success("✅ Data processed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing data: {str(e)}")
    
    st.divider()
    
    # Fine-tuning section
    if model_type == "RAG with Fine-tuned Gemma" and st.session_state.data_uploaded:
        st.header("🎯 Fine-tuning")
        if st.button("🚀 Start Fine-tuning"):
            if st.session_state.fine_tuner:
                with st.spinner("Fine-tuning model... This may take a while."):
                    try:
                        st.session_state.fine_tuner.fine_tune_model()
                        st.success("✅ Model fine-tuned successfully!")
                    except Exception as e:
                        st.error(f"❌ Fine-tuning error: {str(e)}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main chat interface
if not groq_api_key:
    st.warning("⚠️ Please enter your Groq API key in the sidebar to get started.")
elif not st.session_state.data_uploaded:
    st.info("📁 Please upload data files and initialize the chatbot using the sidebar.")
else:
    # Chat interface
    st.header("💬 Chat Interface")
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**👤 You:** {question}")
            st.markdown(f"**🤖 Vichar-Chitre:** {answer}")
            st.divider()
    
    # Input for new question
    user_question = st.text_input(
        "आपला प्रश्न मराठीत विचारा / Ask your question in Marathi:",
        placeholder="उदा: Sunk cost fallacy या mental model ला मराठीत काय म्हणातात आणि त्याचे उदाहरण द्या",
        key="user_input"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        ask_button = st.button("🚀 Ask Question", type="primary")
    
    with col2:
        if st.button("📝 Example Questions"):
            st.info("""
            **Example Questions:**
            - Sunk cost fallacy या mental model ला मराठीत काय म्हणातात आणि त्याचे उदाहरण द्या
            - Confirmation bias बद्दल मराठीत सांगा
            - Decision making मध्ये कोणते mental models वापरावे?
            - Anchoring bias म्हणजे काय?
            """)
    
    if ask_button and user_question:
        if st.session_state.rag_chatbot:
            with st.spinner("Thinking... विचार करत आहे..."):
                try:
                    # Get response based on model type
                    if model_type == "RAG with Fine-tuned Gemma" and st.session_state.fine_tuner:
                        # Use fine-tuned model if available
                        response = st.session_state.rag_chatbot.get_response_with_finetuned(
                            user_question, 
                            st.session_state.fine_tuner.model
                        )
                    else:
                        # Use raw model with RAG
                        response = st.session_state.rag_chatbot.get_response(user_question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((user_question, response))
                    
                    # Display the new response
                    with st.container():
                        st.markdown(f"**👤 You:** {user_question}")
                        st.markdown(f"**🤖 Vichar-Chitre:** {response}")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
        else:
            st.error("❌ Chatbot not initialized. Please upload data first.")

# Footer
with st.container():
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🧠 Ask Vichar-Chitre Chatbot | Powered by Gemma & LlamaIndex | Built with Streamlit
    </div>
    """, unsafe_allow_html=True)