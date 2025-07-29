# import streamlit as st
# from rag_module import RAGChatbot

# # Initialize the chatbot once
# @st.cache_resource
# def load_bot():
#     return RAGChatbot(data_directory="data")

# st.title("Chatbot (Marathi + MuRIL Powered)")
# bot = load_bot()

# # User input
# user_input = st.text_input("तुमचा प्रश्न विचाराः", placeholder="उदा. स्वॉट विश्लेषण म्हणजे काय?")

# if user_input:
#     with st.spinner("🤖 उत्तर शोधले जात आहे..."):
#         response = bot.chat(user_input)
#     st.markdown("### 🗣️ तुमचं उत्तर:")
#     st.success(response)

# # Optional: Show raw retrieved nodes (debug mode)
# if st.checkbox("📄 Show Retrieved Nodes (Debug Mode)"):
#     response = bot.query_engine.query(user_input)
#     for i, node in enumerate(response.source_nodes):
#         st.markdown(f"#### 🔹 Node {i+1} (Score: {node.score:.3f})")
#         st.write(node.text[:500])
import os
import streamlit as st
from dotenv import load_dotenv

from rag_muril import RAGChatbot as MurilChatbot
from rag_gemma import RAGChatbot as GemmaChatbot

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Streamlit page setup
st.set_page_config(page_title="🧠 विचार-चित्रे", layout="wide")
st.title("🧠 विचार-चित्रे Chatbot (Marathi RAG)")

# Initialize session state
if "bot" not in st.session_state:
    st.session_state.bot = None
if "history" not in st.session_state:
    st.session_state.history = []
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "MuRIL"

# Sidebar config
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.radio("🔍 Select Model", ["MuRIL", "Gemma"])
st.session_state.model_choice = model_choice

# File uploader
uploaded_files = st.sidebar.file_uploader(
    "📁 Upload .txt, .md, or .tex files",
    type=["txt", "md", "tex"],
    accept_multiple_files=True
)

# Chatbot initialization
if st.sidebar.button("🔄 Initialize Chatbot"):
    with st.spinner("⚙️ Processing and initializing..."):
        data_folder = "data"
        os.makedirs(data_folder, exist_ok=True)

        for file in uploaded_files:
            with open(os.path.join(data_folder, file.name), "wb") as f:
                f.write(file.read())

        try:
            if model_choice == "Gemma":
                if not GROQ_API_KEY:
                    st.error("❌ GROQ_API_KEY missing from .env")
                    st.stop()
                st.session_state.bot = GemmaChatbot(data_directory=data_folder, groq_api_key=GROQ_API_KEY)

            elif model_choice == "MuRIL":
                if not HUGGINGFACE_TOKEN:
                    st.error("❌ HUGGINGFACE_TOKEN missing from .env")
                    st.stop()
                st.session_state.bot = MurilChatbot(data_directory=data_folder)

            st.session_state.history.clear()
            st.success(f"✅ {model_choice} chatbot ready!")

        except Exception as e:
            st.error(f"❌ Initialization failed: {e}")

# Chat interface
if st.session_state.bot:
    user_input = st.text_input("✍️ तुमचा प्रश्न:", key="user_input")

    if st.button("📨 Send") and user_input:
        with st.spinner("🤖 Thinking..."):
            try:
                # Unified get_response for both models
                result = st.session_state.bot.get_response(user_input)
                answer = result.get("answer", "")
                context = result.get("context", "")

                # Save and display chat
                st.session_state.history.append(("🧑‍💻 तुम्ही", user_input))
                st.session_state.history.append(("🤖 बॉट", answer))

                st.markdown(f"**🤖 बॉटचे उत्तर:** {answer}")
                if context:
                    st.markdown("🧩 **Retrieved Context:**")
                    st.code(context)

            except Exception as e:
                st.error(f"❌ Error: {e}")

# Display chat history
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Chat History")
    for speaker, msg in st.session_state.history:
        st.markdown(f"**{speaker}:** {msg}")
