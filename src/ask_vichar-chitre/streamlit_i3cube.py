# import os
# from dotenv import load_dotenv
# import streamlit as st
# from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core.node_parser import SimpleNodeParser

# # ✅ Load environment variables
# load_dotenv()
# st.set_page_config(page_title="RAG Marathi Chatbot", layout="wide")
# st.title("🤖 Marathi RAG Chatbot (I3Cube)")

# st.markdown("Ask questions in **Marathi**, and the bot will retrieve answers from your documents.")

# # ✅ Disable LLM for retrieval-only mode
# Settings.llm = None
# Settings.embed_model = HuggingFaceEmbedding(
#     model_name="l3cube-pune/marathi-sentence-bert-nli"
# )

# # ✅ Cache the RAG system for faster reload
# @st.cache_resource(show_spinner=True)
# def load_rag(data_directory="./data"):
#     if not os.path.exists(data_directory) or len(os.listdir(data_directory)) == 0:
#         st.error(f"No files found in `{data_directory}`. Please add `.txt` or `.tex` files.")
#         return None

#     st.info(f"📂 Loading documents from `{data_directory}` ...")
#     documents = SimpleDirectoryReader(
#         data_directory,
#         recursive=True,
#         filename_as_id=True,
#         required_exts=[".txt", ".tex"]
#     ).load_data()
#     st.success(f"✅ Loaded {len(documents)} documents!")

#     # Parse documents into chunks
#     parser = SimpleNodeParser.from_defaults(chunk_size=300, chunk_overlap=50)
#     nodes = parser.get_nodes_from_documents(documents)

#     # Create vector index
#     index = VectorStoreIndex(nodes)

#     # Return query engine with top 3 retrievals
#     return index.as_query_engine(
#         similarity_top_k=3,
#         llm=None,
#         response_mode="no_text"
#     )

# # ✅ Initialize RAG
# query_engine = load_rag("./data")

# if query_engine:
#     # ✅ Input Box
#     question = st.text_input("❓ तुमचा प्रश्न विचाराः", placeholder="उदा: स्वॉट विश्लेषण म्हणजे काय?")
    
#     if question:
#         with st.spinner("🔍 शोधत आहे..."):
#             response = query_engine.query(question)

#             if not response.source_nodes:
#                 st.warning("माफ करा, संबंधित माहिती सापडली नाही.")
#             else:
#                 st.subheader("✅ उत्तर:")
#                 for i, node in enumerate(response.source_nodes, start=1):
#                     score = round(node.score or 0, 3)
#                     text = node.node.text.strip().replace("\n", " ")
#                     short_text = text[:300] + "..." if len(text) > 300 else text
#                     st.markdown(f"**🔹 [Score: {score}]** {short_text}")
import os
from dotenv import load_dotenv
import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser

# ✅ Load environment variables
load_dotenv()
st.set_page_config(page_title="RAG Marathi Chatbot", layout="wide")

# ✅ Inject Custom CSS
with open("static/style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# ✅ Disable LLM
Settings.llm = None
Settings.embed_model = HuggingFaceEmbedding(
    model_name="l3cube-pune/marathi-sentence-bert-nli"
)

# ✅ Cache RAG
@st.cache_resource(show_spinner=True)
def load_rag(data_directory="./data"):
    if not os.path.exists(data_directory) or len(os.listdir(data_directory)) == 0:
        st.error(f"No files found in `{data_directory}`.")
        return None

    documents = SimpleDirectoryReader(
        data_directory,
        recursive=True,
        filename_as_id=True,
        required_exts=[".txt", ".tex"]
    ).load_data()

    parser = SimpleNodeParser.from_defaults(chunk_size=300, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)

    return index.as_query_engine(similarity_top_k=3, llm=None, response_mode="no_text")

query_engine = load_rag("./data")

# ✅ Custom Chat UI
st.markdown("<h1>🤖 Marathi RAG Chatbot (I3Cube)</h1>", unsafe_allow_html=True)
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

question = st.text_input("तुमचा प्रश्न:", placeholder="उदा: स्वॉट विश्लेषण म्हणजे काय?")

submit = st.button("🔍 शोधा")

if submit and query_engine:
    response = query_engine.query(question)

    if not response.source_nodes:
        st.markdown('<div class="answer-box">❌ माफ करा, संबंधित माहिती सापडली नाही.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        for i, node in enumerate(response.source_nodes, start=1):
            score = round(node.score or 0, 3)
            text = node.node.text.strip().replace("\n", " ")
            short_text = text[:300] + "..." if len(text) > 300 else text
            st.markdown(f'<div class="answer-item"><b>Score: {score}</b> — {short_text}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
