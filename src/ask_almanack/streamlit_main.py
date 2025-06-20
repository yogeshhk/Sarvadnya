import streamlit as st
import pickle
import faiss
from config import FAISS_STORE_PKL, DOCS_INDEX
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Ask Docs (Groq )")

st.title("Ask Documents")
st.markdown("Ask anything from your ingested documents. Powered by Groq & LLaMA3.")

# Load vectorstore (FAISS index)
@st.cache_resource
def load_vectorstore():
    index = faiss.read_index(DOCS_INDEX)
    with open(FAISS_STORE_PKL, "rb") as f:
        store = pickle.load(f)
    store.index = index
    return store

store = load_vectorstore()

# Load Groq LLM
llm = ChatGroq(
    api_key="Groq_api_key",  # use os.getenv()
    model_name="llama3-70b-8192"
)

# Setup Retrieval QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=store.as_retriever())

# User Input
query = st.text_input("Ask your question:")

if query:
    with st.spinner("Thinking..."):
        result = qa.invoke({"query": query})
        st.markdown("###Answer")
        st.write(result['result'])
