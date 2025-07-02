import streamlit as st
import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import CSVLoader, UnstructuredHTMLLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load .env variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE", "https://api.groq.com/openai/v1")
llm_model = os.getenv("OPENAI_API_MODEL", "llama3-70b-8192")

# Streamlit app setup
st.set_page_config(page_title="GST Query Bot", layout="centered")
st.title("🧾 GST FAQs Bot")
st.markdown("Ask your questions about **Goods and Services Tax (GST)** and get instant answers powered by **LLaMA 3 on Groq**.")

# Prompt template for GST domain
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a highly knowledgeable assistant in Indian Goods and Services Tax (GST).
Always answer only GST-related queries. If the question is outside GST, politely refuse to answer.

### Context:
{context}

### Question:
{question}

### Answer:
"""
)

@st.cache_resource(show_spinner="Setting up the knowledge base...")
def build_QnA_chain():
    # Load documents
    csv_docs = CSVLoader(file_path="./data/nlp_faq_engine_faqs.csv").load()
    html_docs = UnstructuredHTMLLoader(file_path="./data/cbic-gst_gov_in_fgaq.html").load()
    documents = csv_docs + html_docs

    # Vector store with embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()

    # LLaMA 3 on Groq via OpenAI-compatible endpoint
    llm = ChatOpenAI(
        model=llm_model,
        temperature=0.0,
    )

    # Retrieval QA chain with prompt
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )
    return chain

# Load the chain once
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = build_QnA_chain()

# Question form
with st.form("gst_query_form"):
    question = st.text_area(" Ask a GST question", height=140)
    submitted = st.form_submit_button("🔍 Get Answer")
    if submitted and question.strip():
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.qa_chain.run(question)
                st.success(" Answer:")
                st.write(response)
            except Exception as e:
                st.error("Something went wrong while generating the answer.")
                st.exception(e)

st.markdown("---")
st.caption("Built with using LLaMA 3 on Groq, LangChain, and Streamlit.")
